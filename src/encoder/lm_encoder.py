"""
Compresor lossless basado en rangos de modelo de lenguaje.

Cada token del texto se codifica por su posición (rango) en la distribución
de probabilidad del LM dado el contexto previo. El receptor, con el mismo LM
y el mismo contexto, reconstruye exactamente el token original.

Esquema de bits por rango:
  Rango 0       →  0                          (1 bit)
  Rango 1-3     →  10 rr                      (4 bits)
  Rango 4-15    →  110 rrrr                   (7 bits)
  Rango 16-63   →  1110 rrrrrr                (10 bits)
  OOV compacto  →  1111 1 [id: id_bits bits]  (5 + id_bits bits, ej. 23 para vocab ~152k)
  OOV UTF-8     →  1111 0 LLLLLLLL [L bytes]  (13 + 8L bits, fallback Ollama)

Header del paquete:
  0x4C + uint32(CRC32) + uint16(N) + bits  →  lossless (N = número de tokens)
  0x52 + utf8                               →  raw UTF-8 (fallback adaptativo)

El CRC32 permite al decoder verificar que la reconstrucción fue exacta.
"""

from __future__ import annotations

import logging
import math
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.llm_backend import LLMBackend, TOP_K, normalize_token

log = logging.getLogger(__name__)

HDR_LOSSLESS = b"\x4C"   # 'L'
HDR_RAW      = b"\x52"   # 'R'

# Umbral dinámico de gap: se escala linealmente con la entropía de Shannon del top-20.
# Baja entropía (modelo seguro) → umbral pequeño → más ranks, mejor compresión.
# Alta entropía (modelo inseguro) → umbral grande → más OOV, reconstrucción más segura.
GAP_MIN = 0.001          # umbral cuando el modelo es casi determinista
GAP_MAX = 0.1            # umbral cuando la distribución es plana
_H_MAX  = math.log(20)   # entropía máxima posible de una distribución uniforme sobre 20 tokens


def _dynamic_gap(top: List[Tuple[str, float]]) -> float:
    """Gap mínimo de logprob requerido, calculado a partir de la entropía del top-20."""
    probs = [math.exp(lp) for _, lp in top]
    s = sum(probs)
    if s <= 0:
        return GAP_MAX
    h = -sum((p / s) * math.log(p / s) for p in probs if p > 0)
    t = min(h / _H_MAX, 1.0)   # t ∈ [0, 1]
    return GAP_MIN + t * (GAP_MAX - GAP_MIN)


@dataclass
class PaqueteLossless:
    datos: bytes
    texto_original: str
    rangos: List[int]
    bytes_originales: int
    bytes_transmitidos: int
    modo: str = "lossless"
    encode_ms: float = 0.0

    @property
    def n_tokens(self) -> int:
        return len(self.rangos)

    @property
    def tokens_count(self) -> int:
        return self.n_tokens

    @property
    def porcentaje_ahorro(self) -> float:
        if self.bytes_originales == 0:
            return 0.0
        return (1.0 - self.bytes_transmitidos / self.bytes_originales) * 100.0

    @property
    def esqueleto(self) -> str:
        return f"[lossless·{self.n_tokens}tok·{self.modo}]"

    @property
    def ids_tokens(self) -> List[int]:
        return self.rangos

    @property
    def rank_distribution(self) -> dict:
        """Porcentaje de tokens en cada bucket de rank (para diagnóstico)."""
        if not self.rangos:
            return {}
        n = len(self.rangos)
        counts = {"r0": 0, "r1_3": 0, "r4_15": 0, "r16_63": 0, "oov": 0}
        for r in self.rangos:
            if r == 0:
                counts["r0"] += 1
            elif r <= 3:
                counts["r1_3"] += 1
            elif r <= 15:
                counts["r4_15"] += 1
            elif r <= 63:
                counts["r16_63"] += 1
            else:
                counts["oov"] += 1
        return {k: round(100 * v / n, 1) for k, v in counts.items()}


# ─── Codificación de rangos ────────────────────────────────────────────────────

def _encode_rank_bits(rank: int) -> List[int]:
    if rank == 0:
        return [0]
    if 1 <= rank <= 3:
        r = rank - 1
        return [1, 0, (r >> 1) & 1, r & 1]
    if 4 <= rank <= 15:
        r = rank - 4
        return [1, 1, 0, (r >> 3) & 1, (r >> 2) & 1, (r >> 1) & 1, r & 1]
    # 16-63
    r = rank - 16
    return [1, 1, 1, 0,
            (r >> 5) & 1, (r >> 4) & 1, (r >> 3) & 1,
            (r >> 2) & 1, (r >> 1) & 1,  r       & 1]


def _encode_oov(tok_str: str, oov_id: int, id_bits: int) -> List[int]:
    """
    Codifica un token OOV.
    Compacto (1111 1 [id]):  cuando id_bits > 0 y el ID es válido.
    UTF-8   (1111 0 L [bytes]): fallback para Ollama o IDs no disponibles.
    """
    if id_bits > 0 and oov_id >= 0:
        bits: List[int] = [1, 1, 1, 1, 1]   # prefijo + discriminador=1
        for i in range(id_bits - 1, -1, -1):
            bits.append((oov_id >> i) & 1)
        return bits
    # UTF-8 fallback
    b = tok_str.encode("utf-8")
    L = len(b)
    bits = [1, 1, 1, 1, 0]                  # prefijo + discriminador=0
    for i in range(7, -1, -1):
        bits.append((L >> i) & 1)
    for byte in b:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i : i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


# ─── LMEncoder ────────────────────────────────────────────────────────────────

class LMEncoder:
    """
    Compresor lossless usando la distribución del LM para codificar rangos.
    Acepta cualquier LLMBackend (Ollama, LlamaCpp, etc.).
    """

    def __init__(self, backend: LLMBackend):
        self.backend = backend
        vs = backend.vocab_size
        self._id_bits = vs.bit_length() if vs > 0 else 0

    def _next_token_from_hf(self, text_remaining: str) -> str:
        return self.backend.next_token(text_remaining)

    def comprimir(self, texto: str) -> PaqueteLossless:
        t0 = time.perf_counter()
        bytes_orig = len(texto.encode("utf-8"))
        self.backend.reset()  # KV cache limpio → determinismo

        bits: List[int] = []
        rangos: List[int] = []
        context = ""
        oov_count = 0

        # Tokenizar con IDs → boundaries canónicos + IDs para OOV compacto
        if hasattr(self.backend, "tokenize_with_ids"):
            tok_pairs = self.backend.tokenize_with_ids(texto)
        elif hasattr(self.backend, "tokenize_text"):
            tok_pairs = [(t, -1) for t in self.backend.tokenize_text(texto)]
        else:
            tok_pairs = [(c, -1) for c in texto]

        # ── Teacher forcing: una sola forward pass para todo el texto ──────────
        # Calcula distribuciones para todas las posiciones en paralelo.
        # Si falla (backend sin soporte o error), cae al camino secuencial.
        batch_tops: Optional[List] = None
        if hasattr(self.backend, "batch_top_tokens"):
            batch_tops = self.backend.batch_top_tokens(texto)
            if batch_tops is not None and len(batch_tops) < len(tok_pairs):
                log.debug("batch_top_tokens: desalineación (%d vs %d), usando secuencial",
                          len(batch_tops), len(tok_pairs))
                batch_tops = None
        # ──────────────────────────────────────────────────────────────────────

        for i, (tok, tok_id) in enumerate(tok_pairs):
            if not tok:
                continue
            top = batch_tops[i] if batch_tops is not None else self.backend.top_tokens(context)

            # Buscar el rank del token canónico en el top-K
            matched_rank: Optional[int] = None
            for rank, (tok_str, _) in enumerate(top):
                if normalize_token(tok_str) == tok:
                    matched_rank = rank
                    break

            # Si la ventaja sobre el siguiente candidato es demasiado pequeña,
            # los dos nodos podrían ver rankings distintos por no-determinismo
            # de KV cache. En ese caso preferimos OOV.
            if matched_rank is not None and matched_rank < len(top) - 1:
                gap       = top[matched_rank][1] - top[matched_rank + 1][1]
                gap_req   = _dynamic_gap(top)
                if gap < gap_req:
                    matched_rank = None

            if matched_rank is not None:
                bits.extend(_encode_rank_bits(matched_rank))
                rangos.append(matched_rank)
            else:
                bits.extend(_encode_oov(tok, tok_id, self._id_bits))
                rangos.append(TOP_K)
                oov_count += 1

            context += tok

        if rangos:
            log.debug(
                "comprimir: %d tokens, %d OOV (%.0f%%)",
                len(rangos), oov_count, 100 * oov_count / len(rangos),
            )

        payload = _bits_to_bytes(list(bits))
        n_tok = len(rangos)
        crc32 = zlib.crc32(texto.encode("utf-8")) & 0xFFFFFFFF
        datos_lossless = HDR_LOSSLESS + struct.pack(">IH", crc32, n_tok) + payload

        datos_raw = HDR_RAW + texto.encode("utf-8")
        if len(datos_raw) <= len(datos_lossless):
            datos = datos_raw
            modo  = "raw"
        else:
            datos = datos_lossless
            modo  = "lossless"

        encode_ms = (time.perf_counter() - t0) * 1000
        return PaqueteLossless(
            datos=datos,
            texto_original=texto,
            rangos=rangos,
            bytes_originales=bytes_orig,
            bytes_transmitidos=len(datos),
            modo=modo,
            encode_ms=encode_ms,
        )
