"""
Decodificador lossless: reconstruye el texto exacto a partir de rangos LM.

El receptor usa el mismo modelo y el mismo contexto inicial que el emisor,
por lo que las distribuciones de probabilidad en cada paso son idénticas.
Al recibir el rango k, toma el k-ésimo token más probable → texto exacto.

Esquema de decodificación de bits:
  0                        → rango 0
  10 rr                    → rango 1-3
  110 rrrr                 → rango 4-15
  1110 rrrrrr              → rango 16-63
  1111 1 [id: id_bits]     → token OOV compacto (vocab ID → string)
  1111 0 LLLLLLLL [L bytes] → token OOV UTF-8 (fallback Ollama)
"""

from __future__ import annotations

import logging
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Iterator, List, Tuple

from src.llm_backend import BLOCK_TOKENS, LLMBackend, TOP_K, normalize_token

log = logging.getLogger(__name__)

HDR_LOSSLESS = 0x4C
HDR_RAW      = 0x52


@dataclass
class MensajeDecodificado:
    texto_reconstruido: str
    esqueleto: str
    ids_tokens: List[int]
    confianza_modelo: float = 1.0
    modo: str = "lossless"
    crc_ok: bool = True
    decode_ms: float = 0.0


# ─── Iterador de bits ──────────────────────────────────────────────────────────

def _byte_iter(data: bytes) -> Iterator[int]:
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1


def _read_bits(it: Iterator[int], n: int) -> int:
    val = 0
    for _ in range(n):
        val = (val << 1) | next(it)
    return val


def _decode_rank(it: Iterator[int], id_bits: int = 0) -> Tuple[int, int, str]:
    """
    Retorna (rank, oov_id, oov_str).
    rank < TOP_K  → oov_id y oov_str vacíos.
    rank == TOP_K → oov_id >= 0 (compacto) o oov_str != "" (UTF-8 fallback).
    """
    b0 = next(it)
    if b0 == 0:
        return 0, -1, ""
    b1 = next(it)
    if b1 == 0:
        r = _read_bits(it, 2)
        return r + 1, -1, ""
    b2 = next(it)
    if b2 == 0:
        r = _read_bits(it, 4)
        return r + 4, -1, ""
    b3 = next(it)
    if b3 == 0:
        r = _read_bits(it, 6)   # 6 bits → 0-47 → ranks 16-63
        return r + 16, -1, ""
    # OOV: leer bit discriminador
    disc = next(it)
    if disc == 1 and id_bits > 0:
        # Compacto: ID del vocabulario
        oov_id = _read_bits(it, id_bits)
        return TOP_K, oov_id, ""
    # UTF-8 fallback
    L = _read_bits(it, 8)
    raw = bytes(_read_bits(it, 8) for _ in range(L))
    tok_str = raw.decode("utf-8", errors="replace")
    return TOP_K, -1, tok_str


# ─── LMDecoder ────────────────────────────────────────────────────────────────

class LMDecoder:
    """
    Decodificador lossless complementario a LMEncoder.
    Acepta cualquier LLMBackend (Ollama, LlamaCpp, etc.).
    """

    def __init__(self, backend: LLMBackend):
        self.backend = backend
        vs = backend.vocab_size
        self._id_bits = vs.bit_length() if vs > 0 else 0

    def _top_tokens(self, context: str) -> List[str]:
        """Retorna los TOP_K tokens más probables (strings limpios) dado el contexto."""
        raw = self.backend.top_tokens(context)
        return [normalize_token(tok) for tok, _ in raw]

    def decodificar(self, datos: bytes) -> MensajeDecodificado:
        t0 = time.perf_counter()
        if not datos:
            return MensajeDecodificado(texto_reconstruido="", esqueleto="", ids_tokens=[])

        hdr = datos[0]

        if hdr == HDR_RAW:
            texto = datos[1:].decode("utf-8", errors="replace")
            return MensajeDecodificado(
                texto_reconstruido=texto,
                esqueleto=texto,
                ids_tokens=[],
                modo="raw",
                decode_ms=(time.perf_counter() - t0) * 1000,
            )

        if hdr != HDR_LOSSLESS:
            texto = datos.decode("utf-8", errors="replace")
            return MensajeDecodificado(
                texto_reconstruido=texto,
                esqueleto=texto,
                ids_tokens=[],
                modo="legacy",
                decode_ms=(time.perf_counter() - t0) * 1000,
            )

        if len(datos) < 7:
            return MensajeDecodificado(
                texto_reconstruido="",
                esqueleto="[error: payload demasiado corto]",
                ids_tokens=[],
                modo="lossless_crc_error",
                crc_ok=False,
                decode_ms=(time.perf_counter() - t0) * 1000,
            )

        crc_expected = struct.unpack(">I", datos[1:5])[0]
        n_tokens     = struct.unpack(">H", datos[5:7])[0]
        payload      = datos[7:]
        self.backend.reset()  # KV cache limpio → mismo estado que el encoder

        it = _byte_iter(payload)
        raw    = bytearray()
        rangos: List[int] = []
        by_ids = hasattr(self.backend, "start_block")

        try:
            if by_ids:
                top: List[Tuple[int, float]] = []
                for k in range(n_tokens):
                    if k % BLOCK_TOKENS == 0:
                        top = self.backend.start_block()
                    rank, oov_id, oov_str = _decode_rank(it, self._id_bits)
                    rangos.append(rank)

                    if rank >= TOP_K:
                        tid = oov_id
                    else:
                        tid = top[min(rank, len(top) - 1)][0]

                    if tid >= 0:
                        raw += self.backend.id_to_bytes(tid)
                    else:
                        raw += oov_str.encode("utf-8")

                    last_in_block = (k + 1) % BLOCK_TOKENS == 0
                    if tid >= 0 and not last_in_block and k + 1 < n_tokens:
                        top = self.backend.advance(tid)
            else:
                context = ""
                for _ in range(n_tokens):
                    rank, oov_id, oov_str = _decode_rank(it, self._id_bits)
                    rangos.append(rank)

                    # SIEMPRE llamamos top_tokens para que el KV cache evolucione
                    # igual que en el encoder (que llama top_tokens para cada token).
                    top_s = self._top_tokens(context)

                    if rank >= TOP_K:
                        if oov_id >= 0:
                            tok = self.backend.id_to_token(oov_id) or ""
                        else:
                            tok = oov_str
                    elif rank < len(top_s):
                        tok = top_s[rank]
                    elif top_s:
                        tok = top_s[-1]
                    else:
                        tok = " "

                    context += tok
                    raw += tok.encode("utf-8")

        except StopIteration:
            pass

        texto = bytes(raw).decode("utf-8", errors="replace")
        crc_actual = zlib.crc32(bytes(raw)) & 0xFFFFFFFF
        crc_ok     = (crc_actual == crc_expected)

        if not crc_ok:
            log.warning(
                "CRC mismatch: esperado=%08X actual=%08X — no-determinismo en KV cache",
                crc_expected, crc_actual,
            )

        return MensajeDecodificado(
            texto_reconstruido=texto,
            esqueleto=f"[lossless·{n_tokens}tok]",
            ids_tokens=rangos,
            confianza_modelo=1.0,
            modo="lossless" if crc_ok else "lossless_crc_error",
            crc_ok=crc_ok,
            decode_ms=(time.perf_counter() - t0) * 1000,
        )
