"""
Sincronización numérica entre nodos (handshake de canal).

Dos nodos con el mismo modelo pueden producir logits levemente distintos si
corren en arquitecturas o builds distintos (x86 vs ARM). El encoder codifica
por rank, así que una diferencia puede hacer que el receptor reconstruya mal.

Handshake:
  1. Cada nodo publica su perfil: modelo, formato, bloque, arch, versión y una
     huella (hash de la distribución top-K sobre un texto de calibración fijo).
  2. Huellas iguales   → modo "rapido": se comprime como siempre.
  3. Huellas distintas → se intercambian los vectores de calibración, ambos
     nodos calculan la misma discrepancia δ y un gap mínimo = SAFETY * 2 * δ.
     El encoder manda como OOV los tokens cuya ventaja sobre un vecino sea menor.
  4. Perfil incompatible (otro modelo/formato) → "incompatible": solo modo raw.
El resultado se persiste por par en data/peers.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.llm_backend import BLOCK_TOKENS, TOP_K

log = logging.getLogger(__name__)

FORMAT_VERSION = 1
SAFETY = 2.0            # margen sobre la discrepancia medida
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PEERS_FILE = DATA_DIR / "peers.json"

# Texto fijo: mezcla de prosa, números y puntuación para cubrir distintas entropías.
CALIB_TEXT = (
    "Hola, veo que el cielo es azul y las nubes son blancas. Mañana, a las 10:30, "
    "probaremos la comunicación por radio entre los dos nodos del proyecto. "
    "El mensaje 42 llegó sin errores: ¿qué opinas del resultado? Todo listo para continuar."
)


def discrepancia(a: dict, b: dict) -> float:
    """Máxima diferencia de logprob entre dos calibraciones (simétrica)."""
    n = min(len(a["pos"]), len(b["pos"]))
    delta = 0.0
    for k in range(n):
        A = dict(zip(a["pos"][k]["i"], a["pos"][k]["l"]))
        B = dict(zip(b["pos"][k]["i"], b["pos"][k]["l"]))
        min_a, min_b = min(A.values()), min(B.values())
        for tid, la in A.items():
            # ausente en el otro top-K: su logprob real es <= al mínimo de esa lista
            delta = max(delta, abs(la - B.get(tid, min_b)))
        for tid, lb in B.items():
            if tid not in A:
                delta = max(delta, abs(lb - min_a))
    return delta


class Sincronizador:
    def __init__(self, backend):
        self.backend = backend
        self._cal: Optional[dict] = None
        self._perfil: Optional[dict] = None
        self.peers: Dict[str, dict] = {}
        self.verificados: set = set()   # pares confirmados en esta sesión
        self._load()

    # ── Perfil y calibración locales ──────────────────────────────────────────

    def calibracion(self) -> dict:
        if self._cal is None:
            be = self.backend
            ids = [i for _, i in be.tokenize_with_ids(CALIB_TEXT)]
            top = be.start_block()
            pos = []
            for k, tid in enumerate(ids):
                pos.append({"i": [i for i, _ in top], "l": [round(lp, 4) for _, lp in top]})
                if k + 1 < len(ids):
                    top = be.advance(tid)
            self._cal = {"ids_texto": ids, "pos": pos}
        return self._cal

    def perfil(self) -> dict:
        if self._perfil is None:
            cal = self.calibracion()
            orden = json.dumps([p["i"] for p in cal["pos"]], separators=(",", ":"))
            try:
                import llama_cpp
                llama_ver = llama_cpp.__version__
            except Exception:
                llama_ver = "?"
            self._perfil = {
                "fmt":    FORMAT_VERSION,
                "modelo": self.backend.model_id,
                "block":  BLOCK_TOKENS,
                "top_k":  TOP_K,
                "arch":   platform.machine(),
                "llama":  llama_ver,
                "fp":     hashlib.sha256(orden.encode()).hexdigest()[:16],
            }
        return self._perfil

    # ── Evaluación de un par ──────────────────────────────────────────────────

    def compatible(self, remoto: dict) -> bool:
        local = self.perfil()
        return all(local[k] == remoto.get(k) for k in ("fmt", "modelo", "block", "top_k"))

    def evaluar_con_calibracion(self, remoto: dict, cal_remota: dict) -> dict:
        from src.encoder.lm_encoder import GAP_MIN
        if cal_remota.get("ids_texto") != self.calibracion()["ids_texto"]:
            return {"modo": "incompatible", "motivo": "tokenizador distinto"}
        delta = discrepancia(self.calibracion(), cal_remota)
        gap_min = max(GAP_MIN, SAFETY * 2 * delta)
        return {"modo": "compensado", "delta": round(delta, 5), "gap_min": round(gap_min, 4),
                **self.estimar_costo(gap_min)}

    def estimar_costo(self, gap_min: float) -> dict:
        """Costo relativo en bits de usar gap_min, medido sobre el texto de calibración."""
        from src.encoder.lm_encoder import bits_de_rank, rank_seguro
        cal = self.calibracion()
        id_bits = self.backend.vocab_size.bit_length()
        base = comp = 0
        for tid, p in zip(cal["ids_texto"], cal["pos"]):
            top = list(zip(p["i"], p["l"]))
            rank = next((r for r, (i, _) in enumerate(top) if i == tid), None)
            r_base = rank if rank is not None and rank_seguro(top, rank, None) else None
            r_comp = rank if rank is not None and rank_seguro(top, rank, gap_min) else None
            base += bits_de_rank(r_base, id_bits)
            comp += bits_de_rank(r_comp, id_bits)
        return {
            "costo_pct": round(100 * (comp - base) / base, 1) if base else 0.0,
            "muestra_tokens": len(cal["ids_texto"]),
        }

    def evaluar_rapido(self, remoto: dict) -> Optional[dict]:
        """Resultado sin intercambio de calibración, o None si hace falta."""
        if not self.compatible(remoto):
            return {"modo": "incompatible", "motivo": "modelo/formato distinto"}
        if remoto.get("fp") == self.perfil()["fp"]:
            return {"modo": "rapido", "delta": 0.0, "gap_min": None}
        return None

    # ── Registro persistente por par ──────────────────────────────────────────

    def registrar(self, clave: str, remoto: dict, resultado: dict) -> dict:
        rec = {
            **resultado,
            "fp_local":  self.perfil()["fp"],
            "fp_remoto": remoto.get("fp"),
            "arch_remota": remoto.get("arch"),
            "remoto":    remoto,
            "ts": time.time(),
        }
        self.peers[clave] = rec
        self.verificados.add(clave)
        self._save()
        return rec

    def vigente(self, clave: str, remoto: dict) -> Optional[dict]:
        """Registro guardado si sigue valiendo para los perfiles actuales."""
        rec = self.peers.get(clave)
        if rec and rec.get("fp_local") == self.perfil()["fp"] and rec.get("fp_remoto") == remoto.get("fp"):
            self.verificados.add(clave)
            return rec
        return None

    def _load(self) -> None:
        try:
            if PEERS_FILE.exists():
                self.peers = json.loads(PEERS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Sync: no se pudo cargar %s: %s", PEERS_FILE, exc)

    def _save(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            PEERS_FILE.write_text(json.dumps(self.peers, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            log.warning("Sync: no se pudo guardar: %s", exc)
