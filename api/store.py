"""
Store de mensajes con persistencia JSON y soporte para eliminar.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .models import Mensaje

log = logging.getLogger(__name__)

DATA_DIR     = Path(__file__).resolve().parent.parent / "data"
MENSAJES_FILE = DATA_DIR / "mensajes.json"


class MensajeStore:
    def __init__(self, max_mensajes: int = 500):
        self._store: OrderedDict[str, Mensaje] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max  = max_mensajes
        self._load()

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not MENSAJES_FILE.exists():
            return
        try:
            raw  = json.loads(MENSAJES_FILE.read_text(encoding="utf-8"))
            for item in raw:
                try:
                    msg = Mensaje(**item)
                    self._store[msg.id] = msg
                except Exception:
                    pass
            log.info("Store: %d mensajes cargados desde disco", len(self._store))
        except Exception as exc:
            log.warning("Store: no se pudo cargar %s: %s", MENSAJES_FILE, exc)

    def _save(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            items = [json.loads(m.model_dump_json()) for m in self._store.values()]
            MENSAJES_FILE.write_text(
                json.dumps(items, ensure_ascii=False, indent=None),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Store: no se pudo guardar: %s", exc)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def guardar(self, msg: Mensaje) -> Mensaje:
        async with self._lock:
            if len(self._store) >= self._max:
                self._store.popitem(last=False)
            self._store[msg.id] = msg
            self._save()
        return msg

    async def actualizar(self, id: str, **kwargs) -> Optional[Mensaje]:
        async with self._lock:
            msg = self._store.get(id)
            if msg:
                for k, v in kwargs.items():
                    setattr(msg, k, v)
                self._save()
                return msg
        return None

    async def eliminar(self, id: str) -> bool:
        async with self._lock:
            if id in self._store:
                del self._store[id]
                self._save()
                return True
        return False

    async def limpiar(self) -> int:
        async with self._lock:
            n = len(self._store)
            self._store.clear()
            self._save()
        return n

    def obtener(self, id: str) -> Optional[Mensaje]:
        return self._store.get(id)

    def listar(self) -> list[Mensaje]:
        return list(reversed(list(self._store.values())))

    def stats(self) -> dict:
        msgs     = list(self._store.values())
        salientes = [m for m in msgs if m.direccion == "saliente"]
        entrantes = [m for m in msgs if m.direccion == "entrante"]
        ratios    = [m.ratio_pct for m in msgs if m.ratio_pct > 0]
        return {
            "total_enviados":    len(salientes),
            "total_recibidos":   len(entrantes),
            "compresion_media_pct": round(sum(ratios) / len(ratios), 1) if ratios else 0.0,
        }
