"""
Gestión de canales de comunicación con persistencia JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).resolve().parent.parent.parent / "data"
CANALES_FILE = DATA_DIR / "canales.json"


def _uid() -> str:
    return str(uuid.uuid4())[:8].upper()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Canal:
    def __init__(
        self,
        nombre: str,
        peer_url: str,
        canal_id: str | None = None,
        tipo: str = "http",
    ):
        self.id        = canal_id or _uid()
        self.nombre    = nombre
        self.peer_url  = peer_url.rstrip("/") if tipo == "http" else peer_url
        self.tipo      = tipo
        self.creado_en = _now()
        self.activo    = True

    def to_dict(self) -> dict:
        return {
            "id":        self.id,
            "nombre":    self.nombre,
            "peer_url":  self.peer_url,
            "tipo":      self.tipo,
            "creado_en": self.creado_en.isoformat(),
            "activo":    self.activo,
        }


class GestorCanales:
    def __init__(self):
        self._canales: Dict[str, Canal] = {}
        self._lock = asyncio.Lock()
        self._load()

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not CANALES_FILE.exists():
            return
        try:
            raw = json.loads(CANALES_FILE.read_text(encoding="utf-8"))
            for item in raw:
                try:
                    c = Canal(
                        nombre=item["nombre"],
                        peer_url=item["peer_url"],
                        canal_id=item["id"],
                        tipo=item.get("tipo", "http"),
                    )
                    c.creado_en = datetime.fromisoformat(item["creado_en"])
                    c.activo    = item.get("activo", True)
                    self._canales[c.id] = c
                except Exception:
                    pass
            log.info("Canales: %d cargados desde disco", len(self._canales))
        except Exception as exc:
            log.warning("Canales: no se pudo cargar %s: %s", CANALES_FILE, exc)

    def _save(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            items = [c.to_dict() for c in self._canales.values()]
            CANALES_FILE.write_text(
                json.dumps(items, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Canales: no se pudo guardar: %s", exc)

    # ── Inicialización ────────────────────────────────────────────────────────

    def inicializar_desde_env(self, peer_url: Optional[str], node_id: str = "") -> None:
        """Crea el canal 'default' desde variable de entorno si no existe ya."""
        if not peer_url:
            return
        # No duplicar si ya está guardado en disco
        for c in self._canales.values():
            if c.peer_url.rstrip("/") == peer_url.rstrip("/"):
                return
        canal = Canal(nombre="default", peer_url=peer_url, canal_id="DEFAULT0")
        self._canales[canal.id] = canal
        self._save()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def crear(self, nombre: str, peer_url: str, tipo: str = "http") -> Canal:
        async with self._lock:
            canal = Canal(nombre=nombre, peer_url=peer_url, tipo=tipo)
            self._canales[canal.id] = canal
            self._save()
            return canal

    async def actualizar(
        self,
        canal_id: str,
        nombre: Optional[str] = None,
        peer_url: Optional[str] = None,
    ) -> Optional[Canal]:
        async with self._lock:
            c = self._canales.get(canal_id)
            if not c:
                return None
            if nombre is not None:
                c.nombre = nombre.strip()
            if peer_url is not None:
                c.peer_url = peer_url.rstrip("/") if c.tipo == "http" else peer_url
            self._save()
            return c

    async def eliminar(self, canal_id: str) -> bool:
        async with self._lock:
            if canal_id in self._canales:
                del self._canales[canal_id]
                self._save()
                return True
        return False

    def obtener(self, canal_id: str) -> Optional[Canal]:
        return self._canales.get(canal_id)

    def listar(self) -> List[Canal]:
        return list(self._canales.values())

    def primero(self) -> Optional[Canal]:
        canales = list(self._canales.values())
        return canales[0] if canales else None

    def peer_url_de(self, canal_id: Optional[str]) -> Optional[str]:
        if canal_id:
            c = self.obtener(canal_id)
            return c.peer_url if c else None
        primero = self.primero()
        return primero.peer_url if primero else None
