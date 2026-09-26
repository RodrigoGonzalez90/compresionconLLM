"""Nombre de este nodo, elegido desde la web y persistido en data/nodo.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

NODO_FILE = Path(__file__).resolve().parent.parent / "data" / "nodo.json"


def cargar() -> Optional[str]:
    try:
        if NODO_FILE.exists():
            nombre = json.loads(NODO_FILE.read_text(encoding="utf-8")).get("nombre", "").strip()
            return nombre or None
    except Exception as exc:
        log.warning("Nodo: no se pudo leer %s: %s", NODO_FILE, exc)
    return None


def guardar(nombre: str) -> None:
    NODO_FILE.parent.mkdir(parents=True, exist_ok=True)
    NODO_FILE.write_text(json.dumps({"nombre": nombre}, ensure_ascii=False), encoding="utf-8")
