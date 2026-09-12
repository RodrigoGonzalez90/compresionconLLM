from __future__ import annotations

import re
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api")

# Cargado una sola vez al importar el módulo
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker(language="es")
    _available = True
except Exception:
    _spell = None
    _available = False

# Patrones que nunca se corrigen
_SKIP = re.compile(
    r'\d'           # contiene número: SF10, 34.6037, 10.0.1.7
    r'|^[A-Z]{2,}$' # todo mayúsculas: RSSI, ACK, GPS, CRC
    r'|[-/\\]'      # tiene guión o slash: nodo-beta
    r'|^.{1,2}$'    # muy corta: ok, al, de…
)


class SpellRequest(BaseModel):
    texto: str


class Correccion(BaseModel):
    original: str
    corrected: str
    start: int
    end: int


class SpellResponse(BaseModel):
    correcciones: List[Correccion]
    disponible: bool


@router.post("/spellcheck", response_model=SpellResponse)
async def spellcheck(body: SpellRequest) -> SpellResponse:
    if not _available or not body.texto.strip():
        return SpellResponse(correcciones=[], disponible=_available)

    correcciones: List[Correccion] = []

    for m in re.finditer(r"\b[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+\b", body.texto):
        word = m.group()
        if _SKIP.search(word):
            continue
        # pyspellchecker trabaja en minúsculas internamente
        unknown = _spell.unknown([word.lower()])
        if not unknown:
            continue
        best = _spell.correction(word.lower())
        if best and best != word.lower():
            # Preservar capitalización inicial si la tenía
            if word[0].isupper():
                best = best.capitalize()
            correcciones.append(Correccion(
                original=word,
                corrected=best,
                start=m.start(),
                end=m.end(),
            ))

    return SpellResponse(correcciones=correcciones, disponible=True)
