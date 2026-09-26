"""
Medios de comunicación (LoRa, RF, Bluetooth, serie, red...):
  GET    /api/medios/tipos            — catálogo de tipos y sus campos (genera el formulario)
  GET    /api/medios                  — medios configurados con su estado
  POST   /api/medios                  — crea un medio
  PATCH  /api/medios/{id}             — edita nombre / parámetros / habilitado
  DELETE /api/medios/{id}             — elimina (el incorporado no se puede borrar)
  POST   /api/medios/{id}/conectar    — (re)conecta
  POST   /api/medios/{id}/desconectar — desconecta
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.medios import CATALOGO

router = APIRouter(prefix="/api/medios")


class CrearMedioRequest(BaseModel):
    tipo:   str = Field(..., min_length=1)
    nombre: str = Field("", max_length=64)
    params: dict = Field(default_factory=dict)


class ActualizarMedioRequest(BaseModel):
    nombre:  Optional[str] = Field(None, max_length=64)
    params:  Optional[dict] = None
    enabled: Optional[bool] = None


def _mm(request: Request):
    return request.app.state.medios


@router.get("/tipos")
async def tipos():
    return [{"tipo": k, **v} for k, v in CATALOGO.items()]


@router.get("")
async def listar(request: Request):
    return [m.to_dict() for m in _mm(request).listar()]


@router.post("", status_code=201)
async def crear(body: CrearMedioRequest, request: Request):
    try:
        m = await _mm(request).crear(body.tipo, body.nombre, body.params)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return m.to_dict()


@router.patch("/{medio_id}")
async def actualizar(medio_id: str, body: ActualizarMedioRequest, request: Request):
    m = await _mm(request).actualizar(medio_id, body.nombre, body.params, body.enabled)
    if not m:
        raise HTTPException(status_code=404, detail="Medio no encontrado")
    return m.to_dict()


@router.delete("/{medio_id}", status_code=204)
async def eliminar(medio_id: str, request: Request):
    if not await _mm(request).eliminar(medio_id):
        raise HTTPException(status_code=404, detail="Medio no encontrado o no se puede eliminar")


@router.post("/{medio_id}/conectar")
async def conectar(medio_id: str, request: Request):
    m = await _mm(request).conectar(medio_id)
    if not m:
        raise HTTPException(status_code=404, detail="Medio no encontrado")
    return m.to_dict()


@router.post("/{medio_id}/desconectar")
async def desconectar(medio_id: str, request: Request):
    m = await _mm(request).desconectar(medio_id)
    if not m:
        raise HTTPException(status_code=404, detail="Medio no encontrado")
    return m.to_dict()
