"""
Rutas de gestión de canales:
  GET    /api/canales         — lista todos los canales
  POST   /api/canales         — crea un nuevo canal
  PATCH  /api/canales/{id}    — renombra o edita un canal existente
  DELETE /api/canales/{id}    — elimina un canal
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.routes.ws import manager as ws_manager

router = APIRouter(prefix="/api/canales")


class CrearCanalRequest(BaseModel):
    nombre:   str = Field(..., min_length=1, max_length=64)
    peer_url: str = Field(..., min_length=1,
                          description="URL HTTP (ej: http://192.168.1.50:8000) "
                                      "o dirección LoRa (ej: 2)")
    tipo:     str = Field("http", pattern="^(http|lora)$")


class ActualizarCanalRequest(BaseModel):
    nombre:   Optional[str] = Field(None, min_length=1, max_length=64)
    peer_url: Optional[str] = Field(None, min_length=1)


@router.get("")
async def listar_canales(request: Request):
    gestor = request.app.state.gestor_canales
    return [c.to_dict() for c in gestor.listar()]


@router.post("", status_code=201)
async def crear_canal(body: CrearCanalRequest, request: Request):
    gestor = request.app.state.gestor_canales
    canal  = await gestor.crear(nombre=body.nombre, peer_url=body.peer_url, tipo=body.tipo)
    return canal.to_dict()


@router.patch("/{canal_id}")
async def actualizar_canal(canal_id: str, body: ActualizarCanalRequest, request: Request):
    gestor = request.app.state.gestor_canales
    canal  = await gestor.actualizar(canal_id, nombre=body.nombre, peer_url=body.peer_url)
    if not canal:
        raise HTTPException(status_code=404, detail=f"Canal '{canal_id}' no encontrado")
    await ws_manager.broadcast("canales_actualizados", canal.to_dict())
    return canal.to_dict()


@router.delete("/{canal_id}", status_code=204)
async def eliminar_canal(canal_id: str, request: Request):
    gestor = request.app.state.gestor_canales
    if not await gestor.eliminar(canal_id):
        raise HTTPException(status_code=404, detail=f"Canal '{canal_id}' no encontrado")
