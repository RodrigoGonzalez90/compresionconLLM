"""
Identidad de este nodo:
  GET /api/nodo   — nombre actual, si ya fue configurado desde la web e ID de instancia
  PUT /api/nodo   — cambia el nombre del nodo (se guarda y se propaga a los pares)
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from api.routes.ws import manager as ws_manager
from src import nodo as nodo_store
from src.sync import instancia_id

router = APIRouter(prefix="/api/nodo")


class NodoRequest(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=40)


@router.get("")
async def obtener(request: Request):
    st = request.app.state
    return {"nombre": st.node_id, "configurado": st.nodo_configurado, "instancia": instancia_id()}


@router.put("")
async def cambiar(body: NodoRequest, request: Request):
    st = request.app.state
    nombre = " ".join(body.nombre.split())
    st.node_id = nombre
    st.nodo_configurado = True
    nodo_store.guardar(nombre)
    await ws_manager.broadcast("nodo_actualizado", {"node_id": nombre})
    return {"nombre": nombre, "configurado": True, "instancia": instancia_id()}
