"""
Pares que nos escribieron pero cuya dirección no se pudo determinar
(Docker oculta la IP de origen y el emisor no declaró una URL alcanzable):
  GET  /api/pares/pendientes                    — lista de pares sin dirección
  POST /api/pares/pendientes/{instancia}/direccion — el usuario indica su IP y se crea el canal
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api.peers import sondear, sync_en_segundo_plano
from api.routes.ws import manager as ws_manager
from src.sync import instancia_id

router = APIRouter(prefix="/api/pares")


class DireccionRequest(BaseModel):
    host: str = Field(..., min_length=3, max_length=200, description="IP o nombre; opcional http:// y :puerto")


@router.get("/pendientes")
async def pendientes(request: Request):
    return [{"instancia": k, **v} for k, v in request.app.state.pares_pendientes.items()]


@router.post("/pendientes/{instancia}/direccion", status_code=201)
async def resolver(instancia: str, body: DireccionRequest, request: Request):
    st = request.app.state
    par = st.pares_pendientes.get(instancia)
    if not par:
        raise HTTPException(status_code=404, detail="Ese par ya no está pendiente")

    host = body.host.strip().rstrip("/")
    if "://" not in host:
        host = f"http://{host}"
    if host.count(":") < 2:               # sin puerto explícito: usar el que declaró el par
        host = f"{host}:{par['puerto']}"

    remoto = await sondear(host)
    if remoto == instancia_id():
        raise HTTPException(status_code=422, detail="Esa dirección es este mismo nodo")
    if remoto and remoto != instancia:
        raise HTTPException(status_code=422, detail="Esa dirección responde otro nodo distinto")

    gestor = st.gestor_canales
    canal = gestor.por_url(host)
    if canal is None:
        canal = await gestor.crear(par["nombre"] or f"par-{instancia[:6]}", host)
        gestor.marcar_auto(canal)
    gestor.vincular(canal, instancia)
    st.pares_pendientes.pop(instancia, None)
    await ws_manager.broadcast("canales_actualizados", {"nuevo": canal.to_dict()})
    sync_en_segundo_plano(st, host)
    return {**canal.to_dict(), "verificado": remoto == instancia}
