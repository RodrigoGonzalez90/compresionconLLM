"""
Handshake de sincronización numérica entre nodos:
  GET  /api/sync/perfil             — perfil y huella de este nodo
  GET  /api/sync/calibracion        — vector de calibración (para que el par calcule δ)
  GET  /api/sync/estado             — pares sincronizados y su modo
  POST /api/sync/canal/{canal_id}   — fuerza el handshake con el par de un canal HTTP
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/sync")
log = logging.getLogger(__name__)


def _sync(request: Request):
    sync = request.app.state.sync
    if sync is None:
        raise HTTPException(status_code=501, detail="El backend no soporta sincronización")
    return sync


@router.get("/perfil")
async def perfil(request: Request):
    return _sync(request).perfil()


@router.get("/calibracion")
async def calibracion(request: Request):
    return _sync(request).calibracion()


@router.get("/estado")
async def estado(request: Request):
    sync = _sync(request)
    return {"local": sync.perfil(), "pares": sync.peers}


async def sincronizar(st, peer_url: str, forzar: bool = False) -> dict:
    """Handshake con un par HTTP. Lanza excepción si el par no responde."""
    sync = st.sync
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{peer_url}/api/sync/perfil")
        r.raise_for_status()
        remoto = r.json()

        if not forzar:
            rec = sync.vigente(peer_url, remoto)
            if rec:
                return rec

        resultado = sync.evaluar_rapido(remoto)
        if resultado is None:
            r = await client.get(f"{peer_url}/api/sync/calibracion")
            r.raise_for_status()
            resultado = sync.evaluar_con_calibracion(remoto, r.json())

    rec = sync.registrar(peer_url, remoto, resultado)
    log.info("Sync con %s: %s (δ=%s, gap_min=%s)", peer_url, rec["modo"],
             rec.get("delta"), rec.get("gap_min"))
    return rec


async def asegurar_sync(st, peer_url: str) -> Optional[dict]:
    """Registro de sync para el par; nunca lanza (si el par no responde usa lo guardado)."""
    sync = st.sync
    if sync is None:
        return None
    if peer_url in sync.verificados and peer_url in sync.peers:
        return sync.peers[peer_url]
    try:
        return await sincronizar(st, peer_url)
    except Exception as exc:
        log.warning("Sync con %s falló (%s); se usa el registro guardado, si hay", peer_url, exc)
        return sync.peers.get(peer_url)


@router.post("/canal/{canal_id}")
async def sincronizar_canal(canal_id: str, request: Request):
    st = request.app.state
    _sync(request)
    canal = st.gestor_canales.obtener(canal_id)
    if not canal:
        raise HTTPException(status_code=404, detail=f"Canal '{canal_id}' no encontrado")
    if canal.tipo != "http":
        raise HTTPException(status_code=422, detail="El handshake solo está disponible en canales HTTP")
    try:
        return await sincronizar(st, canal.peer_url, forzar=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"No se pudo sincronizar con el par: {exc}")
