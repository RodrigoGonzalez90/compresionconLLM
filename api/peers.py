"""
Descubrimiento dinámico de pares: cada nodo se identifica por un ID de instancia
estable y el par aprende su dirección real de las conexiones que recibe, así un
cambio de IP (DHCP) se corrige solo en cuanto el par vuelve a comunicarse.

Cabeceras que envía cada nodo en sus llamadas a otros nodos:
  X-Meshstatic-Instancia   ID único y persistente del nodo
  X-Meshstatic-Puerto      puerto público en el que atiende (para armar la URL de vuelta)

La URL de vuelta se arma con la IP de origen de la conexión. Si Docker enmascara esa
IP (gateway del contenedor, loopback) se usa la URL que declaró el emisor (OWN_URL).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import struct
from typing import Optional
from urllib.parse import quote, unquote, urlparse

from fastapi import Request

from src.sync import instancia_id

log = logging.getLogger(__name__)

H_INSTANCIA = "x-meshstatic-instancia"
H_PUERTO    = "x-meshstatic-puerto"
H_NOMBRE    = "x-meshstatic-nombre"


def puerto_propio(st) -> int:
    try:
        return urlparse(st.own_url or "").port or 8000
    except ValueError:
        return 8000


def cabeceras_propias(st) -> dict:
    return {"X-Meshstatic-Instancia": instancia_id(),
            "X-Meshstatic-Puerto":    str(puerto_propio(st)),
            "X-Meshstatic-Nombre":    quote(st.node_id)}


def _gateway() -> Optional[str]:
    """Gateway por defecto del contenedor: Docker enmascara con él los orígenes del host."""
    try:
        with open("/proc/net/route", encoding="ascii") as f:
            for line in f.read().splitlines()[1:]:
                p = line.split()
                if len(p) > 2 and p[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(p[2], 16)))
    except Exception:
        pass
    return None


def _ip_utilizable(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.version != 4 or ip.is_loopback or ip.is_unspecified or ip.is_link_local:
        return False
    return host != _gateway()


def url_de_par(request: Request, puerto: Optional[int], declarada: Optional[str]) -> Optional[str]:
    host = request.client.host if request.client else None
    if host and puerto and _ip_utilizable(host):
        return f"http://{host}:{puerto}"
    return declarada.rstrip("/") if declarada else None


async def aprender_par(st, request: Request, nombre: str = "", declarada: Optional[str] = None) -> None:
    """Actualiza (o crea) el canal de retorno hacia el nodo que nos está llamando."""
    from api.routes.ws import manager as ws_manager

    iid = request.headers.get(H_INSTANCIA)
    nombre = unquote(request.headers.get(H_NOMBRE, "")).strip() or nombre
    if iid == instancia_id():
        return   # somos nosotros mismos (canal que apunta al propio nodo)
    try:
        puerto = int(request.headers.get(H_PUERTO, ""))
    except ValueError:
        puerto = None
    url = url_de_par(request, puerto, declarada)
    if not url:
        return

    gestor = st.gestor_canales
    canal = gestor.por_instancia(iid) if iid else None
    if canal is None:
        canal = gestor.por_url(url)
        if canal and iid:
            gestor.vincular(canal, iid)

    if canal is not None:
        if canal.nombre_auto and nombre and canal.nombre != nombre:
            gestor.renombrar_auto(canal, nombre)
            await ws_manager.broadcast("canales_actualizados", canal.to_dict())
        if canal.peer_url != url:
            viejo = canal.peer_url
            gestor.cambiar_url(canal, url)
            if st.sync:
                st.sync.migrar(viejo, url)
            log.info("Par %s cambió de dirección: %s → %s", canal.nombre, viejo, url)
            await ws_manager.broadcast("canales_actualizados", canal.to_dict())
        return

    nuevo = await gestor.crear(nombre or f"par-{(iid or url)[:6]}", url)
    gestor.marcar_auto(nuevo)
    if iid:
        gestor.vincular(nuevo, iid)
    await ws_manager.broadcast("canales_actualizados", {"nuevo": nuevo.to_dict()})
