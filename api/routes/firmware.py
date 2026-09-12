"""
GET /api/firmware/xiao-esp32s3
  Genera y descarga el archivo .ino del gateway WiFi para XIAO ESP32S3 + Wio-SX1262.
  Parámetros (query):
    ssid      — SSID de la red WiFi (requerido)
    password  — contraseña WiFi (requerido)
  Los parámetros LoRa se toman de la configuración actual del nodo.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from src.lora.firmware_xiao import render

router = APIRouter(prefix="/api/firmware")


@router.get("/xiao-esp32s3", response_class=PlainTextResponse)
async def descargar_firmware(
    request: Request,
    ssid:     str = Query(..., description="SSID de la red WiFi"),
    password: str = Query(..., description="Contraseña WiFi"),
):
    if not ssid.strip():
        raise HTTPException(status_code=422, detail="ssid no puede estar vacío")

    cfg = request.app.state.lora_transport.config

    codigo = render(
        ssid=ssid,
        password=password,
        freq_mhz=cfg.freq_mhz,
        sf=cfg.sf,
        bw=cfg.bw,
        cr=cfg.cr,
        tx_power=cfg.tx_power,
        address=cfg.address,
        network_id=cfg.network_id,
        tcp_port=cfg.tcp_port,
        tcp_host_hint="<IP del ESP32S3>",
    )

    nombre = f"meshstatic_gateway_addr{cfg.address}.ino"
    return PlainTextResponse(
        content=codigo,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
    )
