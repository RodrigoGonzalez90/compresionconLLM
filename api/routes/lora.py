"""
Rutas de configuración y estado del transporte LoRa:
  GET  /api/lora/config    — configuración actual
  POST /api/lora/config    — actualizar y reconectar
  GET  /api/lora/status    — estado de conexión (connected, rssi, error…)
  GET  /api/lora/puertos   — lista de puertos serie disponibles
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/lora")


class LoRaConfigRequest(BaseModel):
    enabled:    bool = False
    port:       str  = ""
    baudrate:   int  = Field(115200, ge=1200, le=921600)
    sf:         int  = Field(9,  ge=7, le=12)
    bw:         int  = Field(7,  ge=7, le=9)
    cr:         int  = Field(1,  ge=1, le=4)
    tx_power:   int  = Field(14, ge=0, le=20)
    address:    int  = Field(1,  ge=0, le=65535)
    network_id: int  = Field(18, ge=0, le=255)
    max_chunk:  int  = Field(120, ge=20, le=240)
    modo:       str  = "at_rylr"


@router.get("/config")
async def get_config(request: Request):
    return request.app.state.lora_transport.config.to_dict()


@router.post("/config")
async def set_config(body: LoRaConfigRequest, request: Request):
    transport = request.app.state.lora_transport
    cfg = transport.config
    for field, value in body.dict().items():
        setattr(cfg, field, value)
    cfg.save()

    # Reconectar con la nueva configuración
    await transport.disconnect()
    if cfg.enabled:
        await transport.connect()

    return {**cfg.to_dict(), "status": transport.status()}


@router.get("/status")
async def get_status(request: Request):
    return request.app.state.lora_transport.status()


@router.get("/puertos")
async def listar_puertos():
    """Lista los puertos serie disponibles en el sistema."""
    try:
        import serial.tools.list_ports
        return [
            {"port": p.device, "desc": p.description, "hwid": p.hwid}
            for p in serial.tools.list_ports.comports()
        ]
    except ImportError:
        return []
    except Exception as exc:
        return [{"error": str(exc)}]
