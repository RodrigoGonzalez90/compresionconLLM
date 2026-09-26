"""
Medios de comunicación (agnósticos al canal físico).

Meshstatic solo entrega bytes al medio y recibe bytes de él; el enlace fragmenta,
reensambla y confirma (ACK) sin importar qué haya debajo. Cada medio es una instancia
de configuración sobre el mismo motor de enlace (LoRaTransport):

  lora       Módulo LoRa por serial (AT RYLR) o gateway WiFi/TCP
  rf         Módem de radio por UART (HC-12, APC220, E32, SiK, ...)
  bluetooth  Bluetooth clásico (SPP) expuesto como puerto serie (rfcomm / COMx)
  serial     Cable serie / USB directo entre nodos
  tcp        Gateway por red (WiFi/Ethernet), protocolo de gateway Meshstatic

Para agregar un tipo nuevo: sumarlo a CATALOGO y mapearlo en `_config_de`.
El catálogo describe los campos, y la web genera el formulario a partir de él.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.lora.config import LoRaConfig
from src.lora.transport import LoRaTransport

log = logging.getLogger(__name__)

MEDIOS_FILE = Path("data/medios.json")
MEDIO_DEFECTO = "lora"       # medio incorporado; usa data/lora_config.json


def _campo(k, label, tipo="text", default="", ayuda="", opciones=None, **extra):
    c = {"k": k, "label": label, "tipo": tipo, "default": default, "ayuda": ayuda}
    if opciones:
        c["opciones"] = opciones
    c.update(extra)
    return c


_PUERTO = _campo("port", "Puerto", "puerto", "", "COM3, /dev/ttyUSB0, /dev/rfcomm0 ...")
_BAUD   = lambda d: _campo("baudrate", "Baudios", "number", d, min=1200, max=921600)
_CHUNK  = lambda d: _campo("max_chunk", "Bytes por paquete", "number", d,
                           "Incluye 4 B de cabecera; ajustalo al MTU del medio", min=20, max=240)

CATALOGO: Dict[str, dict] = {
    "lora": {
        "label": "LoRa", "icono": "📡",
        "desc": "Módulo LoRa por serial (AT RYLR896/406) o gateway WiFi/TCP.",
        "campos": [
            _campo("modo", "Conexión", "select", "at_rylr", opciones=[
                {"v": "at_rylr", "t": "Serial · AT RYLR"},
                {"v": "tcp", "t": "Gateway WiFi/TCP"}]),
            _PUERTO, _BAUD(115200),
            _campo("tcp_host", "Host del gateway", "text", "", "Solo modo TCP"),
            _campo("tcp_port", "Puerto TCP", "number", 9000),
            _campo("address", "Dirección propia", "number", 1, min=0, max=65535),
            _campo("network_id", "ID de red", "number", 18, min=0, max=255),
            _campo("sf", "Spreading Factor", "number", 9, min=7, max=12),
            _campo("bw", "Ancho de banda (idx)", "number", 7, "7=125k · 8=250k · 9=500k", min=7, max=9),
            _campo("cr", "Coding rate", "number", 1, min=1, max=4),
            _campo("tx_power", "Potencia (dBm)", "number", 14, min=0, max=20),
            _campo("freq_mhz", "Frecuencia (MHz)", "number", 915.0),
            _CHUNK(120),
        ],
    },
    "rf": {
        "label": "Radio RF (UART)", "icono": "📻",
        "desc": "Módem de radio por UART: HC-12, APC220, E32, SiK... Transparente, sin direccionamiento.",
        "campos": [_PUERTO, _BAUD(9600), _CHUNK(58)],
    },
    "bluetooth": {
        "label": "Bluetooth", "icono": "🔵",
        "desc": "Bluetooth clásico (SPP). Emparejalo en el sistema y usá el puerto serie que crea "
                "(Windows: COMx · Linux: /dev/rfcomm0).",
        "campos": [_PUERTO, _BAUD(9600), _CHUNK(120)],
    },
    "serial": {
        "label": "Serie / USB", "icono": "🔌",
        "desc": "Cable serie o USB directo entre nodos.",
        "campos": [_PUERTO, _BAUD(115200), _CHUNK(120)],
    },
    "tcp": {
        "label": "Red (gateway TCP)", "icono": "🌐",
        "desc": "Gateway por WiFi/Ethernet con el protocolo de gateway Meshstatic.",
        "campos": [
            _campo("tcp_host", "Host", "text", ""),
            _campo("tcp_port", "Puerto", "number", 9000),
            _campo("address", "Dirección propia", "number", 1, min=0, max=65535),
            _CHUNK(120),
        ],
    },
}


def _defaults(tipo: str) -> dict:
    return {c["k"]: c["default"] for c in CATALOGO[tipo]["campos"]}


def _config_de(tipo: str, params: dict, enabled: bool) -> LoRaConfig:
    """Traduce (tipo, params) a la configuración del motor de enlace."""
    validos = LoRaConfig.__dataclass_fields__
    p = {k: v for k, v in {**_defaults(tipo), **params}.items()
         if k in validos and k not in ("enabled", "modo")}
    if tipo == "lora":
        modo = params.get("modo", "at_rylr")
        modo = modo if modo in ("at_rylr", "tcp") else "at_rylr"
    elif tipo == "tcp":
        modo = "tcp"
    else:
        modo = "transparent"
    return LoRaConfig(enabled=enabled, modo=modo, **p)


def _params_de(tipo: str, cfg: LoRaConfig) -> dict:
    return {c["k"]: getattr(cfg, c["k"], c["default"]) for c in CATALOGO[tipo]["campos"]}


@dataclass
class Medio:
    id: str
    tipo: str
    nombre: str
    transporte: LoRaTransport
    incorporado: bool = False

    @property
    def enabled(self) -> bool:
        return self.transporte.config.enabled

    @property
    def params(self) -> dict:
        return _params_de(self.tipo, self.transporte.config)

    def to_dict(self) -> dict:
        cat = CATALOGO.get(self.tipo, {})
        return {
            "id": self.id, "tipo": self.tipo, "nombre": self.nombre,
            "label": cat.get("label", self.tipo), "icono": cat.get("icono", "•"),
            "params": self.params, "enabled": self.enabled,
            "incorporado": self.incorporado, "status": self.transporte.status(),
        }


class MedioManager:
    def __init__(self, lora_transport: LoRaTransport, on_new: Callable[[Medio], None]):
        self._on_new = on_new
        self._medios: Dict[str, Medio] = {}
        self._medios[MEDIO_DEFECTO] = Medio(
            id=MEDIO_DEFECTO, tipo="lora", nombre="LoRa",
            transporte=lora_transport, incorporado=True,
        )
        self._load()

    # ── Persistencia (solo medios extra; el incorporado usa lora_config.json) ──

    def _load(self) -> None:
        try:
            if not MEDIOS_FILE.exists():
                return
            for it in json.loads(MEDIOS_FILE.read_text(encoding="utf-8")):
                if it["tipo"] not in CATALOGO or it["id"] in self._medios:
                    continue
                self._medios[it["id"]] = self._crear_medio(
                    it["id"], it["tipo"], it["nombre"], it.get("params", {}), it.get("enabled", False))
        except Exception as exc:
            log.warning("Medios: no se pudo cargar %s: %s", MEDIOS_FILE, exc)

    def _save(self) -> None:
        try:
            MEDIOS_FILE.parent.mkdir(parents=True, exist_ok=True)
            items = [{"id": m.id, "tipo": m.tipo, "nombre": m.nombre, "params": m.params,
                      "enabled": m.enabled}
                     for m in self._medios.values() if not m.incorporado]
            MEDIOS_FILE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            log.warning("Medios: no se pudo guardar: %s", exc)

    def _crear_medio(self, id_, tipo, nombre, params, enabled) -> Medio:
        m = Medio(id=id_, tipo=tipo, nombre=nombre,
                  transporte=LoRaTransport(_config_de(tipo, params, enabled)))
        self._on_new(m)
        return m

    # ── Consulta ──────────────────────────────────────────────────────────────

    def listar(self) -> List[Medio]:
        return list(self._medios.values())

    def obtener(self, id_: str) -> Optional[Medio]:
        return self._medios.get(id_)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def crear(self, tipo: str, nombre: str, params: dict) -> Medio:
        if tipo not in CATALOGO:
            raise ValueError(f"Tipo de medio desconocido: {tipo}")
        m = self._crear_medio(uuid.uuid4().hex[:8], tipo, nombre.strip() or CATALOGO[tipo]["label"],
                              params, False)
        self._medios[m.id] = m
        self._save()
        return m

    async def actualizar(self, id_: str, nombre: Optional[str] = None,
                         params: Optional[dict] = None, enabled: Optional[bool] = None) -> Optional[Medio]:
        m = self._medios.get(id_)
        if not m:
            return None
        if nombre is not None:
            m.nombre = nombre.strip() or m.nombre
        nuevos = {**m.params, **(params or {})}
        activo = m.enabled if enabled is None else enabled
        await m.transporte.disconnect()
        m.transporte.config = _config_de(m.tipo, nuevos, activo)
        if m.incorporado:
            m.transporte.config.save()
        else:
            self._save()
        if activo:
            await m.transporte.connect()
        return m

    async def conectar(self, id_: str) -> Optional[Medio]:
        m = self._medios.get(id_)
        if m:
            await m.transporte.disconnect()
            await m.transporte.connect()
        return m

    async def desconectar(self, id_: str) -> Optional[Medio]:
        m = self._medios.get(id_)
        if m:
            await m.transporte.disconnect()
        return m

    async def eliminar(self, id_: str) -> bool:
        m = self._medios.get(id_)
        if not m or m.incorporado:
            return False
        await m.transporte.disconnect()
        del self._medios[id_]
        self._save()
        return True

    async def conectar_habilitados(self) -> None:
        for m in self._medios.values():
            if m.enabled:
                await m.transporte.connect()

    async def desconectar_todos(self) -> None:
        for m in self._medios.values():
            await m.transporte.disconnect()
