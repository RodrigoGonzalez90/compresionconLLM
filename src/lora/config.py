"""
Configuración LoRa — persiste en data/lora_config.json.

Parámetros:
  enabled     Activa el transporte LoRa al iniciar el nodo
  port        Puerto serie: /dev/ttyUSB0, /dev/ttyAMA0, COM3, etc.
  baudrate    Velocidad del puerto serie (módulo RYLR896 → 115200)
  sf          Spreading Factor (7-12). Mayor SF = más alcance, menor velocidad
  bw          Ancho de banda: 7=125kHz · 8=250kHz · 9=500kHz (índice RYLR)
  cr          Coding Rate: 1=4/5 · 2=4/6 · 3=4/7 · 4=4/8
  tx_power    Potencia de TX en dBm (0-20)
  address     Dirección de este nodo en la red LoRa (0-65535)
  network_id  ID de red (todos los nodos deben compartirlo, 3-15 o 18)
  max_chunk   Bytes máximos por paquete LoRa incluyendo header (4B)
  freq_mhz    Frecuencia en MHz (915.0 para América, 868.0 para Europa)
  modo        "at_rylr"    — módulos RYLR896/406 con AT commands por serial
              "transparent" — serial crudo
              "tcp"         — gateway WiFi (XIAO ESP32S3 + SX1262)
  tcp_host    IP o hostname del gateway WiFi (modo tcp)
  tcp_port    Puerto TCP del gateway WiFi (modo tcp, default 9000)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_PATH = Path("data/lora_config.json")

# Mapa BW index → kHz (para la UI)
BW_KHZ = {7: 125, 8: 250, 9: 500}


@dataclass
class LoRaConfig:
    enabled:    bool = False
    port:       str  = ""
    baudrate:   int  = 115200
    sf:         int  = 9
    bw:         int  = 7       # índice RYLR896: 7=125kHz
    cr:         int  = 1       # 4/5
    tx_power:   int  = 14
    address:    int  = 1
    network_id: int  = 18
    max_chunk:  int  = 120     # bytes por paquete (incluye 4B de header)
    freq_mhz:   float = 915.0  # MHz
    modo:       str  = "at_rylr"
    tcp_host:   str  = ""
    tcp_port:   int  = 9000

    @classmethod
    def load(cls) -> "LoRaConfig":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**valid)
            except Exception:
                pass
        return cls()

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bw_khz"] = BW_KHZ.get(self.bw, self.bw)
        return d
