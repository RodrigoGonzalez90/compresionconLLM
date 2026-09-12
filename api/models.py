from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


def _uid() -> str:
    return str(uuid.uuid4())[:8].upper()


def _now() -> datetime:
    return datetime.now(timezone.utc)


ESTADOS = {
    "borrador":    "Borrador",
    "enviando":    "Enviando…",
    "transmitido": "Transmitido",
    "confirmado":  "Confirmado",
    "error_tx":    "Error de transmisión",
    "error_rx":    "Error de recepción (CRC)",
    "recibido":    "Recibido",
}


class Mensaje(BaseModel):
    id: str = Field(default_factory=_uid)
    texto_original: str
    esqueleto: Optional[str] = None
    bytes_hex: Optional[str] = None
    texto_reconstruido: Optional[str] = None
    estado: str = "borrador"
    bytes_original: int = 0
    bytes_tx: int = 0
    n_tokens: int = 0
    ratio_pct: float = 0.0
    confianza_ml: float = 0.0
    encoding: str = "lossless"
    timestamp: datetime = Field(default_factory=_now)
    timestamp_confirmacion: Optional[datetime] = None
    nodo_origen: str = ""
    nodo_destino: str = ""
    canal_id: str = ""
    direccion: str = "saliente"
    rank_dist: Optional[Dict[str, float]] = None
    rangos_seq: List[int] = Field(default_factory=list)
    encode_ms: float = 0.0
    decode_ms: float = 0.0

    model_config = {"from_attributes": True}

    def dict_ui(self) -> dict:
        return {
            "id": self.id,
            "texto_original": self.texto_original,
            "esqueleto": self.esqueleto,
            "texto_reconstruido": self.texto_reconstruido,
            "estado": self.estado,
            "estado_label": ESTADOS.get(self.estado, self.estado),
            "bytes_original": self.bytes_original,
            "bytes_tx": self.bytes_tx,
            "n_tokens": self.n_tokens,
            "ratio_pct": round(self.ratio_pct, 1),
            "confianza_ml": round(self.confianza_ml * 100, 1),
            "encoding": self.encoding,
            "timestamp": self.timestamp.isoformat(),
            "timestamp_confirmacion": (
                self.timestamp_confirmacion.isoformat()
                if self.timestamp_confirmacion else None
            ),
            "nodo_origen": self.nodo_origen,
            "nodo_destino": self.nodo_destino,
            "canal_id": self.canal_id,
            "direccion": self.direccion,
            "rank_dist": self.rank_dist,
            "rangos_seq": self.rangos_seq,
            "encode_ms": round(self.encode_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
        }


class EnviarRequest(BaseModel):
    texto: str = Field(..., min_length=1, max_length=5000)
    nodo_destino: str = Field(default="")
    contexto: str = Field(default="")
    canal_id: str = Field(default="")


class RecibirRequest(BaseModel):
    bytes_hex: str
    id_mensaje: str
    nodo_origen: str = "desconocido"
    nodo_origen_url: Optional[str] = None   # URL del emisor para poder responder
    contexto: str = ""
    timestamp_envio: Optional[str] = None


class RecibirResponse(BaseModel):
    ok: bool = True
    id_mensaje: str
    texto_reconstruido: str
    esqueleto: str
    confianza_ml: float
    bytes_tx: int
    crc_ok: bool = True
    decode_ms: float = 0.0


class EstadoNodo(BaseModel):
    node_id: str
    peer_url: Optional[str]
    backend: str
    total_enviados: int
    total_recibidos: int
    compresion_media_pct: float
    online: bool = True
