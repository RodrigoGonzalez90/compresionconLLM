"""
Transporte por archivo .msh (Meshstatic).

Permite comprimir un mensaje a un archivo portátil que puede enviarse
por cualquier medio (pendrive, correo, WhatsApp, etc.) y decodificarse
en cualquier nodo con el mismo modelo.

Formato .msh:
  4 bytes  : magic "MSH\x01"
  2 bytes  : uint16 BE  = longitud del bloque de metadatos JSON
  N bytes  : JSON UTF-8 (sender, timestamp, modelo, bytes_orig, modo)
  rest     : payload del encoder (autocontenido: header + CRC32 + bits)
"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api")

MSH_MAGIC   = b"MSH\x01"
MSH_EXT     = "msh"
MAX_UPLOAD  = 1 * 1024 * 1024   # 1 MB, más que suficiente para texto comprimido


# ── Serialización del formato .msh ────────────────────────────────────────────

def empaquetar(datos: bytes, meta: dict) -> bytes:
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return MSH_MAGIC + struct.pack(">H", len(meta_bytes)) + meta_bytes + datos


def desempaquetar(raw: bytes) -> tuple[bytes, dict]:
    if not raw.startswith(MSH_MAGIC):
        # Compatibilidad: bytes sin envoltorio (payload directo del encoder)
        return raw, {}
    if len(raw) < 6:
        raise ValueError("Archivo .msh demasiado corto")
    meta_len = struct.unpack(">H", raw[4:6])[0]
    if len(raw) < 6 + meta_len:
        raise ValueError("Metadatos truncados en archivo .msh")
    meta = json.loads(raw[6 : 6 + meta_len])
    datos = raw[6 + meta_len :]
    return datos, meta


# ── Endpoint: exportar ────────────────────────────────────────────────────────

class ExportarRequest(BaseModel):
    texto:   str = Field(..., min_length=1, max_length=5000)
    nombre:  str = Field(default="mensaje", max_length=64)
    emisor:  str = Field(default="", max_length=64)


@router.post("/exportar")
async def exportar(body: ExportarRequest, request: Request):
    """
    Codifica `texto` y devuelve un archivo .msh descargable.
    El receptor lo importa con POST /api/importar.
    """
    st = request.app.state
    paquete = st.lm_encoder.comprimir(body.texto)

    meta = {
        "emisor":      body.emisor or st.node_id,
        "ts":          datetime.now(timezone.utc).isoformat(),
        "modelo":      str(st.backend),
        "bytes_orig":  paquete.bytes_originales,
        "bytes_tx":    paquete.bytes_transmitidos,
        "ratio_pct":   round(paquete.porcentaje_ahorro, 1),
        "modo":        paquete.modo,
        "n_tokens":    paquete.n_tokens,
        "encode_ms":   round(paquete.encode_ms, 1),
    }

    archivo = empaquetar(paquete.datos, meta)
    nombre_archivo = f"{body.nombre}.{MSH_EXT}"

    return Response(
        content=archivo,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{nombre_archivo}"',
            "X-Meshstatic-Modo":   paquete.modo,
            "X-Meshstatic-Ratio":  str(meta["ratio_pct"]),
        },
    )


# ── Endpoint: importar ────────────────────────────────────────────────────────

@router.post("/importar")
async def importar(
    request: Request,
    archivo: UploadFile = File(..., description="Archivo .msh generado por /api/exportar"),
):
    """
    Recibe un archivo .msh, lo decodifica y devuelve el texto original.
    Requiere el mismo modelo que usó el emisor para codificar.
    """
    raw = await archivo.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 1 MB)")

    try:
        datos, meta = desempaquetar(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Formato .msh inválido: {e}")

    st = request.app.state
    decoded = st.lm_decoder.decodificar(datos)

    return {
        "texto_reconstruido": decoded.texto_reconstruido,
        "crc_ok":             decoded.crc_ok,
        "modo":               decoded.modo,
        "decode_ms":          round(decoded.decode_ms, 1),
        "meta_emisor":        meta,
    }
