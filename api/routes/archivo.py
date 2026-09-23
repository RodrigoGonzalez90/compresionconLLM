"""
Transporte por archivo .msh (Meshstatic).

El archivo .msh es exactamente el payload que produciría el encoder para
cualquier otro canal (LoRa, HF, APRS, etc.): sin cabecera extra, sin
metadatos, sin overhead adicional.  La compresion es identica.

Formato: los bytes directos del encoder
  0x4C + CRC32(4B) + N_tokens(2B) + bits  →  modo lossless
  0x52 + UTF-8                             →  modo raw (fallback adaptativo)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api")

MSH_EXT    = "msh"
MAX_UPLOAD = 1 * 1024 * 1024   # 1 MB — mucho mas que cualquier texto comprimido


# ── Endpoint: exportar ────────────────────────────────────────────────────────

class ExportarRequest(BaseModel):
    texto:  str = Field(..., min_length=1, max_length=5000)
    nombre: str = Field(default="mensaje", max_length=64)


@router.post("/exportar")
async def exportar(body: ExportarRequest, request: Request):
    """
    Codifica `texto` y devuelve un archivo .msh.
    El contenido es identico al payload que se transmitiria por radio.
    """
    st      = request.app.state
    paquete = st.lm_encoder.comprimir(body.texto)

    return Response(
        content=paquete.datos,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{body.nombre}.{MSH_EXT}"',
            "X-Meshstatic-Modo":   paquete.modo,
            "X-Meshstatic-Ratio":  str(round(paquete.porcentaje_ahorro, 1)),
            "X-Meshstatic-Bytes-Orig": str(paquete.bytes_originales),
            "X-Meshstatic-Bytes-Tx":   str(paquete.bytes_transmitidos),
        },
    )


# ── Endpoint: importar ────────────────────────────────────────────────────────

@router.post("/importar")
async def importar(
    request: Request,
    archivo: UploadFile = File(...),
):
    """
    Recibe un archivo .msh y decodifica el texto original.
    Funciona con cualquier archivo producido por /api/exportar
    o directamente capturado de un canal de radio.
    """
    datos = await archivo.read(MAX_UPLOAD + 1)
    if len(datos) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (max 1 MB)")
    if not datos:
        raise HTTPException(status_code=422, detail="Archivo vacío")

    decoded = request.app.state.lm_decoder.decodificar(datos)

    return {
        "texto_reconstruido": decoded.texto_reconstruido,
        "crc_ok":             decoded.crc_ok,
        "modo":               decoded.modo,
        "decode_ms":          round(decoded.decode_ms, 1),
    }
