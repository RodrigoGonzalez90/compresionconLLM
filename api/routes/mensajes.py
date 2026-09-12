from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request

from api.models import EnviarRequest, EstadoNodo, Mensaje, RecibirRequest, RecibirResponse
from api.routes.ws import manager as ws_manager

router = APIRouter(prefix="/api")


def _st(request: Request):
    return request.app.state


@router.post("/enviar", response_model=dict)
async def enviar(body: EnviarRequest, request: Request):
    st = _st(request)

    paquete  = st.lm_encoder.comprimir(body.texto)
    peer_url = st.gestor_canales.peer_url_de(body.canal_id or None)
    destino  = peer_url or "—"

    msg = Mensaje(
        texto_original=body.texto,
        esqueleto=paquete.esqueleto,
        bytes_hex=paquete.datos.hex(),
        estado="enviando",
        bytes_original=paquete.bytes_originales,
        bytes_tx=paquete.bytes_transmitidos,
        n_tokens=paquete.tokens_count,
        ratio_pct=paquete.porcentaje_ahorro,
        nodo_origen=st.node_id,
        nodo_destino=destino,
        canal_id=body.canal_id,
        encoding=f"lossless/{paquete.modo}",
        direccion="saliente",
        rank_dist=paquete.rank_distribution if paquete.modo == "lossless" else None,
        rangos_seq=paquete.rangos if paquete.modo == "lossless" else [],
        encode_ms=paquete.encode_ms,
    )
    await st.store.guardar(msg)
    await ws_manager.broadcast("nuevo_mensaje", msg.dict_ui())

    canal_obj = st.gestor_canales.obtener(body.canal_id) if body.canal_id else None
    usar_lora = canal_obj is not None and canal_obj.tipo == "lora"

    if usar_lora:
        # ── Envío por LoRa ────────────────────────────────────────────────────
        try:
            dest = int(canal_obj.peer_url)
        except (ValueError, AttributeError):
            dest = 0
        try:
            lora_msg_id = await st.lora_transport.send(paquete.datos, dest_address=dest)
            st.lora_transport.register_pending(lora_msg_id, msg.id)
            await st.store.actualizar(msg.id, estado="transmitido")
        except Exception as exc:
            await st.store.actualizar(msg.id, estado="error_tx")
            raise HTTPException(status_code=503, detail=f"LoRa TX error: {exc}")
        await ws_manager.broadcast("estado_actualizado", st.store.obtener(msg.id).dict_ui())

    elif peer_url:
        # ── Envío por HTTP ────────────────────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{peer_url}/api/recibir",
                    json={
                        "bytes_hex":       paquete.datos.hex(),
                        "id_mensaje":      msg.id,
                        "nodo_origen":     st.node_id,
                        "nodo_origen_url": st.own_url,
                        "contexto":        body.contexto,
                        "timestamp_envio": msg.timestamp.isoformat(),
                    },
                )
                resp.raise_for_status()
                ack = resp.json()

            await st.store.actualizar(
                msg.id,
                estado="confirmado",
                texto_reconstruido=ack.get("texto_reconstruido"),
                confianza_ml=ack.get("confianza_ml", 1.0),
                timestamp_confirmacion=datetime.now(timezone.utc),
                decode_ms=ack.get("decode_ms", 0.0),
            )

        except httpx.RequestError as exc:
            await st.store.actualizar(msg.id, estado="error_tx")
            raise HTTPException(status_code=503, detail=f"No se pudo contactar al par: {exc}")

        except httpx.HTTPStatusError as exc:
            await st.store.actualizar(msg.id, estado="error_tx")
            raise HTTPException(status_code=502, detail=f"El par respondió {exc.response.status_code}")

        await ws_manager.broadcast("estado_actualizado", st.store.obtener(msg.id).dict_ui())

    else:
        await st.store.actualizar(msg.id, estado="transmitido")
        await ws_manager.broadcast("estado_actualizado", st.store.obtener(msg.id).dict_ui())

    return st.store.obtener(msg.id).dict_ui()


@router.post("/recibir", response_model=RecibirResponse)
async def recibir(body: RecibirRequest, request: Request):
    st = _st(request)

    try:
        datos = bytes.fromhex(body.bytes_hex)
    except ValueError:
        raise HTTPException(status_code=422, detail="bytes_hex inválido")

    decoded = st.lm_decoder.decodificar(datos)

    # Auto-crear canal de vuelta si el emisor nos informó su URL
    if body.nodo_origen_url:
        canales = st.gestor_canales.listar()
        urls_existentes = {c.peer_url for c in canales}
        if body.nodo_origen_url.rstrip("/") not in urls_existentes:
            await st.gestor_canales.crear(body.nodo_origen, body.nodo_origen_url)
            await ws_manager.broadcast("canales_actualizados", {
                "nuevo": {"nombre": body.nodo_origen, "peer_url": body.nodo_origen_url}
            })

    bytes_orig = len(decoded.texto_reconstruido.encode("utf-8"))
    bytes_rx   = len(datos)
    ratio      = round((1 - bytes_rx / bytes_orig) * 100, 1) if bytes_orig else 0.0

    msg = Mensaje(
        id=body.id_mensaje,
        texto_original=decoded.texto_reconstruido,
        esqueleto=decoded.esqueleto,
        bytes_hex=body.bytes_hex,
        texto_reconstruido=decoded.texto_reconstruido,
        estado="recibido",
        bytes_original=bytes_orig,
        bytes_tx=bytes_rx,
        ratio_pct=ratio,
        n_tokens=len(decoded.ids_tokens),
        confianza_ml=decoded.confianza_modelo,
        nodo_origen=body.nodo_origen,
        nodo_destino=st.node_id,
        encoding=f"lossless/{decoded.modo}",
        direccion="entrante",
        timestamp=datetime.fromisoformat(body.timestamp_envio)
        if body.timestamp_envio else datetime.now(timezone.utc),
        decode_ms=decoded.decode_ms,
    )
    await st.store.guardar(msg)
    await ws_manager.broadcast("nuevo_mensaje", msg.dict_ui())

    return RecibirResponse(
        id_mensaje=msg.id,
        texto_reconstruido=decoded.texto_reconstruido,
        esqueleto=decoded.esqueleto,
        confianza_ml=decoded.confianza_modelo,
        bytes_tx=len(datos),
        crc_ok=decoded.crc_ok,
        decode_ms=decoded.decode_ms,
    )


@router.get("/mensajes")
async def listar_mensajes(request: Request):
    return [m.dict_ui() for m in _st(request).store.listar()]


@router.delete("/mensajes/{msg_id}", status_code=204)
async def eliminar_mensaje(msg_id: str, request: Request):
    if not await _st(request).store.eliminar(msg_id):
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")


@router.delete("/mensajes", status_code=204)
async def limpiar_mensajes(request: Request):
    await _st(request).store.limpiar()


@router.get("/estado", response_model=EstadoNodo)
async def estado_nodo(request: Request):
    st = _st(request)
    return EstadoNodo(
        node_id=st.node_id,
        peer_url=st.gestor_canales.peer_url_de(None),
        backend=st.backend,
        **st.store.stats(),
    )
