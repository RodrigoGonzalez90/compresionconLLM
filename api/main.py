from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.channels.manager import GestorCanales
from src.encoder.lm_encoder import LMEncoder
from src.decoder.lm_decoder import LMDecoder
from src.llm_backend import build_backend
from src.lora.config import LoRaConfig
from src.lora.transport import LoRaTransport
from api.store import MensajeStore
from api.models import Mensaje
from api.routes import mensajes as mensajes_router
from api.routes import ws as ws_router
from api.routes import canales as canales_router
from api.routes import lora as lora_router
from api.routes import firmware as firmware_router
from api.routes import spellcheck as spellcheck_router
from api.routes import archivo as archivo_router
from api.routes.ws import manager as ws_manager

WEB_PATH = ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    node_id     = os.getenv("NODE_ID",     "nodo-1")
    peer_url    = os.getenv("PEER_URL",    "").strip() or None
    own_url     = os.getenv("OWN_URL",     "").strip() or None
    backend     = os.getenv("BACKEND",     "ollama")
    modelo      = os.getenv("MODELO",      "phi3:mini").strip()
    ollama_host = os.getenv("OLLAMA_HOST", "http://ollama:11434").strip()

    print(f"[meshstatic] Nodo: {node_id}  |  Backend: {backend}  |  Modelo: {modelo}")

    llm_backend = build_backend(backend=backend, modelo=modelo, ollama_host=ollama_host)

    gestor_canales = GestorCanales()
    gestor_canales.inicializar_desde_env(peer_url, node_id)

    lm_encoder = LMEncoder(backend=llm_backend)
    lm_decoder = LMDecoder(backend=llm_backend)
    store      = MensajeStore()

    # ── LoRa transport ────────────────────────────────────────────────────────
    lora_cfg       = LoRaConfig.load()
    lora_transport = LoRaTransport(lora_cfg)

    # Callback: cuando llega un mensaje LoRa, decodificarlo y guardarlo
    def _lora_on_recv(payload: bytes, src_addr: int, lora_msg_id: int) -> None:
        asyncio.create_task(_handle_lora_recv(payload, src_addr, lora_msg_id,
                                              lm_decoder, store, node_id,
                                              lora_transport))

    lora_transport.on_receive(_lora_on_recv)

    # Callback: cuando llega un ACK LoRa, actualizar el estado del mensaje
    def _lora_on_ack(db_msg_id: str, crc_ok: bool) -> None:
        asyncio.create_task(_handle_lora_ack(db_msg_id, crc_ok, store))

    lora_transport.on_ack(_lora_on_ack)

    if lora_cfg.enabled and lora_cfg.port:
        await lora_transport.connect()

    app.state.node_id        = node_id
    app.state.own_url        = own_url
    app.state.backend        = backend
    app.state.lm_encoder     = lm_encoder
    app.state.lm_decoder     = lm_decoder
    app.state.gestor_canales = gestor_canales
    app.state.store          = store
    app.state.lora_transport = lora_transport

    print(f"[meshstatic] API lista — http://0.0.0.0:{os.getenv('PORT', 8000)}")
    yield

    await lora_transport.disconnect()


async def _handle_lora_recv(
    payload: bytes,
    src_addr: int,
    lora_msg_id: int,
    decoder,
    store: MensajeStore,
    node_id: str,
    lora_transport,
) -> None:
    """Procesa un mensaje recibido por LoRa: decodifica, persiste y envía ACK."""
    decoded = decoder.decodificar(payload)
    bytes_orig = len(decoded.texto_reconstruido.encode())
    msg = Mensaje(
        texto_original=decoded.texto_reconstruido,
        texto_reconstruido=decoded.texto_reconstruido,
        esqueleto=decoded.esqueleto,
        estado="recibido",
        bytes_tx=len(payload),
        bytes_original=bytes_orig,
        ratio_pct=round((1 - len(payload) / max(bytes_orig, 1)) * 100, 1),
        n_tokens=len(decoded.ids_tokens),
        confianza_ml=decoded.confianza_modelo,
        encoding=f"lora/{decoded.modo}",
        nodo_origen=f"lora:{src_addr}",
        nodo_destino=node_id,
        direccion="entrante",
        timestamp=datetime.now(timezone.utc),
    )
    await store.guardar(msg)
    await ws_manager.broadcast("nuevo_mensaje", msg.dict_ui())

    # Enviar ACK de vuelta al nodo emisor
    try:
        await lora_transport.send_ack(lora_msg_id, decoded.crc_ok, src_addr)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("LoRa ACK TX fallido: %s", exc)


async def _handle_lora_ack(db_msg_id: str, crc_ok: bool, store: MensajeStore) -> None:
    """Cuando llega un ACK: actualiza el estado del mensaje saliente."""
    estado = "confirmado" if crc_ok else "error_rx"
    await store.actualizar(db_msg_id, estado=estado,
                           timestamp_confirmacion=datetime.now(timezone.utc))
    msg = store.obtener(db_msg_id)
    if msg:
        await ws_manager.broadcast("estado_actualizado", msg.dict_ui())


app = FastAPI(
    title="Meshstatic",
    description="Compresión lossless LM sobre LoRa",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mensajes_router.router)
app.include_router(ws_router.router)
app.include_router(canales_router.router)
app.include_router(lora_router.router)
app.include_router(firmware_router.router)
app.include_router(spellcheck_router.router)
app.include_router(archivo_router.router)

if WEB_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_PATH)), name="static")


@app.get("/", include_in_schema=False)
async def ui():
    index = WEB_PATH / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"mensaje": "Meshstatic API", "docs": "/docs"}
