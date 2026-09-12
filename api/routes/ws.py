"""
WebSocket: canal de tiempo real para actualizaciones de mensajes.
El servidor hace broadcast a todos los clientes conectados cuando:
  - Llega un mensaje nuevo
  - Un mensaje cambia de estado
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class WSManager:
    """Gestiona conexiones WebSocket activas y hace broadcast."""

    def __init__(self):
        self._activos: set[WebSocket] = set()

    async def conectar(self, ws: WebSocket):
        await ws.accept()
        self._activos.add(ws)

    def desconectar(self, ws: WebSocket):
        self._activos.discard(ws)

    async def broadcast(self, evento: str, data: dict):
        """Envía un evento JSON a todos los clientes conectados."""
        payload = {"evento": evento, "data": data}
        muertos: set[WebSocket] = set()
        for ws in self._activos:
            try:
                await ws.send_json(payload)
            except Exception:
                muertos.add(ws)
        self._activos -= muertos

    @property
    def n_clientes(self) -> int:
        return len(self._activos)


# Instancia global compartida
manager = WSManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.conectar(websocket)
    st = websocket.app.state
    await websocket.send_json({
        "evento": "bienvenida",
        "data": {
            "node_id": st.node_id,
            "backend": st.backend,
            "clientes_conectados": manager.n_clientes,
        },
    })
    try:
        while True:
            # Mantener la conexión viva; el servidor solo hace push
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.desconectar(websocket)
