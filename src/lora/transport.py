"""
Transporte LoRa con fragmentación automática y ACK de aplicación.

── Protocolo de fragmentación (4 bytes de header por paquete) ──────────────
  byte 0: msg_id  — ID del mensaje (0-253, circular; 254 reservado para ACK)
  byte 1: seq     — índice del fragmento (0-based)
  byte 2: total   — total de fragmentos (≥ 1 para datos; 0 señala que es ACK)
  byte 3: len     — bytes de datos en este fragmento (o crc_ok en ACK)

Paquete de datos:
  [msg_id 0-253][seq][total ≥ 1][len][...datos...]

Paquete ACK (4 bytes, sin datos):
  [acked_msg_id][0][0][0x01=ok | 0x00=crc_fail]
  total == 0 es el discriminador: nunca puede ser cero en datos reales.

── Modos de módulo ──────────────────────────────────────────────────────────
  at_rylr    RYLR896/RYLR406 con AT commands (más común en DIY)
  transparent Módulos que pasan bytes crudos por el puerto serie

── Flujo ACK ────────────────────────────────────────────────────────────────
  Emisor:
    lora_msg_id = await transport.send(payload, dest)
    transport.register_pending(lora_msg_id, db_msg_id)

  Receptor (al recibir mensaje completo):
    callback(payload, src_addr, lora_msg_id)
    → decodifica → await transport.send_ack(lora_msg_id, crc_ok, src_addr)

  Emisor (al recibir ACK):
    _handle_ack(acked_id, crc_ok)
    → busca db_msg_id en _pending → llama on_ack callback
    → store actualiza estado a "confirmado" o "error_rx"
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from typing import Callable, Dict, List, Optional

from .config import LoRaConfig

log = logging.getLogger(__name__)

# Callbacks
# (payload_bytes, src_addr, lora_msg_id)
RecvCallback = Callable[[bytes, int, int], None]
# (db_msg_id, crc_ok)
AckCallback  = Callable[[str, bool], None]

# msg_id 0-253 para datos; 254 está reservado (no se usa para evitar colisiones)
_MSG_ID_MAX = 254


class FragmentBuffer:
    """Reensambla los fragmentos de un mensaje mientras llegan."""

    TIMEOUT = 15.0

    def __init__(self, total: int, src: int):
        self.total  = total
        self.src    = src   # dirección LoRa del emisor (para el ACK)
        self.chunks: Dict[int, bytes] = {}
        self._ts    = time.monotonic()

    def add(self, seq: int, data: bytes) -> None:
        self.chunks[seq] = data

    def complete(self) -> bool:
        return len(self.chunks) == self.total

    def assemble(self) -> bytes:
        return b"".join(self.chunks[i] for i in range(self.total))

    def expired(self) -> bool:
        return (time.monotonic() - self._ts) > self.TIMEOUT


class LoRaTransport:
    """
    Capa de transporte LoRa sobre puerto serie.
    El loop de recepción corre como tarea asyncio en segundo plano.
    """

    def __init__(self, config: LoRaConfig):
        self.config              = config
        self._ser                = None
        self._tcp_reader: Optional[asyncio.StreamReader] = None
        self._tcp_writer: Optional[asyncio.StreamWriter] = None
        self._connected          = False
        self._error: Optional[str]  = None
        self._rssi_last: Optional[int] = None
        self._snr_last:  Optional[int] = None
        self._msg_id             = 0
        self._recv_buf: Dict[int, FragmentBuffer] = {}
        self._pending:  Dict[int, str] = {}   # lora_msg_id → db_msg_id
        self._recv_callback: Optional[RecvCallback] = None
        self._ack_callback:  Optional[AckCallback]  = None
        self._loop_task: Optional[asyncio.Task] = None

    # ── Estado ────────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    def status(self) -> dict:
        return {
            "enabled":    self.config.enabled,
            "connected":  self._connected,
            "port":       self.config.port,
            "sf":         self.config.sf,
            "bw":         self.config.bw,
            "tx_power":   self.config.tx_power,
            "address":    self.config.address,
            "modo":       self.config.modo,
            "rssi_last":  self._rssi_last,
            "snr_last":   self._snr_last,
            "error":      self._error,
        }

    # ── Conexión ──────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        if self.config.modo == "tcp":
            return await self._connect_tcp()
        if not self.config.port:
            self._error = "Sin puerto configurado"
            return False
        try:
            import serial
            self._ser = await asyncio.to_thread(
                serial.Serial,
                self.config.port,
                self.config.baudrate,
                timeout=1.0,
            )
            if self.config.modo == "at_rylr":
                await self._init_rylr()
            self._connected = True
            self._error     = None
            log.info("LoRa conectado: %s SF%d BW_idx=%d addr=%d",
                     self.config.port, self.config.sf,
                     self.config.bw, self.config.address)
            self._loop_task = asyncio.create_task(self._recv_loop())
            return True
        except ImportError:
            self._error = "pyserial no instalado"
            log.warning(self._error)
            return False
        except Exception as exc:
            self._error     = str(exc)
            self._connected = False
            log.warning("LoRa no disponible: %s", exc)
            return False

    async def _connect_tcp(self) -> bool:
        host = self.config.tcp_host
        port = self.config.tcp_port
        if not host:
            self._error = "Sin host TCP configurado"
            return False
        try:
            self._tcp_reader, self._tcp_writer = await asyncio.open_connection(host, port)
            self._connected = True
            self._error     = None
            log.info("LoRa TCP conectado: %s:%d SF%d addr=%d",
                     host, port, self.config.sf, self.config.address)
            self._loop_task = asyncio.create_task(self._recv_loop())
            return True
        except Exception as exc:
            self._error     = str(exc)
            self._connected = False
            log.warning("LoRa TCP no disponible: %s", exc)
            return False

    async def disconnect(self) -> None:
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        if self._tcp_writer:
            try:
                self._tcp_writer.close()
                await self._tcp_writer.wait_closed()
            except Exception:
                pass
            self._tcp_reader = None
            self._tcp_writer = None
        if self._ser:
            try:
                await asyncio.to_thread(self._ser.close)
            except Exception:
                pass
        self._connected = False
        log.info("LoRa desconectado")

    # ── AT commands RYLR896 ───────────────────────────────────────────────────

    async def _at(self, cmd: str) -> str:
        await asyncio.to_thread(self._ser.write, f"{cmd}\r\n".encode())
        await asyncio.sleep(0.15)
        raw = await asyncio.to_thread(self._ser.read_all)
        return raw.decode(errors="ignore").strip()

    async def _init_rylr(self) -> None:
        await self._at("AT")
        await self._at(f"AT+ADDRESS={self.config.address}")
        await self._at(f"AT+NETWORKID={self.config.network_id}")
        r = await self._at(
            f"AT+PARAMETER={self.config.sf},{self.config.bw},{self.config.cr},12"
        )
        await self._at(f"AT+CRFOP={self.config.tx_power}")
        await asyncio.sleep(0.2)
        log.debug("RYLR init: %s", r)

    # ── Envío de datos ────────────────────────────────────────────────────────

    async def send(self, payload: bytes, dest_address: int = 0) -> int:
        """
        Fragmenta y envía el payload. Retorna el lora_msg_id asignado
        (necesario para registrar el pending ACK).
        """
        if not self._connected:
            raise RuntimeError("LoRa no conectado")

        data_per_frag = max(1, self.config.max_chunk - 4)
        frags: List[bytes] = [
            payload[i: i + data_per_frag]
            for i in range(0, len(payload), data_per_frag)
        ]
        total  = len(frags)
        msg_id = self._msg_id
        self._msg_id = (self._msg_id + 1) % _MSG_ID_MAX

        log.debug("LoRa TX: %dB → %d frag (msg_id=%d dest=%d)",
                  len(payload), total, msg_id, dest_address)

        for seq, chunk in enumerate(frags):
            header = struct.pack("BBBB", msg_id, seq, total, len(chunk))
            await self._send_frame(header + chunk, dest_address)
            if seq < total - 1:
                await asyncio.sleep(0.5)

        return msg_id

    # ── Envío de ACK ─────────────────────────────────────────────────────────

    async def send_ack(self, acked_msg_id: int, crc_ok: bool, dest: int) -> None:
        """Envía un paquete ACK de 4 bytes al nodo emisor."""
        # total=0 discrimina ACK de datos; byte 3 lleva el resultado CRC
        frame = struct.pack("BBBB", acked_msg_id, 0, 0, 0x01 if crc_ok else 0x00)
        await self._send_frame(frame, dest)
        log.debug("LoRa ACK TX: msg_id=%d crc_ok=%s dest=%d",
                  acked_msg_id, crc_ok, dest)

    async def _send_frame(self, frame: bytes, dest: int) -> None:
        if self.config.modo == "tcp":
            # Protocolo: [dest:2 BE][len:2 BE][payload...]
            hdr = struct.pack(">HH", dest, len(frame))
            self._tcp_writer.write(hdr + frame)
            await self._tcp_writer.drain()
        elif self.config.modo == "at_rylr":
            hex_d = frame.hex().upper()
            resp  = await self._at(f"AT+SEND={dest},{len(hex_d)},{hex_d}")
            if "+ERR" in resp:
                log.warning("LoRa AT+SEND error: %s", resp)
        else:
            await asyncio.to_thread(self._ser.write, frame)

    # ── Pending ACKs ─────────────────────────────────────────────────────────

    def register_pending(self, lora_msg_id: int, db_msg_id: str) -> None:
        """Asocia un lora_msg_id con el ID de mensaje en la base de datos."""
        self._pending[lora_msg_id] = db_msg_id

    def on_receive(self, callback: RecvCallback) -> None:
        self._recv_callback = callback

    def on_ack(self, callback: AckCallback) -> None:
        """Registra el callback invocado cuando llega un ACK."""
        self._ack_callback = callback

    def _handle_ack(self, acked_id: int, crc_ok: bool) -> None:
        db_id = self._pending.pop(acked_id, None)
        if db_id is None:
            log.debug("LoRa ACK RX: msg_id=%d sin pending (ya procesado o tardío)", acked_id)
            return
        log.debug("LoRa ACK RX: msg_id=%d db_id=%s crc_ok=%s", acked_id, db_id, crc_ok)
        if self._ack_callback:
            self._ack_callback(db_id, crc_ok)

    # ── Loop de recepción ─────────────────────────────────────────────────────

    async def _recv_loop(self) -> None:
        log.info("LoRa recv_loop iniciado (modo=%s)", self.config.modo)
        if self.config.modo == "tcp":
            await self._recv_loop_tcp()
            return
        while True:
            try:
                if self.config.modo == "at_rylr":
                    line = await asyncio.to_thread(self._ser.readline)
                    if not line:
                        await asyncio.sleep(0.05)
                        continue
                    text = line.decode(errors="ignore").strip()
                    if text.startswith("+RCV="):
                        frame, src = self._parse_rylr(text)
                        if frame:
                            self._process_frame(frame, src)
                else:
                    hdr = await asyncio.to_thread(self._ser.read, 4)
                    if len(hdr) < 4:
                        await asyncio.sleep(0.05)
                        continue
                    _, _, _, data_len = struct.unpack("BBBB", hdr)
                    data = await asyncio.to_thread(self._ser.read, data_len)
                    self._process_frame(hdr + data, 0)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("LoRa recv error: %s", exc)
                await asyncio.sleep(0.5)

    async def _recv_loop_tcp(self) -> None:
        """
        Protocolo entrante desde el gateway ESP32S3:
          [src:2 BE][rssi:2 BE signed][snr:1 signed][len:2 BE][payload...]
        El payload ES el frame de fragmentación de Meshstatic.
        """
        while True:
            try:
                hdr = await self._tcp_reader.readexactly(7)
                src  = struct.unpack(">H",  hdr[0:2])[0]
                rssi = struct.unpack(">h",  hdr[2:4])[0]   # signed
                snr  = struct.unpack("b",   hdr[4:5])[0]   # signed
                plen = struct.unpack(">H",  hdr[5:7])[0]
                payload = await self._tcp_reader.readexactly(plen)
                self._rssi_last = rssi
                self._snr_last  = snr
                self._process_frame(payload, src)
            except asyncio.CancelledError:
                break
            except asyncio.IncompleteReadError:
                log.warning("LoRa TCP: conexión cerrada por el gateway")
                self._connected = False
                break
            except Exception as exc:
                log.warning("LoRa TCP recv error: %s", exc)
                await asyncio.sleep(0.5)

    def _parse_rylr(self, line: str):
        try:
            rest  = line[5:]
            parts = rest.split(",")
            src   = int(parts[0])
            hex_d = parts[2]
            if len(parts) > 3:
                self._rssi_last = int(parts[3])
            if len(parts) > 4:
                self._snr_last  = int(parts[4])
            return bytes.fromhex(hex_d), src
        except Exception as exc:
            log.warning("LoRa parse error: %s — %s", exc, line)
            return None, 0

    def _process_frame(self, frame: bytes, src: int) -> None:
        if len(frame) < 4:
            return
        msg_id, seq, total, data_len = struct.unpack("BBBB", frame[:4])

        # ── ACK recibido ──────────────────────────────────────────────────────
        if total == 0:
            crc_ok = (data_len == 0x01)
            self._handle_ack(msg_id, crc_ok)
            return

        # ── Datos ─────────────────────────────────────────────────────────────
        data = frame[4: 4 + data_len]

        # Limpiar buffers vencidos
        for mid in list(self._recv_buf):
            if self._recv_buf[mid].expired():
                log.warning("LoRa: fragmento vencido msg_id=%d", mid)
                del self._recv_buf[mid]

        if total == 1:
            if self._recv_callback:
                self._recv_callback(data, src, msg_id)
            return

        if msg_id not in self._recv_buf:
            self._recv_buf[msg_id] = FragmentBuffer(total, src)
        buf = self._recv_buf[msg_id]
        buf.add(seq, data)

        if buf.complete():
            payload = buf.assemble()
            buf_src = buf.src
            del self._recv_buf[msg_id]
            log.debug("LoRa RX: %dB msg_id=%d src=%d", len(payload), msg_id, buf_src)
            if self._recv_callback:
                self._recv_callback(payload, buf_src, msg_id)
