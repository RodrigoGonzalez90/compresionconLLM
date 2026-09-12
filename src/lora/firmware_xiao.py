"""
Template del firmware Arduino para XIAO ESP32S3 + Wio-SX1262.
Se genera con los parámetros LoRa actuales del nodo + credenciales WiFi.

Protocolo TCP (Meshstatic ↔ ESP32S3):
  TX  PC→ESP:  [dest:2 BE][len:2 BE][payload...]
  RX  ESP→PC:  [src:2 BE][rssi:2 BE signed][snr:1 signed][len:2 BE][payload...]

El payload es el frame de fragmentación de Meshstatic (4-byte header + data).
El firmware añade un header LoRa de 4 bytes [dest:2][src:2] en cada paquete
para que los nodos puedan filtrar por dirección.
"""

# BW index → kHz (mismo que LoRaConfig.bw)
_BW_KHZ = {7: 125.0, 8: 250.0, 9: 500.0}
# CR RYLR (1-4) → denominador RadioLib (5-8)
_CR_RLIB = {1: 5, 2: 6, 3: 7, 4: 8}

TEMPLATE = """\
// =============================================================
//  Meshstatic Gateway — XIAO ESP32S3 + Wio-SX1262
//  Generado por Meshstatic v2  |  {ts}
//
//  Instrucciones:
//  1. Abrí Arduino IDE
//  2. Agregá soporte para ESP32: https://docs.espressif.com/projects/arduino-esp32
//  3. Instalá la librería "RadioLib" desde el Gestor de Librerías
//  4. Seleccioná la placa: "XIAO_ESP32S3"
//  5. Flasheá este archivo
//  6. En Meshstatic → Configurar LoRa → Modo TCP → IP {tcp_host_hint}:{tcp_port}
// =============================================================

#include <RadioLib.h>
#include <WiFi.h>

// ── WiFi ───────────────────────────────────────────────────────────────────
const char*    WIFI_SSID  = "{ssid}";
const char*    WIFI_PASS  = "{password}";
const uint16_t TCP_PORT   = {tcp_port};

// ── LoRa ───────────────────────────────────────────────────────────────────
const float    LORA_FREQ  = {freq_mhz}f;   // MHz  (915.0 = América del Sur)
const float    LORA_BW    = {bw_khz}f;     // kHz
const uint8_t  LORA_SF    = {sf};
const uint8_t  LORA_CR    = {cr_rlib};     // denominador (5 = 4/5)
const int8_t   LORA_PWR   = {tx_power};    // dBm
const uint16_t LORA_ADDR  = {address};     // dirección de ESTE gateway
const uint8_t  LORA_SYNC  = {sync_word};   // sync word (= network_id)

// ── Pines XIAO ESP32S3 + Wio-SX1262 ───────────────────────────────────────
// Si no funciona, verificá el pinout en: https://wiki.seeedstudio.com/wio_sx1262
#define PIN_NSS   41
#define PIN_DIO1  39
#define PIN_RST   42
#define PIN_BUSY  40

SX1262 radio = new Module(PIN_NSS, PIN_DIO1, PIN_RST, PIN_BUSY);

WiFiServer server(TCP_PORT);
WiFiClient client;

volatile bool rxReady = false;
void IRAM_ATTR onDio1() {{ rxReady = true; }}

// ── Setup ──────────────────────────────────────────────────────────────────
void setup() {{
  Serial.begin(115200);
  delay(500);
  Serial.println("\\n[Meshstatic GW] Iniciando...");

  // WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  for (int i = 0; WiFi.status() != WL_CONNECTED && i < 40; i++) {{
    delay(500); Serial.print(".");
  }}
  if (WiFi.status() != WL_CONNECTED) {{
    Serial.println(" ERROR - reiniciando");
    ESP.restart();
  }}
  Serial.println(" OK  IP: " + WiFi.localIP().toString());

  // LoRa SX1262
  int st = radio.begin(LORA_FREQ, LORA_BW, LORA_SF, LORA_CR, LORA_SYNC, LORA_PWR);
  if (st != RADIOLIB_ERR_NONE) {{
    Serial.println("ERROR LoRa: " + String(st));
    ESP.restart();
  }}
  radio.setDio1Action(onDio1);
  radio.startReceive();
  Serial.println("LoRa  OK  SF" + String(LORA_SF) +
                 " BW" + String((int)LORA_BW) + "kHz" +
                 " addr=" + String(LORA_ADDR));

  // TCP server
  server.begin();
  Serial.println("TCP   OK  :" + String(TCP_PORT));
}}

// ── Loop ───────────────────────────────────────────────────────────────────

// Protocolo LoRa on-air:  [dest:2 BE][src:2 BE][payload...]
// Protocolo TCP TX:       [dest:2 BE][len:2 BE][payload...]
// Protocolo TCP RX:       [src:2 BE][rssi:2 BE][snr:1][len:2 BE][payload...]

void loop() {{
  // Aceptar cliente Meshstatic
  if (!client || !client.connected()) {{
    WiFiClient nc = server.available();
    if (nc) {{
      client = nc;
      client.setNoDelay(true);
      Serial.println("Cliente: " + client.remoteIP().toString());
    }}
  }}

  // ── TCP → LoRa ──────────────────────────────────────────────────────────
  if (client && client.connected() && client.available() >= 4) {{
    uint8_t hdr[4];
    if (client.readBytes(hdr, 4) < 4) return;
    uint16_t dest = ((uint16_t)hdr[0] << 8) | hdr[1];
    uint16_t plen = ((uint16_t)hdr[2] << 8) | hdr[3];
    if (plen == 0 || plen > 248) {{ return; }}

    uint8_t payload[252];
    if ((uint16_t)client.readBytes(payload, plen) < plen) return;

    // Envolver: [dest:2][LORA_ADDR:2][payload...]
    uint8_t pkt[256];
    pkt[0] = (dest >> 8) & 0xFF;
    pkt[1] = dest & 0xFF;
    pkt[2] = (LORA_ADDR >> 8) & 0xFF;
    pkt[3] = LORA_ADDR & 0xFF;
    memcpy(pkt + 4, payload, plen);
    uint16_t total = plen + 4;

    radio.clearDio1Action();
    int s = radio.transmit(pkt, total);
    if (s != RADIOLIB_ERR_NONE)
      Serial.println("TX err: " + String(s));
    radio.setDio1Action(onDio1);
    radio.startReceive();
  }}

  // ── LoRa → TCP ──────────────────────────────────────────────────────────
  if (rxReady) {{
    rxReady = false;
    uint8_t pkt[256];
    size_t  total = 0;
    int     st    = radio.readData(pkt, total);
    radio.startReceive();

    if (st != RADIOLIB_ERR_NONE || total < 4) return;

    uint16_t dest = ((uint16_t)pkt[0] << 8) | pkt[1];
    uint16_t src  = ((uint16_t)pkt[2] << 8) | pkt[3];

    // Filtrar: solo procesar si es para este nodo o broadcast
    if (dest != LORA_ADDR && dest != 0xFFFF) return;

    uint16_t plen = (uint16_t)(total - 4);
    if (!client || !client.connected()) return;

    int16_t rssi = (int16_t)radio.getRSSI();
    int8_t  snr  = (int8_t)radio.getSNR();

    // Header TCP: [src:2][rssi:2 signed][snr:1 signed][len:2]
    uint8_t hdr[7];
    hdr[0] = (src  >> 8) & 0xFF;
    hdr[1] = src  & 0xFF;
    hdr[2] = (rssi >> 8) & 0xFF;
    hdr[3] = rssi & 0xFF;
    hdr[4] = (uint8_t)snr;
    hdr[5] = (plen >> 8) & 0xFF;
    hdr[6] = plen & 0xFF;
    client.write(hdr, 7);
    client.write(pkt + 4, plen);
  }}
}}
"""


def render(
    ssid: str,
    password: str,
    freq_mhz: float,
    sf: int,
    bw: int,
    cr: int,
    tx_power: int,
    address: int,
    network_id: int,
    tcp_port: int,
    tcp_host_hint: str = "<IP del gateway>",
) -> str:
    from datetime import datetime, timezone
    bw_khz   = _BW_KHZ.get(bw, 125.0)
    cr_rlib  = _CR_RLIB.get(cr, 5)
    sync_word = network_id & 0xFF
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return TEMPLATE.format(
        ssid=ssid,
        password=password,
        freq_mhz=freq_mhz,
        bw_khz=bw_khz,
        sf=sf,
        cr_rlib=cr_rlib,
        tx_power=tx_power,
        address=address,
        sync_word=f"0x{sync_word:02X}",
        tcp_port=tcp_port,
        tcp_host_hint=tcp_host_hint,
        ts=ts,
    )
