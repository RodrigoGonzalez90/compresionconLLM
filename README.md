# Meshstatic v2 — Compresión lossless guiada por LLM para LoRa

Meshstatic es un sistema de comunicación que comprime mensajes de texto usando un modelo de lenguaje (LLM) corriendo localmente en cada nodo. El objetivo es reducir radicalmente el tamaño de los datos transmitidos para hacerlos viables en redes de baja velocidad como **LoRa**, donde cada byte tiene un costo real en tiempo de antena, energía y probabilidad de colisión.

**Resultado validado en 25 mensajes (100% lossless):** mensajes en lenguaje natural de más de 70 bytes se comprimen entre 50% y 80%, superando a gzip en todos los rangos de tamaño medio y largo.

---

## El problema: LoRa y el ancho de banda

LoRa (Long Range) es una tecnología de radio que permite comunicaciones de varios kilómetros con muy poca energía. La contrapartida es una velocidad de transferencia extremadamente baja: entre 250 y 5500 bits por segundo dependiendo del spreading factor (SF). Un mensaje de texto de 800 bytes puede tardar varios segundos en transmitirse y ocupa una fracción significativa del tiempo de canal disponible. En redes con múltiples nodos, esto genera colisiones y degrada la red.

| Spreading Factor | Velocidad útil | Alcance típico |
|---|---|---|
| SF7 | ~5.5 kbps | Corto alcance, máxima velocidad |
| SF10 | ~0.98 kbps | Alcance medio, equilibrio |
| SF12 | ~0.29 kbps | Máximo alcance, mínima velocidad |

Los algoritmos clásicos como gzip funcionan razonablemente bien a partir de ~40–50 bytes, pero su ahorro queda limitado porque solo explotan **repetición de secuencias dentro del mismo mensaje**. Para un texto de 130 bytes el ahorro de gzip es apenas −19%; para 800 bytes, −38%. Se necesita un enfoque que explote el **significado** del texto, no solo sus patrones locales.

---

## La solución: compresión guiada por modelo de lenguaje

La idea central: si dos nodos tienen el mismo modelo de lenguaje, el emisor no necesita enviar los tokens del mensaje — solo necesita enviar **el lugar que cada token ocupa en la lista de predicciones del modelo**.

### Cómo funciona paso a paso

Dado el mensaje: *"La historia de las telecomunicaciones comenzó con el telégrafo"*

**En el emisor (nodo 1):**

1. El texto se tokeniza con el vocabulario del modelo (qwen2.5:1.5b, ~152k tokens).
2. Para cada token, el modelo genera una distribución de probabilidad dado el contexto anterior. Se toman los **64 tokens más probables** (top-64).
3. Se busca el rank del token real dentro de ese top-64:
   - "historia" está en posición 0 → se codifica como `0` → **1 bit**.
   - "telecomunicaciones" está en posición 3 → se codifica como `10 10` → **4 bits**.
   - Si el token no está en el top-64, se codifica como **OOV** (vocab ID compacto, ~23 bits).
4. El payload resultante es una secuencia compacta de bits.

**En el receptor (nodo 2):**

1. Se recibe el payload comprimido.
2. Para cada posición, el receptor corre el **mismo modelo** con el **mismo contexto acumulado** y obtiene el mismo top-64.
3. Según el rank recibido, selecciona el token correspondiente. Para tokens OOV, lee el vocab ID del payload y lo convierte al string del token.
4. El texto se reconstruye exactamente, token por token, sin pérdida de información.

No se transmite el texto: se transmite **dónde mirar para encontrarlo**. El modelo actúa como un diccionario compartido de ~1 GB que nunca necesita enviarse.

### Esquema de codificación de bits

```
Rank 0       →  0                          (1 bit)
Rank 1–3     →  10 rr                      (4 bits)
Rank 4–15    →  110 rrrr                   (7 bits)
Rank 16–63   →  1110 rrrrrr               (10 bits)
OOV          →  11111 [vocab_id: 18 bits] (23 bits, vocab ID compacto)
```

Tokens muy predecibles (rank 0) cuestan solo 1 bit. Tokens impredecibles se incluyen como su ID numérico en el vocabulario del modelo (18 bits para ~152k tokens), mucho más compacto que UTF-8 crudo (~40–50 bits promedio).

### ¿Cuándo se usa cada código?

El umbral entre rank y OOV es **dinámico**: si la ventaja de logprob del token correcto sobre el siguiente candidato es demasiado pequeña, el ranking entre nodos podría diferir por acumulación de punto flotante, lo que rompería la reconstrucción. En ese caso, se prefiere OOV aunque el token esté en el top-64.

El umbral se escala con la **entropía de Shannon** de la distribución:

- **Baja entropía** (modelo muy seguro) → umbral pequeño → más ranks usados → mejor compresión.
- **Alta entropía** (distribución plana) → umbral grande → más OOV → reconstrucción más segura.

```
GAP_MIN = 0.001   # cuando el modelo es casi determinista
GAP_MAX = 0.10    # cuando la distribución es plana
umbral  = GAP_MIN + (H / H_MAX) * (GAP_MAX - GAP_MIN)
```

### Formato del payload

```
┌────────┬────────────────────┬───────────────┬─────────────────────────┐
│ 0x4C   │  CRC32 (4 bytes)   │ N tok (2 B)   │    bits de payload …    │
│ 1 byte │  big-endian uint32 │ big-endian u16│    (relleno a múlt. 8)  │
└────────┴────────────────────┴───────────────┴─────────────────────────┘
```

- **`0x4C`** (`'L'`) — byte marcador lossless.
- **CRC32** — checksum del texto original en UTF-8. Permite al receptor detectar divergencia silenciosa del LLM.
- **N tokens** — cantidad de tokens que el receptor debe decodificar.
- **bits de payload** — secuencia de ranks y tokens OOV codificados según el esquema de bits.

Si el largo del payload lossless supera el texto crudo, el marcador cambia a `0x52` (`'R'`) y el payload son simplemente los bytes UTF-8 del texto (**raw fallback adaptativo**). Esto ocurre automáticamente para mensajes cortos donde el overhead del header supera el ahorro.

#### ¿Qué es el CRC32 y por qué lo necesitamos?

CRC32 es un checksum rápido (4 bytes) calculado sobre el texto original. **No es para recuperación de errores** — LoRa ya tiene corrección de errores (FEC) a nivel de radio. Es un **detector de divergencia del KV cache**: si los dos nodos no reproducen exactamente el mismo estado interno del LLM, el texto reconstruido diverge silenciosamente. El CRC32 lo detecta. Si `CRC32(texto_reconstruido) ≠ CRC32_del_header`, el receptor reporta `crc_ok: false`.

---

## Resultados validados

Experimento con 25 mensajes en 5 rangos de tamaño y 4 categorías de contenido. Todos los mensajes se reconstruyeron exactamente (100% lossless).

### Compresión por rango de tamaño

| Rango | Bytes aprox. | Ratio LLM (media) | Ratio gzip (media) | Modo elegido |
|---|---|---|---|---|
| micro | < 20 B | −18.7% | −373.3% | `raw` (fallback automático) |
| corto | 20–70 B | +18.6% | −53.1% | `lossless` |
| medio | 70–200 B | +57.6% | +9.7% | `lossless` |
| largo | 200–500 B | +69.8% | +32.3% | `lossless` |
| muy_largo | > 500 B | +78.7% | +44.1% | `lossless` |

**Cómo leer el ratio:** positivo = ahorro de bytes, negativo = overhead. `-373%` para mensajes micro no es un error — significa que el paquete lossless pesa 4.7x más que el texto original (3 bytes de texto vs ~14 bytes de cabecera + bits), por eso el sistema cae automáticamente a modo `raw`.

### Punto de breakeven

El LLM empieza a comprimir (superar al raw) a partir de **~21 bytes**. gzip no supera al raw hasta ~50 bytes. Para textos entre 21 y 50 bytes, el LLM gana y gzip pierde.

### Compresión por tipo de contenido

| Tipo | Ratio LLM (media) | Interpretación |
|---|---|---|
| narrativo | +77.5% | Lenguaje natural fluido → muy predecible |
| técnico | +65.5% | Vocabulario especializado pero estructurado |
| operativo | +17.8% | Mensajes cortos y directos → breakeven variable |
| telemetría | +13.0% | Números y coordenadas → muchos OOV |

Los mensajes de telemetría (coordenadas GPS, valores numéricos, IPs) generan mayor porcentaje de OOV porque los números raramente coinciden con las predicciones del LM.

### Eficiencia del modelo de lenguaje

En un mensaje típico de texto en español, la distribución de ranks es aproximadamente:

| Bucket | Bits | % promedio de tokens |
|---|---|---|
| rank 0 | 1 bit | ~31% |
| rank 1–3 | 4 bits | ~19% |
| rank 4–15 | 7 bits | ~11% |
| rank 16–63 | 10 bits | ~1% |
| OOV | ~23 bits | ~37% |

**Costo efectivo promedio:** ~8 bits/token con LLM vs ~40 bits/token en UTF-8 crudo para mensajes largos.

El 37% de OOV es alto en mensajes cortos y de telemetría; baja significativamente en mensajes narrativos y operativos largos donde el modelo predice bien.

---

## Costos de procesamiento

El LLM tiene un costo computacional real: hay que correr inferencia por cada token. A diferencia de gzip (microsegundos), el encoder tarda entre 5 y 40 segundos dependiendo del tamaño del mensaje.

| Métrica | Descripción |
|---|---|
| `encode_ms` | Tiempo del nodo emisor en comprimir (inferencia token a token) |
| `decode_ms` | Tiempo del nodo receptor en descomprimir (misma inferencia) |
| `ms/token` | Costo marginal por token (constante ≈ 280 ms/token con Qwen 1.5B en CPU) |

**encode_ms ≈ decode_ms** porque ambos nodos ejecutan el mismo número de inferencias. La API expone ambos valores en cada mensaje.

### ¿Vale la pena el costo computacional?

La variable relevante para LoRa es el **tiempo total de uso del canal**. Si el ahorro en tiempo de vuelo (ToA) supera el costo de procesamiento, el sistema gana aunque tarde más en preparar el mensaje.

```
costo_neto = (encode_ms + decode_ms) − ahorro_toa

costo_neto < 0  →  el LLM gana incluso en tiempo total (para SF10+)
costo_neto > 0  →  el overhead de cómputo supera el ahorro (para mensajes cortos)
```

Para mensajes de más de 150 bytes con SF10 o SF12, el ahorro en tiempo de canal supera el costo de inferencia. Para SF7 (alta velocidad), el trade-off es menos favorable.

---

## Por qué funciona mejor que gzip

| Método | micro (7 B) | corto (38 B) | medio (156 B) | largo (416 B) | muy_largo (736 B) |
|---|---|---|---|---|---|
| Raw UTF-8 | 7 B (—) | 38 B (—) | 156 B (—) | 416 B (—) | 736 B (—) |
| gzip -9 | 27 B (−286%) | 57 B (−50%) | 138 B (+12%) | 282 B (+32%) | 411 B (+44%) |
| **Meshstatic LLM** | **8 B (raw)** | **31 B (+18%)** | **63 B (+60%)** | **125 B (+70%)** | **156 B (+79%)** |

gzip requiere repetición de secuencias dentro del mensaje para comprimir. Meshstatic explota la predictibilidad semántica del texto usando un LLM compartido — funciona incluso cuando no hay repetición local.

---

## Arquitectura del sistema

```
┌─────────────────────────────┐     LoRa radio      ┌─────────────────────────────┐
│         Nodo 1 (emisor)     │ ──────────────────▶  │       Nodo 2 (receptor)     │
│                             │   payload: 125 B      │                             │
│  Texto: 416 bytes           │                       │  payload: 125 bytes         │
│       ↓                     │                       │       ↓                     │
│  LMEncoder                  │                       │  LMDecoder                  │
│  ├─ tokenize_with_ids()     │                       │  ├─ reset() KV cache        │
│  ├─ reset() KV cache        │◀── ACK (4 bytes) ─────│  ├─ top_tokens(context) ×N  │
│  ├─ top_tokens(context) ×N  │   [msg_id][0][0][ok]  │  └─ rank → token            │
│  └─ rank/OOV → bits         │                       │       ↓                     │
│       ↓                     │                       │  Texto exacto + CRC check   │
│  [0x4C][CRC32][N][bits]     │                       │                             │
│  + fragmentación LoRa       │                       │  qwen2.5-1.5b Q4_K_M        │
│                             │                       │  (llama-cpp-python)         │
│  qwen2.5-1.5b Q4_K_M        │                       │                             │
└─────────────────────────────┘                       └─────────────────────────────┘
        ↑ USB serial / WiFi TCP
        │
┌───────────────────┐
│  RYLR896  (AT)    │  ← opción A: módulo serial con AT commands
│  ó                │
│  XIAO ESP32S3 +   │  ← opción B: gateway WiFi→LoRa
│  Wio-SX1262 (TCP) │     (firmware .ino descargable desde la UI)
└───────────────────┘
```

### Componentes principales

| Archivo | Responsabilidad |
|---|---|
| `src/llm_backend.py` | Abstracción del LLM. `LlamaCppBackend` descarga y corre el modelo con logprobs nativos. Expone `tokenize_with_ids()`, `id_to_token()`, `vocab_size`. |
| `src/encoder/lm_encoder.py` | `LMEncoder`: tokeniza con IDs, consulta el top-64 por token, codifica ranks en bits. Mide `encode_ms`. |
| `src/decoder/lm_decoder.py` | `LMDecoder`: lee bits, consulta el mismo top-64, recupera tokens. Mide `decode_ms`. |
| `src/lora/transport.py` | Capa de transporte LoRa: fragmentación, ACK de aplicación, modos serial y TCP. |
| `src/lora/config.py` | `LoRaConfig`: parámetros LoRa persistidos en `data/lora_config.json`. |
| `src/lora/firmware_xiao.py` | Template Arduino para gateway XIAO ESP32S3 + Wio-SX1262. |
| `src/channels/manager.py` | `GestorCanales`: gestión de canales (peers) con persistencia en `data/canales.json`. |
| `api/main.py` | FastAPI app. Ciclo de vida: carga el modelo, conecta el radio LoRa. |
| `api/store.py` | `MensajeStore`: CRUD de mensajes con persistencia en `data/mensajes.json`. |
| `api/routes/mensajes.py` | Endpoints de mensajes: enviar, recibir, listar, borrar. |
| `api/routes/canales.py` | Endpoints de canales: crear, listar, editar, borrar. |
| `api/routes/lora.py` | Endpoints de configuración y estado LoRa. |
| `api/routes/firmware.py` | `GET /api/firmware/xiao-esp32s3`: genera y descarga el firmware Arduino. |
| `api/routes/ws.py` | WebSocket para actualizaciones en tiempo real a la UI. |
| `api/routes/spellcheck.py` | `POST /api/spellcheck`: corrector ortográfico en español, filtra términos técnicos. |
| `web/index.html` | Interfaz web completa: chat responsive, canales, métricas, configuración LoRa. |
| `notebooks/analisis_lossless_v2.ipynb` | Análisis completo: compresión, costos de procesamiento, densidad de ranks, clustering por categoría. |

---

## Protocolo LoRa

### Fragmentación

Los mensajes grandes se fragmentan en chunks de hasta `max_chunk` bytes (default 240). Cada fragmento lleva un header de 4 bytes:

```
[msg_id: 1B][seq: 1B][total: 1B][len: 1B][...datos...]
```

- `msg_id` (0–253): identifica el mensaje completo (circular).
- `seq`: índice del fragmento (0-based).
- `total ≥ 1`: total de fragmentos. **`total = 0` reservado para ACK.**

### ACK de aplicación

Cuando el receptor decodifica correctamente un mensaje, envía un ACK de 4 bytes de vuelta al emisor:

```
[acked_msg_id][0][0][0x01=crc_ok | 0x00=crc_fail]
```

El discriminador es `total == 0`: los paquetes de datos siempre tienen `total ≥ 1`.

El emisor registra cada `lora_msg_id → db_msg_id` en un mapa de pendientes. Al recibir el ACK:
- `crc_ok = true` → estado del mensaje pasa a `"confirmado"`
- `crc_ok = false` → estado pasa a `"error_rx"` (divergencia de KV cache entre nodos)

### Modos de conexión

#### Modo serial AT (`at_rylr`)
Para módulos **REYAX RYLR896** o RYLR406 conectados por USB-UART (CH340/CP2102). El transporte usa AT commands: `AT+SEND=<dest>,<len>,<hex>` y parsea respuestas `+RCV=<src>,<len>,<hex>,<rssi>,<snr>`.

#### Modo TCP/WiFi (`tcp`)
Para el gateway **XIAO ESP32S3 + Wio-SX1262**. Protocolo binario sobre TCP:

```
PC → ESP32S3:  [dest: 2B BE][len: 2B BE][payload...]
ESP32S3 → PC:  [src: 2B BE][rssi: 2B signed][snr: 1B signed][len: 2B BE][payload...]
```

---

## Gateway WiFi: XIAO ESP32S3 + Wio-SX1262

Esta opción conecta la antena LoRa por WiFi en lugar de USB serial. El ESP32S3 actúa como puente entre TCP y LoRa.

### Hardware necesario

- Seeed Studio XIAO ESP32S3 (~$7)
- Seeed Studio Wio-SX1262 (~$12)
- Cable USB-C para programar

### Obtener el firmware

1. En la UI → ⚙ Configurar LoRa → Modo: **TCP/WiFi (XIAO ESP32S3)**
2. Hacer clic en **↓ Descargar firmware para ESP32S3**
3. Ingresar SSID y contraseña WiFi → **Descargar .ino**

O directamente via API:

```
GET /api/firmware/xiao-esp32s3?ssid=MiRed&password=MiClave
```

### Programar el ESP32S3

1. Instalar [Arduino IDE](https://www.arduino.cc/en/software)
2. Agregar soporte ESP32: `https://docs.espressif.com/projects/arduino-esp32`
3. Instalar librería **RadioLib** desde el Gestor de Librerías
4. Seleccionar placa: **XIAO_ESP32S3**
5. Abrir el `.ino` descargado → Subir
6. Abrir Monitor Serial (115200 baud) → anotar la IP asignada
7. En Meshstatic → Configurar LoRa → IP del gateway: la IP del ESP32S3

---

## Decisiones técnicas críticas

### 1. El decoder SIEMPRE llama `top_tokens()`, incluso para tokens OOV

Esta es la invariante más importante del sistema. El encoder llama `top_tokens(context)` para **cada** token. Si el decoder saltara esa llamada para tokens OOV, el KV cache divergiría desde el primer token ambiguo y todos los ranks siguientes se calcularían sobre un contexto diferente:

```
Encoder: top_tokens("") → top_tokens("La") → top_tokens("La historia") → ...
Decoder: top_tokens("") → [skip OOV] → top_tokens("La historia") ← ESTADO DIFERENTE
```

### 2. Umbral dinámico basado en entropía de Shannon

Cuando dos tokens tienen logprobs muy similares, el orden entre ellos puede variar entre instancias del modelo por diferencias de punto flotante acumuladas. Si el token se codifica como rank X pero el receptor lo ve en rank X+1, el texto diverge silenciosamente (sin que el encoder lo note).

Solución: si `logprob[rank] - logprob[rank+1] < umbral_dinámico`, el token se incluye como OOV. El umbral se escala con la entropía de la distribución: a mayor incertidumbre del modelo, mayor exigencia para aceptar un rank.

### 3. OOV como vocab ID compacto, no bytes UTF-8

Para tokens OOV, se transmite el **ID numérico del token en el vocabulario del modelo** (18 bits para ~152k tokens de Qwen) en lugar del string UTF-8 crudo (promedio 40–52 bits). El receptor usa `id_to_token(id)` para recuperar el string exacto. Ahorro por token OOV: ~55%.

```
OOV compacto: 11111 [id: 18 bits]   → 23 bits total (LlamaCpp)
```

### 4. `reset()` del KV cache antes de cada mensaje

Ambos nodos parten del mismo estado inicial para cada mensaje, independientemente del historial de llamadas anteriores. Sin esto, el contexto de un mensaje anterior contaminaría el siguiente.

### 5. `n_threads=1` y `logits_all=True`

`n_threads=1` elimina no-determinismo por paralelismo de punto flotante. `logits_all=True` es requerido por llama-cpp-python para habilitar el parámetro `logprobs`.

### 6. `n_ctx=512` en lugar del default 4096

El KV cache se reserva en memoria al cargar el modelo según `n_ctx`. Con el default de 4096, se alloca memoria para posiciones que nunca se usan (los mensajes nunca superan ~250 tokens). Reducirlo a 512 achica el KV cache ~8x, mejora la localidad de caché de CPU y acelera la carga — sin ningún impacto en calidad, ya que los mensajes individuales siempre entran dentro de esa ventana.

### 7. Los typos degradan la compresión

Un token que el modelo predice correctamente ocupa entre 1 y 10 bits. Un typo rompe esa predicción: el tokenizador BPE fragmenta la palabra mal escrita en subwords desconocidas que el modelo jamás hubiera predicho en ese contexto, y cada fragmento se transmite como OOV (~23 bits).

```
"equipos"  →  rank 1            →   4 bits   (predecible en contexto)
"eqipos"   →  "eq" + "ipos" OOV →  46 bits   (~11x más caro)
```

Para mitigar esto, la UI incluye un **corrector ortográfico automático** (`pyspellchecker`, español) que actúa 500 ms después de que el usuario deja de escribir. Las correcciones se aplican al texto antes de enviarlo; cada corrección aparece como un chip que el usuario puede revertir si la palabra era intencional (términos técnicos, nombres propios, etc.). El corrector ignora automáticamente palabras con números (`SF10`, `34.6037`), siglas en mayúsculas (`RSSI`, `ACK`) y palabras con guión (`nodo-beta`).

---

## Requisitos

- Docker Desktop (Windows/Mac) o Docker Engine (Linux)
- 4 GB de RAM mínimo por nodo (el modelo ocupa ~1 GB en Q4)
- Conexión a internet la primera vez (para descargar el modelo desde HuggingFace Hub)
- **Para LoRa serial**: módulo RYLR896 + adaptador USB-UART (CH340/CP2102)
- **Para LoRa WiFi**: XIAO ESP32S3 + Wio-SX1262, en la misma red WiFi que el servidor

---

## Instalación y uso

### Un solo nodo (para desarrollo)

```bash
cp .env.example .env   # editar NODE_ID, PORT si es necesario
./run_docker.sh setup  # build + levantar + descargar modelo
./run_docker.sh up     # iniciar (imagen ya construida)
./run_docker.sh logs   # ver logs en vivo
./run_docker.sh open   # abrir UI en el navegador
```

### Dos nodos en la misma máquina

**Nodo 1:**
```bash
NODE_ID=nodo-1
PORT=8001
BACKEND=llamacpp
MODELO=qwen2.5:1.5b
OWN_URL=http://host.docker.internal:8001

./run_docker.sh setup
```

**Nodo 2:**
```bash
NODE_ID=nodo-2
PORT=8002
BACKEND=llamacpp
MODELO=qwen2.5:1.5b
OWN_URL=http://host.docker.internal:8002

./run_docker.sh setup
```

Una vez ambos nodos están corriendo:
1. Abrir `http://localhost:8001`
2. Crear un canal hacia `http://host.docker.internal:8002`
3. Enviar un mensaje — la UI muestra bytes originales, bytes transmitidos, ratio, encode_ms y decode_ms
4. En `http://localhost:8002` aparece el mensaje recibido con el texto exacto

### Análisis de compresión

```bash
./run_docker.sh jupyter   # Jupyter Lab en :8888
# Abrir notebooks/analisis_lossless_v2.ipynb
```

El notebook ejecuta 25 mensajes de prueba y genera análisis de compresión, costos de procesamiento, distribución de ranks y clustering por tipo de contenido.

### Comandos disponibles

```
./run_docker.sh setup     # Build + levantar + descargar modelo
./run_docker.sh up        # Levantar (imagen ya construida)
./run_docker.sh down      # Detener
./run_docker.sh build     # Reconstruir imagen
./run_docker.sh logs      # Ver logs en vivo
./run_docker.sh jupyter   # Levantar Jupyter Lab en :8888
./run_docker.sh clean     # Eliminar contenedores y volúmenes
```

---

## API

### Mensajes

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/enviar` | POST | Comprime y envía un mensaje al canal seleccionado |
| `/api/recibir` | POST | Recibe, decodifica y almacena un mensaje entrante |
| `/api/mensajes` | GET | Lista todos los mensajes persistidos |
| `/api/mensajes/{id}` | DELETE | Elimina un mensaje por ID |
| `/api/mensajes` | DELETE | Elimina todos los mensajes |

**Respuesta de `/api/enviar`** (campos relevantes):

```json
{
  "bytes_original": 416,
  "bytes_tx": 125,
  "ratio_pct": 69.8,
  "n_tokens": 95,
  "encode_ms": 26400,
  "decode_ms": 25800,
  "encoding": "lossless/lossless",
  "rank_dist": {"r0": 31.0, "r1_3": 18.9, "r4_15": 11.3, "r16_63": 1.2, "oov": 37.6},
  "crc_ok": true
}
```

### Canales

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/canales` | GET | Lista los canales configurados |
| `/api/canales` | POST | Crea un nuevo canal |
| `/api/canales/{id}` | PATCH | Renombra o cambia la URL de un canal |
| `/api/canales/{id}` | DELETE | Elimina un canal |

### LoRa

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/lora/config` | GET | Lee la configuración LoRa actual |
| `/api/lora/config` | POST | Guarda configuración y reconecta el radio |
| `/api/lora/status` | GET | Estado de conexión, RSSI, SNR, errores |
| `/api/lora/puertos` | GET | Lista puertos seriales disponibles |

### Firmware y otros

| Endpoint | Método | Descripción |
|---|---|---|
| `/api/firmware/xiao-esp32s3` | GET | Genera y descarga el `.ino` para el gateway ESP32S3 (`?ssid=X&password=Y`) |
| `/api/estado` | GET | Estado del nodo: backend, modelo, estadísticas |
| `/ws` | WebSocket | Eventos: `nuevo_mensaje`, `estado_actualizado`, `canales_actualizados` |
| `/api/spellcheck` | POST | `{"texto": "..."}` → `{"correcciones": [...], "disponible": true}` |

---

## Persistencia de datos

| Archivo | Contenido |
|---|---|
| `data/mensajes.json` | Historial de mensajes (enviados y recibidos) |
| `data/canales.json` | Lista de canales / peers configurados |
| `data/lora_config.json` | Configuración del radio LoRa |

---

## Estados de un mensaje

| Estado | Descripción |
|---|---|
| `enviando` | En proceso de codificación y transmisión |
| `transmitido` | Fragmentos enviados, esperando ACK LoRa |
| `confirmado` | ACK recibido con `crc_ok = true` |
| `error_rx` | ACK recibido con `crc_ok = false` (divergencia de KV cache) |
| `error_tx` | Error de transmisión de red o LoRa |
| `recibido` | Mensaje entrante decodificado correctamente |

---

## Modelo de lenguaje

Por defecto se usa **qwen2.5-1.5b-instruct-q4_k_m.gguf** (Qwen, Alibaba): 1.5B parámetros, cuantización 4-bit, ~1 GB en disco y memoria. Vocabulario de ~152k tokens. La descarga es automática desde HuggingFace Hub la primera vez.

El sistema es compatible con cualquier modelo GGUF soportado por llama-cpp-python:

| Nombre Ollama | Repo HuggingFace | Archivo GGUF |
|---|---|---|
| `qwen2.5:1.5b` | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | `qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| `phi3:mini` | `microsoft/Phi-3-mini-4k-instruct-gguf` | `Phi-3-mini-4k-instruct-q4.gguf` |
| `llama3.2:1b` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| `tinyllama:1b` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |

Para cambiar el modelo: editar `MODELO` en `.env` y agregar la entrada correspondiente en `src/llm_backend.py:GGUF_MODELS`.

---

## Métricas expuestas en la UI

Para cada mensaje la UI muestra:

| Métrica | Descripción |
|---|---|
| **Bytes originales** | Tamaño del texto en UTF-8 |
| **Bytes TX** | Tamaño del payload transmitido (con header y fragmentación LoRa) |
| **Ratio** | Porcentaje de reducción (positivo = ahorro, negativo = overhead) |
| **Tokens** | Cantidad de tokens en el mensaje |
| **Encode ms** | Tiempo de compresión en el emisor |
| **Decode ms** | Tiempo de descompresión en el receptor |
| **Encoding** | `lossless/lossless` (LLM comprimió) o `lossless/raw` (fallback automático) |
| **Estado** | `confirmado` / `error_rx` / `transmitido` con ícono de ACK LoRa |
