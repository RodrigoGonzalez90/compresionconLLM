#!/usr/bin/env bash
# run_docker.sh — Meshstatic: un nodo, un Docker
set -euo pipefail

BOLD='\033[1m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${CYAN}[meshstatic]${NC} $*"; }
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

command -v docker &>/dev/null || err "Docker no está instalado."
docker info &>/dev/null       || err "El daemon de Docker no está corriendo."

# Leer puerto del .env para mostrarlo en los mensajes
PORT=$(grep '^PORT=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo 8001)
NODE_ID=$(grep '^NODE_ID=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo nodo-1)
MODELO=$(grep '^MODELO=' .env 2>/dev/null | cut -d= -f2 | tr -d ' ' || echo phi3:mini)

usage() {
    echo -e "${BOLD}Meshstatic — Un nodo, un Docker${NC}"
    echo ""
    echo "Uso: $0 <comando>"
    echo ""
    echo "  setup       Build + levanta Ollama + descarga modelo + inicia el nodo"
    echo "  up          Levanta el nodo (imagen ya construida)"
    echo "  down        Detiene el nodo"
    echo "  build       Reconstruye la imagen"
    echo "  logs        Sigue los logs"
    echo "  pull        Descarga/actualiza el modelo LLM en Ollama"
    echo "  jupyter     Levanta el nodo + Jupyter Lab (:8888)"
    echo "  open        Abre la UI en el navegador"
    echo "  clean       Elimina contenedores, imagen y volúmenes"
    echo ""
    echo -e "${BOLD}Config actual (.env):${NC}"
    echo "  NODE_ID = ${NODE_ID}"
    echo "  PORT    = ${PORT}"
    echo "  MODELO  = ${MODELO}"
    echo ""
    echo -e "${BOLD}Para correr un segundo nodo:${NC}"
    echo "  1. Copiá esta carpeta a otra ubicación"
    echo "  2. Editá .env: cambiá PORT y OLLAMA_PORT para que no choquen"
    echo "  3. Ejecutá $0 setup en esa carpeta"
    exit 0
}

# Detecta la IP de este equipo en la red local (para que los pares puedan responderle).
# Se recalcula en cada arranque, así un cambio de IP por DHCP se corrige con ./run_docker.sh up.
# Para fijarla a mano: HOST_IP=192.168.1.50 ./run_docker.sh up   (o HOST_IP= en .env)
detectar_ip() {
    if [ -n "${HOST_IP:-}" ] || grep -q '^HOST_IP=.\+' .env 2>/dev/null; then return; fi
    local ip=""
    if command -v ip >/dev/null 2>&1; then
        ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
    fi
    if [ -z "$ip" ] && command -v powershell.exe >/dev/null 2>&1; then
        ip=$(powershell.exe -NoProfile -Command "(Get-NetIPConfiguration | Where-Object { \$_.IPv4DefaultGateway -ne \$null -and \$_.NetAdapter.Status -eq 'Up' } | Select-Object -First 1).IPv4Address.IPAddress" 2>/dev/null | tr -d '\r')
    fi
    if [ -z "$ip" ] && command -v ipconfig >/dev/null 2>&1; then
        ip=$(ipconfig getifaddr en0 2>/dev/null || true)
    fi
    if [ -z "$ip" ] && command -v hostname >/dev/null 2>&1; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    if [ -n "$ip" ]; then
        export HOST_IP="$ip"
        echo -e "${BOLD}IP de la red:${NC} $HOST_IP  (los pares te responden en http://$HOST_IP:${PORT})"
    else
        echo -e "${YELLOW}No se pudo detectar la IP de la red; los pares tendrán que indicarla a mano.${NC}"
    fi
}

cmd_setup() {
    detectar_ip
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║   MESHSTATIC — SETUP                     ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════╝${NC}"
    echo ""
    log "Nodo: ${NODE_ID}  |  Puerto: ${PORT}  |  Modelo: ${MODELO}"
    echo ""

    echo -e "${BOLD}[1/3] Construyendo imagen...${NC}"
    warn "Primera vez: descarga dependencias (~1.5 GB, ~10 min)."
    docker compose build
    ok "Imagen lista."
    echo ""

    echo -e "${BOLD}[2/3] Levantando nodo + Ollama...${NC}"
    docker compose up -d
    ok "Servicios corriendo."
    echo ""

    echo -e "${BOLD}[3/3] Descargando modelo ${MODELO} en Ollama...${NC}"
    warn "Puede tardar varios minutos (~2 GB para phi3:mini)."
    local intentos=0
    until docker compose exec ollama ollama list &>/dev/null; do
        sleep 3; intentos=$((intentos+1))
        [[ $intentos -gt 30 ]] && err "Ollama no respondió."
    done
    docker compose exec ollama ollama pull "${MODELO}"
    ok "Modelo ${MODELO} listo."
    docker compose restart nodo
    echo ""

    echo -e "${GREEN}✓ Nodo ${NODE_ID} listo en http://localhost:${PORT}${NC}"
    echo ""
    echo "  Para conectarte a otro nodo: abrí la UI y creá un canal con la URL del otro nodo."
}

cmd_up() {
    detectar_ip
    docker compose up -d
    ok "Nodo ${NODE_ID} corriendo en http://localhost:${PORT}"
}

cmd_down() {
    docker compose down
    ok "Nodo ${NODE_ID} detenido."
}

cmd_build() {
    warn "Reconstruyendo imagen..."
    docker compose build
    ok "Imagen lista."
}

cmd_logs() {
    docker compose logs --follow --tail=60
}

cmd_pull() {
    log "Descargando modelo ${MODELO}..."
    docker compose exec ollama ollama pull "${MODELO}"
    ok "Modelo listo."
}

cmd_jupyter() {
    docker compose --profile jupyter up -d
    ok "Jupyter Lab corriendo en http://localhost:8888"
    echo "  Notebook: notebooks/analisis_lossless_v2.ipynb"
}

cmd_open() {
    local url="http://localhost:${PORT}"
    if command -v xdg-open &>/dev/null; then xdg-open "$url" &
    elif command -v open &>/dev/null;    then open "$url"
    else log "Abrí: $url"; fi
}

cmd_clean() {
    warn "Se eliminarán contenedores, imagen y volúmenes de este nodo."
    read -r -p "¿Continuar? [s/N] " c
    [[ "${c,,}" == "s" ]] || { log "Cancelado."; exit 0; }
    docker compose --profile jupyter down -v 2>/dev/null || true
    docker image rm meshstatic:latest 2>/dev/null && ok "Imagen eliminada." || true
    ok "Limpieza completada."
}

case "${1:-help}" in
    setup)   cmd_setup ;;
    up)      cmd_up ;;
    down)    cmd_down ;;
    build)   cmd_build ;;
    logs)    cmd_logs ;;
    pull)    cmd_pull ;;
    jupyter) cmd_jupyter ;;
    open)    cmd_open ;;
    clean)   cmd_clean ;;
    help|-h|--help) usage ;;
    *) err "Comando desconocido: '$1'. Usá '$0 help'." ;;
esac
