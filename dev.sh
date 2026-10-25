#!/usr/bin/env bash
#
# Starts everything (LM Studio + LLM, bot, web client) and stops it all on Ctrl+C.
#
#   ./dev.sh          start; Ctrl+C stops everything and frees the RAM
#   ./dev.sh stop     clean up anything left running (e.g. after a crash)
#
# The LLM to load is read from the active `model="..."` line in server/bot.py.
#
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
LMS="$(command -v lms || echo "$HOME/.lmstudio/bin/lms")"
LLM_PORT=1234
BOT_PORT=7860
CLIENT_PORT=3000
LLM_TTL_SECS=1800 # safety net: LM Studio unloads the model after 30 min without use
LOG_DIR="$ROOT/logs"

log() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }

llm_model() {
    sed -nE 's/^[[:space:]]+model="([^"]+)".*/\1/p' "$ROOT/server/bot.py" | head -1
}

wait_for() { # url-or-port, name, timeout_secs
    local target=$1 name=$2 timeout=${3:-60}
    for _ in $(seq "$timeout"); do
        if [[ $target == http* ]]; then
            curl -sf "$target" >/dev/null && return 0
        else
            nc -z localhost "$target" 2>/dev/null && return 0
        fi
        sleep 1
    done
    log "ERROR: $name did not start within ${timeout}s (see $LOG_DIR)"
    return 1
}

kill_tree() { # pid
    local child
    for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child"; done
    kill -TERM "$1" 2>/dev/null
}

kill_port() { # port
    local pid
    for pid in $(lsof -ti tcp:"$1" -sTCP:LISTEN 2>/dev/null); do kill_tree "$pid"; done
}

stop_all() {
    trap - INT TERM EXIT
    echo
    log "Stopping bot and web client..."
    [[ -n ${BOT_PID:-} ]] && kill_tree "$BOT_PID"
    [[ -n ${CLIENT_PID:-} ]] && kill_tree "$CLIENT_PID"
    kill_port "$BOT_PORT"
    kill_port "$CLIENT_PORT"
    log "Unloading LLM and stopping LM Studio..."
    "$LMS" unload --all >/dev/null 2>&1
    "$LMS" server stop >/dev/null 2>&1
    "$LMS" daemon down >/dev/null 2>&1
    log "Everything stopped."
    exit 0
}

if [[ ${1:-} == stop ]]; then
    stop_all
fi

mkdir -p "$LOG_DIR"
trap stop_all INT TERM EXIT

# LLM_PROVIDER from server/.env: "anthropic" uses Claude, so LM Studio isn't needed
PROVIDER="$(sed -nE 's/^LLM_PROVIDER="?([a-z]+)"?.*/\1/p' "$ROOT/server/.env" 2>/dev/null | tail -1)"
if [[ ${PROVIDER:-local} == anthropic ]]; then
    log "LLM: Claude (LLM_PROVIDER=anthropic), not starting LM Studio."
else
    MODEL="$(llm_model)"
    [[ -z $MODEL ]] && { log "ERROR: no active model=\"...\" line in server/bot.py"; exit 1; }
    log "Starting LM Studio and loading $MODEL..."
    "$LMS" daemon up >/dev/null 2>&1
    if ! "$LMS" ps 2>/dev/null | grep -q "^$MODEL "; then
        "$LMS" unload --all >/dev/null 2>&1
        "$LMS" load "$MODEL" --parallel 1 --ttl "$LLM_TTL_SECS" -y || { log "ERROR: could not load $MODEL"; exit 1; }
    fi
    "$LMS" server start --port "$LLM_PORT" >/dev/null 2>&1
    wait_for "http://127.0.0.1:$LLM_PORT/v1/models" "LM Studio server" 30 || exit 1
fi

log "Starting bot (logs below, also in logs/bot.log)..."
(cd "$ROOT/server" && PYTHONUNBUFFERED=1 exec uv run bot.py --port "$BOT_PORT") \
    > >(tee "$LOG_DIR/bot.log") 2>&1 &
BOT_PID=$!

log "Starting web client (logs in logs/client.log)..."
(cd "$ROOT/client" && exec npm run dev -- --port "$CLIENT_PORT") >"$LOG_DIR/client.log" 2>&1 &
CLIENT_PID=$!

wait_for "$BOT_PORT" "Bot" 90 || exit 1
wait_for "$CLIENT_PORT" "Web client" 60 || exit 1

log "Ready: http://localhost:$CLIENT_PORT  (Ctrl+C stops everything)"
open -a Safari "http://localhost:$CLIENT_PORT" 2>/dev/null || open "http://localhost:$CLIENT_PORT"

# Stop everything if the bot or the client dies on its own
while kill -0 "$BOT_PID" 2>/dev/null && kill -0 "$CLIENT_PID" 2>/dev/null; do
    sleep 1
done
log "The bot or the web client exited."
