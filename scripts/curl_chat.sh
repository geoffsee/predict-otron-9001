#!/usr/bin/env bash
set -euo pipefail

# Simple curl helper for non-streaming chat completions
# Usage:
#   scripts/curl_chat.sh "Who was the 16th president of the United States?"
#   MODEL_ID=google/gemma-2b-it scripts/curl_chat.sh "Hello!"

SERVER_URL=${SERVER_URL:-http://localhost:8080}
MODEL_ID=${MODEL_ID:-gemma-3-1b-it}
PROMPT=${1:-"What is the capital of France?"}
MAX_TOKENS=${MAX_TOKENS:-128}
# Timeout controls (seconds)
CONNECT_TIMEOUT=${CONNECT_TIMEOUT:-2}
MAX_TIME=${MAX_TIME:-20}

cat <<EOF
[info] POST $SERVER_URL/v1/chat/completions
[info] model=$MODEL_ID, max_tokens=$MAX_TOKENS
[info] prompt=$PROMPT
[info] timeouts: connect=${CONNECT_TIMEOUT}s, max=${MAX_TIME}s
EOF

# Quick preflight to avoid long hangs when server is down
if ! curl -sS -o /dev/null -w "%{http_code}" \
      --connect-timeout "$CONNECT_TIMEOUT" \
      --max-time "$CONNECT_TIMEOUT" \
      "$SERVER_URL/" | grep -qE '^(200|3..)'; then
  echo "[warn] Server not reachable at $SERVER_URL (preflight failed)."
  echo "[hint] Start it with ./run_server.sh or adjust SERVER_URL."
  exit 7
fi

curl -sS -X POST \
  --connect-timeout "$CONNECT_TIMEOUT" \
  --max-time "$MAX_TIME" \
  -H "Content-Type: application/json" \
  "$SERVER_URL/v1/chat/completions" \
  -d @- <<JSON
{
  "model": "${MODEL_ID}",
  "messages": [
    {"role": "user", "content": "${PROMPT}"}
  ],
  "max_tokens": ${MAX_TOKENS},
  "stream": false
}
JSON

echo