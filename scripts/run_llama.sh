#!/usr/bin/env bash
set -euo pipefail

PROMPT=${1:-"Say hello in one short sentence."}
MODEL=${2:-"meta-llama/Llama-3.2-1B-Instruct"}
MAX_NEW=${3:-64}
FORCE_CPU=${FORCE_CPU:-0}

# Optional: keep HF cache local to repo if not already set
export HF_HOME=${HF_HOME:-"$PWD/.hf-cache"}

BIN="$(dirname "$0")/../target/release/llama_infer"

if [[ ! -x "$BIN" ]]; then
  echo "Building llama-runner (release)..."
  cargo build -p llama-runner --release
fi

echo "Running llama inference..." >&2
ARGS=(
  --model-id "$MODEL"
  --prompt "$PROMPT"
  --max-new-tokens "$MAX_NEW"
)

if [[ "$FORCE_CPU" == "1" || "$FORCE_CPU" == "true" ]]; then
  ARGS+=( --force-cpu )
fi

"$BIN" "${ARGS[@]}"
