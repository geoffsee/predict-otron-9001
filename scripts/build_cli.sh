#!/usr/bin/env sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

TEMP_DIR="$SCRIPT_DIR/temp"

mkdir -p "$TEMP_DIR"

cp "$SCRIPT_DIR/cli.ts" "$TEMP_DIR/cli.ts"
cp "$SCRIPT_DIR/../package.json" "$TEMP_DIR/package.json"

(
cd "$TEMP_DIR"
bun i
bun build ./cli.ts --compile --outfile "$SCRIPT_DIR/cli"
)

rm -rf "$TEMP_DIR"