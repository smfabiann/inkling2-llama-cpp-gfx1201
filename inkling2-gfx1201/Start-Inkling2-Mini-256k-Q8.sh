#!/usr/bin/env bash
set -e

# Custom-compiled ROCm gfx1201 + Zen 4 AVX-512 engine for Inkling-2-Mini-Uncensored-AD-Q4_K (256k context, q8_0 KV)
LLAMA_SERVER="/home/fabi/models/llamacpps/unsloth-inkling2-gfx1201-src/build/bin/llama-server"
MODEL="/home/fabi/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CONTEXT_SIZE="${CONTEXT_SIZE:-262144}"
# 9 blocks on CPU (blk.0..2 + blk.18..23); keeps 100% of 3x looped core (blk.6..17) on RX 9070 XT (~689 MiB free post-39k)
OT_PATTERN='blk\.([0-2]|1[8-9]|2[0-3])\.ffn_.*exps\.=CPU'

echo "Starting Inkling-2-Mini-Uncensored 256k (q8_0 KV) on http://${HOST}:${PORT} (ctx=${CONTEXT_SIZE}, OT=${OT_PATTERN})..."

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --device ROCm0 \
  --load-mode none \
  --no-op-offload \
  -fit off \
  -ngl 99 \
  -t 8 \
  -tb 16 \
  -ot "$OT_PATTERN" \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -c "$CONTEXT_SIZE" \
  -b 2048 \
  -ub 256 \
  -fa on \
  -np 1 \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --min-p 0.0 \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
