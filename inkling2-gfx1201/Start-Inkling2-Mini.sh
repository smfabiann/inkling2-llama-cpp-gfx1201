#!/usr/bin/env bash
set -e

# Custom-compiled ROCm gfx1201 + Zen 4 AVX-512 engine for Inkling-2-Mini
LLAMA_SERVER="/home/fabi/models/llamacpps/unsloth-inkling2-gfx1201-src/build/bin/llama-server"
MODEL="/home/fabi/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_M.gguf"
DRAFT_MODEL="/home/fabi/models/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
# 132k context window (129 * 1024 = 132096)
CONTEXT_SIZE="${CONTEXT_SIZE:-132096}"
# Default to q8_0 KV cache (6 blocks on CPU -> ~41.9 tok/s).
# Set KV_TYPE=q4_0 to use 4-bit KV cache (4 blocks on CPU -> ~46.4 tok/s).
KV_TYPE="${KV_TYPE:-q8_0}"

if [ "$KV_TYPE" = "q8_0" ]; then
  # 6 blocks on CPU (blk.0..1 + blk.20..23); 18 blocks (including 100% of 3x looped core blk.6..17) on RX 9070 XT
  OT_PATTERN="blk\.([0-1]|2[0-3])\.ffn_.*exps\.=CPU"
else
  # 4 blocks on CPU (blk.20..23 only); all 20 nvfp4 blocks (blk.0..19, 44/48 layer passes) on RX 9070 XT
  OT_PATTERN="blk\.2[0-3]\.ffn_.*exps\.=CPU"
fi

echo "Starting Inkling-2-Mini on http://${HOST}:${PORT} with ${CONTEXT_SIZE} context (KV=${KV_TYPE}, OT=${OT_PATTERN})..."

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --model-draft "$DRAFT_MODEL" \
  --spec-type draft-dspark \
  --spec-draft-n-max 3 \
  --n-gpu-layers-draft 99 \
  --device ROCm0 \
  --load-mode none \
  --no-op-offload \
  -fit off \
  -ngl 99 \
  -ot "$OT_PATTERN" \
  --cache-type-k "$KV_TYPE" \
  --cache-type-v "$KV_TYPE" \
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
