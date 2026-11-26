#!/usr/bin/env bash
set -e

# Custom-compiled ROCm gfx1201 + Zen 4 AVX-512 engine for Inkling-2-Mini-Uncensored-AD-Q4_K
LLAMA_SERVER="/home/fabi/models/llamacpps/unsloth-inkling2-gfx1201-src/build/bin/llama-server"
MODEL="/home/fabi/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"
DRAFT_MODEL="/home/fabi/models/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
# 132k context window (129 * 1024 = 132096)
CONTEXT_SIZE="${CONTEXT_SIZE:-132096}"
# Default to q8_0 KV cache; set KV_TYPE=q4_0 for maximum speed.
KV_TYPE="${KV_TYPE:-q8_0}"
# Default to single-token decode (USE_DRAFT=0) to enable #409 decode fusions and keep more MoE blocks on GPU.
# Set USE_DRAFT=1 to enable dspark speculative decoding.
USE_DRAFT="${USE_DRAFT:-0}"

EXTRA_ARGS=()
if [ "$USE_DRAFT" = "1" ]; then
  if [ "$KV_TYPE" = "q8_0" ]; then
    # 6 blocks on CPU (blk.18..23); 18 blocks (including 100% of 3x looped core blk.6..17) on RX 9070 XT
    OT_PATTERN='blk\.(1[8-9]|2[0-3])\.ffn_.*exps\.=CPU'
  else
    # 5 blocks on CPU (blk.19..23); 19 blocks on RX 9070 XT
    OT_PATTERN='blk\.(19|2[0-3])\.ffn_.*exps\.=CPU'
  fi
  EXTRA_ARGS+=(
    --model-draft "$DRAFT_MODEL"
    --spec-type draft-dspark
    --spec-draft-n-max 3
    --spec-draft-n-ubatch 32
    --spec-draft-p-min 0
    --n-gpu-layers-draft 99
    --spec-draft-type-k "$KV_TYPE"
    --spec-draft-type-v "$KV_TYPE"
  )
else
  if [ "$KV_TYPE" = "q8_0" ]; then
    # 5 blocks on CPU (blk.19..23); 19 blocks (blk.0..18, 12343 MiB on ROCm0) on RX 9070 XT
    OT_PATTERN='blk\.(19|2[0-3])\.ffn_.*exps\.=CPU'
  else
    # 3.38 blocks on CPU (blk.21..23 + blk.20.ffn_gate_up_exps; blk.20.ffn_down_exps + blk.0..19 on RX 9070 XT)
    OT_PATTERN='blk\.2[1-3]\.ffn_.*exps\.=CPU,blk\.20\.ffn_gate_up_exps\.=CPU'
  fi
fi

echo "Starting Inkling-2-Mini-Uncensored on http://${HOST}:${PORT} (ctx=${CONTEXT_SIZE}, KV=${KV_TYPE}, draft=${USE_DRAFT}, OT=${OT_PATTERN})..."

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  "${EXTRA_ARGS[@]}" \
  --device ROCm0 \
  --load-mode none \
  --no-op-offload \
  -fit off \
  -ngl 99 \
  -t 8 \
  -tb 16 \
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
