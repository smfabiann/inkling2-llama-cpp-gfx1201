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
# Default to q8_0 KV cache for maximum accuracy; set KV_TYPE=q4_0 for maximum speed.
KV_TYPE="${KV_TYPE:-q8_0}"
# Default to single-token decode (USE_DRAFT=0): enables #409 n_tokens==1 decode fusions
# and frees ~1.25 GiB VRAM to keep more MoE expert weights on GPU.
# Set USE_DRAFT=1 to enable dspark speculative decoding.
USE_DRAFT="${USE_DRAFT:-0}"

EXTRA_ARGS=()
if [ "$USE_DRAFT" = "1" ]; then
  if [ "$KV_TYPE" = "q8_0" ]; then
    # 6 blocks on CPU (blk.0..1 + blk.20..23); 100% of 3x looped core blk.6..17 on RX 9070 XT
    OT_PATTERN='blk\.([0-1]|2[0-3])\.ffn_.*exps\.=CPU'
  else
    # 4 blocks on CPU (blk.20..23); all 20 NVFP4 blocks on RX 9070 XT
    OT_PATTERN='blk\.2[0-3]\.ffn_.*exps\.=CPU'
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
    # 4 tail Q8_0/Q5_K blocks on CPU (blk.20..23); all 20 NVFP4 blocks (44/48 passes) on RX 9070 XT
    OT_PATTERN='blk\.2[0-3]\.ffn_.*exps\.=CPU'
  else
    # 2.5 tail blocks on CPU (blk.22..23 + blk.21 gate_up; blk.21 down + blk.0..20 on RX 9070 XT)
    OT_PATTERN='blk\.2[2-3]\.ffn_.*exps\.=CPU,blk\.21\.ffn_gate_up_exps\.=CPU'
  fi
fi

echo "Starting Inkling-2-Mini on http://${HOST}:${PORT} (ctx=${CONTEXT_SIZE}, KV=${KV_TYPE}, draft=${USE_DRAFT}, OT=${OT_PATTERN})..."

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
