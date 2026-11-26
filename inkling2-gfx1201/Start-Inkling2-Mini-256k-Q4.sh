#!/usr/bin/env bash
set -e

# Custom-compiled ROCm gfx1201 + Zen 4 AVX-512 engine for Inkling-2-Mini (256k context, q4_0 KV)
LLAMA_SERVER="/home/fabi/models/llamacpps/unsloth-inkling2-gfx1201-src/build/bin/llama-server"
MODEL="/home/fabi/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_M.gguf"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CONTEXT_SIZE="${CONTEXT_SIZE:-262144}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
CACHE_RAM="${CACHE_RAM:-8192}"
DISABLE_HIP_GRAPHS="${DISABLE_HIP_GRAPHS:-1}"
# 4.67 blocks on CPU (blk.20..23 + blk.19.ffn_gate_up_exps; keeps blk.0..18 + blk.19.ffn_down_exps on RX 9070 XT, ~590 MiB free idle / ~420+ MiB post-prompt)
OT_PATTERN='blk\.2[0-3]\.ffn_.*exps\.=CPU,blk\.19\.ffn_gate_up_exps\.=CPU'

if [ "$DISABLE_HIP_GRAPHS" = "1" ]; then
  export GGML_CUDA_DISABLE_GRAPHS=1
fi

echo "Starting Inkling-2-Mini 256k (q4_0 KV) on http://${HOST}:${PORT} (ctx=${CONTEXT_SIZE}, ub=${UBATCH_SIZE}, graphs_disabled=${DISABLE_HIP_GRAPHS}, cache_ram=${CACHE_RAM}, OT=${OT_PATTERN})..."

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
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --cache-ram "$CACHE_RAM" \
  -c "$CONTEXT_SIZE" \
  -b 2048 \
  -ub "$UBATCH_SIZE" \
  -fa on \
  -np 1 \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --min-p 0.0 \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
