# Inkling-2-Mini Custom ROCm (`gfx1201`) + Zen 4 Build

Custom `llama.cpp` build tuned for **AMD Radeon RX 9070 XT (16 GB, `gfx1201`)** + **AMD Ryzen 7 7700 (Zen 4, AVX-512)** running `Inkling-2-Mini-UD-Q4_K_M.gguf` + `dspark-Inkling-2-Mini-Q8_0.gguf` at **132k context (`-c 132096`)**.

## What Is Included in This Branch

- **Base:** `ggml-org/llama.cpp` tag `b12622`
- **Unsloth `pr-set`:** Inkling 2 (`#378`, `#385`, `#386`, `#395`, `#396`, `#397`, `#400`, `#401`, `#402`, `#405`) plus the rest of Unsloth's `b12622` mix pins
- **`unslothai/llama.cpp#406` (`502842e1da`):** AMD RDNA 4 (`gfx1201`) WMMA banded Flash Attention (`q8_0`/`q4_0` KV support, ring-wrapped SWA cache, unaligned full-attention padding), `SSM_CONV` history fusion, and `TANH_GATE_MIX` loop-controller fusion
- **`unslothai/llama.cpp#408` (`7a5c325338`):** 4-slot `Q8_1` activation cache + fused `RMS_NORM + MUL + Q8_1`

## Build Command Used

```bash
PATH=/usr/lib64/rocm/llvm/bin:/usr/bin:/bin \
ROCM_PATH=/usr HIP_PATH=/usr \
cmake -B build \
  -DCMAKE_C_COMPILER=/usr/lib64/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/usr \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS="gfx1201" \
  -DAMDGPU_TARGETS="gfx1201" \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DGGML_HIP=ON \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=OFF \
  -DGGML_RPC=OFF

PATH=/usr/lib64/rocm/llvm/bin:/usr/bin:/bin \
cmake --build build --target llama-server llama-bench llama-completion llama-cli -j 16
```

## Launch Script & Key Flags

See [`Start-Inkling2-Mini.sh`](./Start-Inkling2-Mini.sh):

- **`--device ROCm0`**: Pins execution to the RX 9070 XT (ignores the Raphael iGPU `ROCm1`).
- **`--no-op-offload`**: Prevents `sched_reserve` from allocating ~6.56 GiB of VRAM to copy CPU-offloaded MoE tensors during prefill at `-ub >= 32`.
- **`-ot "blk\.([0-1]|2[0-3])\.ffn_.*exps\.=CPU"`** (default `q8_0` KV) or **`-ot "blk\.2[0-3]\.ffn_.*exps\.=CPU"`** (`KV_TYPE=q4_0`): Keeps 100% of the 3× looped core (`blk.6..17`) on the GPU (`42/48` or `44/48` layer passes per token on `ROCm0`).
- **`-c 132096 -b 2048 -ub 256 -fa on -np 1`**: 132k context window with WMMA banded Flash Attention.
