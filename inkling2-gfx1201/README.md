# Inkling-2-Mini Custom ROCm (`gfx1201`) + Zen 4 Build

Custom `llama.cpp` build and launch configuration tuned specifically for **AMD Radeon RX 9070 XT (16 GB VRAM, `gfx1201`)** + **AMD Ryzen 7 7700 (8c/16t Zen 4, AVX-512)** running `Inkling-2-Mini-UD-Q4_K_M.gguf` at **132k context (`-c 132096`)**.

## Engine Changes Included in This Branch (`inkling2-gfx1201`)

- **Base:** `ggml-org/llama.cpp` tag `b12622` + Unsloth Inkling-2 PR set (`#378`, `#385`, `#386`, `#395`, `#396`, `#397`, `#400`, `#401`, `#402`, `#405`)
- **`unslothai/llama.cpp#406` (`502842e1da`):** AMD RDNA 4 (`gfx1201`) WMMA banded Flash Attention (`q8_0`/`q4_0` KV support, ring-wrapped SWA cache, unaligned full-attention padding), `SSM_CONV` history fusion, and `TANH_GATE_MIX` loop-controller fusion.
- **`unslothai/llama.cpp#408` (`7a5c325338`):** 4-slot `Q8_1` activation cache + fused `RMS_NORM + MUL + Q8_1`.
- **Upstream Inkling-2 correctness fixes (`2d99cf882`):**
  - `ggml-org#32799`: set `ignore_merges = true` for `inkling2` tokenizer pre-type (`llama-tokenize` cleanly encodes Python/code inputs).
  - `ggml-org#32807`: bail out of `sched_copy_experts` on `LLM_ARCH_INKLING2_LOOP` so looped blocks do not fail expert buffer copy.
  - Note: `ggml-org#32833` (PEG parser update) was skipped because this `b12622` base uses the older `build_chat_peg_parser` API.
- **`unslothai/llama.cpp#407` port + RDNA4 `gfx1201` tuning (`afe8ef6e8`):**
  - Ported long-context `FLASH_ATTN_EXT_BANDED` decode support, O(1) `seq_contig_prefix` tracking, and compacted `ROCm0` compute workspace (`n_kq_bias_max = 1<<25`).
  - Eliminated a 242-VGPR register pressure explosion in `flash_attn_ext_banded_vec_gqa` on `gfx1201` using `#pragma unroll 2` and `__launch_bounds__(128, 2)` under `GGML_USE_HIP` (cutting VGPRs to 63–77 with 0 spills).
  - Benchmarked HIP dispatch A/B on `gfx1201`: defaulted `GGML_INKLING2_BANDED_SMALL_LONG=0` and `GGML_INKLING2_BANDED_DECODE=0` under `#if defined(GGML_USE_HIP)` where standard `fattn-vec.cuh` (`nthreads_KQ_q=2`) measured faster on RDNA4.
- **`unslothai/llama.cpp#409` selective HIP port (`725a13da5`):**
  - Ported `GGML_OP_ATTN_RES_MIX` (`attn-res-mix.cu` + CPU fallback) with HIP `__low2bfloat16`/`__high2bfloat16` intrinsics.
  - Ported Inkling-2 `n_tokens == 1` decode kernel fusions (`inkling2-fuse.cu`: `control_gate`, `add_scale`, `loop_down_concat`, `loop_up_gate_r64`, `dyn_shortconv_r32`, `kq_b_gather`, warp reductions via `ggml_cuda_shfl_xor(..., 16)`), while preserving `#406`'s `ssm_conv_hist` (`n_tokens > 1`) and 6-op `tanh_gate_mix` paths.
  - Further cut target `ROCm0` compute buffer from `702.66 MiB` -> `691.54 MiB` -> **`666.04 MiB`**.

## Hardware Tuning Findings (RX 9070 XT 16 GB + Ryzen 7 7700)

- **Exact MoE tensor layout in `Inkling-2-Mini-UD-Q4_K_M.gguf`:**
  - `blk.0..19` use fused NVFP4 gate/up tensors (`ffn_gate_up_exps.{weight,scale}` = `364.5 MiB`, `ffn_down_exps.{weight,scale}` = `182.2 MiB`; `546.8 MiB`/block). Always match trailing dot `.` in `-ot` patterns so `.scale` stays on the same device as `.weight`.
  - `blk.20..23` use `Q8_0`/`Q5_K` (`ffn_gate_up_exps.weight` = `445.5 MiB`, `ffn_down_exps.weight` = `344.2 MiB`; `789.8 MiB`/block).
  - `token_embd.weight` (`469.3 MiB`, `q8_0`) stays on `CPU_Mapped` automatically under `--no-op-offload`.
- **Single-token `#409` decode vs `dspark` speculative decoding on 16 GB VRAM:**
  - Because `#409` decode fusions fire at `n_tokens == 1` (+22.5% decode throughput on `llama-bench`), and `dspark` uses `~1.25 GiB` VRAM (model + draft KV + `525.45 MiB` draft compute buffer + `72 MiB` layer input buffer) while increasing CPU MoE expert traffic on `n_tokens == 4` verify steps, disabling draft by default (`USE_DRAFT=0`) frees enough VRAM to move **2+ additional MoE blocks onto the RX 9070 XT**:
    - `KV_TYPE=q8_0` (`USE_DRAFT=0`): all 20 NVFP4 blocks (`blk.0..19`, `44/48` passes/token) run on `ROCm0`, only `blk.20..23` on CPU (`679 MiB` free VRAM after 39k prompt).
    - `KV_TYPE=q4_0` (`USE_DRAFT=0`): `blk.0..20` + `blk.21.ffn_down_exps` (`45.5/48` passes/token) run on `ROCm0` (`629 MiB` free VRAM after 39k prompt).
  - Set `USE_DRAFT=1` in [`Start-Inkling2-Mini.sh`](./Start-Inkling2-Mini.sh) to re-enable `dspark` with `--spec-draft-n-ubatch 32`.
- **Zen 4 threading:** `-t 8 -tb 16` assigns 1 thread per physical core during CPU MoE decode (avoiding SMT contention that drops decode `-9.9 t/s` at `-t 16`) while using all 16 logical threads for batch prefill (`+15.9 t/s` on 39k prompt processing).
- **CLI `-ot` delimiter note:** `llama-server` / `llama-completion` separate multiple `-ot` patterns with `,` (comma), whereas `llama-bench` uses `;` (semicolon; comma sweeps multiple `-ot` runs). Passing multiple `-ot` flags drops earlier patterns.

## Build Command

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
ROCM_PATH=/usr HIP_PATH=/usr \
cmake --build build --target llama-server llama-bench llama-completion llama-cli llama-tokenize -j 16
```

## Verified Benchmarks (`-c 132096`, RX 9070 XT 16 GB)

### End-to-End `llama-server` (`132096` context, zero host KV spill)

| Configuration | Short Prefill (`pp31`) | Short Decode (`tg128`) | Long 39k Prefill (`pp39051`) | Long 39k Decode (`tg96`) | VRAM Free (Post-39k) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase 1 Baseline (`6cpu`, `q8_0` KV, `dspark n=3`) | 66.95 t/s | 41.60 t/s | 365.36 t/s | 41.60 t/s | 614 MiB |
| **Final Default (`q8_0` KV, `USE_DRAFT=0`, `4cpu_tail`, `-t 8 -tb 16`)** | **76.28 t/s** | **46.47 t/s** (`47.08` @ `-t 8`) | **556.57 t/s** (+52.3%) | **42.12 t/s** (`42.57` max) | **679 MiB** |
| **Fast Override (`KV_TYPE=q4_0`, `USE_DRAFT=0`, `2.5cpu_tail`, `-t 8 -tb 16`)** | **91.43 t/s** | **49.71 t/s** | **648.41 t/s** (+77.5%) | **45.21 t/s** (`45.60` max) | **629 MiB** |

### `llama-bench` Single-Token & Depth Sweep (`-dev ROCm0 -lm none -fa 1 -b 2048 -ub 256 -nopo 1 -r 3`)

| Preset | `pp512` | `pp2048` | `tg128` | `tg32 @ d0` | `tg32 @ d8192` | `tg32 @ d32768` | `tg32 @ d65536` | `tg32 @ d115000` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 1 Baseline (`6cpu`, `q8_0` KV) | — | — | — | 35.75 t/s | 35.05 t/s | 33.41 t/s | 31.26 t/s | — |
| Phase 3 (`6cpu`, `q8_0` KV, `#407`+VGPR) | — | — | — | 35.75 t/s | 35.05 t/s | 33.57 t/s | 31.60 t/s | 28.72 t/s |
| **Final `q8_0` (`4cpu_tail`)** | **651.86** | **646.20** | **48.10** | **45.59** | **45.13** | **43.41** | **39.81** | **36.30** |
| **Final `q4_0` (`2.5cpu_tail`)** | **768.72** | **761.54** | **51.11** | **49.51** | **48.54** | **46.06** | **43.96** | **40.58** |

### Parity & Correctness Checks

- **5k needle-in-a-haystack (`ALPHA-9472-OMEGA`, `-c 8192 -fa on --temp 0`):** exact retrieval (`ALPHA-9472-OMEGA`) on both `q8_0` default and `q4_0` override (`FUSE=0` and `FUSE=1`).
- **Short greedy parity (`509/509` chars vs baseline):** verified across Phase 3 & Phase 4 (`dspark`), plus coherent code/summary outputs across all `USE_DRAFT=0` runs.
- **Tokenizer (`llama-tokenize`):** `"def f(*args, **kwargs):"` exits `0` cleanly (10 tokens).

## Usage

See [`Start-Inkling2-Mini.sh`](./Start-Inkling2-Mini.sh):

```bash
# Default: 132k context, q8_0 KV cache, USE_DRAFT=0 (~679 MiB free VRAM post-39k)
./Start-Inkling2-Mini.sh

# Maximum speed override: 132k context, q4_0 KV cache
KV_TYPE=q4_0 ./Start-Inkling2-Mini.sh

# Optional speculative decoding with dspark
USE_DRAFT=1 ./Start-Inkling2-Mini.sh
```
