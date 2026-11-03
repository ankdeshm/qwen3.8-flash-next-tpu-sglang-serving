# Qwen3.8-Flash-Next Serving on Cloud TPU `v7x` with SGLang-TorchTPU

End-to-end bring-up, runtime patches, quality verification (`GSM8K`), speculative decoding (`MTP` / `NEXTN`), parallelism scaling, multi-precision evaluation (`FP8`, `NVFP4`, `BF16`), and long-context (`64K` & `128K`) serving benchmarks for **`Qwen3.8-Flash-Next`** (`Qwen4ExpForConditionalGeneration`) on a single **Cloud TPU `v7x` host** (`2x2x2` topology, `4` chips = `8` TensorCore ranks, `768 GB` total HBM) running **SGLang-TorchTPU** (`v0.2.0`).

> **Status: Active sweep in progress on TPU `v7x` (`2 / 6` variants completed on `4` chips / `8` TensorCores).** Completed benchmark points are populated below and updated as remaining variants finish. Raw incremental outputs: [`results/SUMMARY.md`](./results/SUMMARY.md) · [`results/summary.json`](./results/summary.json).

## Model & Hardware Overview

| Attribute | Specification |
|---|---|
| **Architecture** | `Qwen4ExpForConditionalGeneration` (`48` layers = `36` Gated-DeltaNet linear attention + `12` full attention) |
| **MoE Routing** | `512` routed experts (`top-10` active, `moe_intermediate_size=640`) + `1` gated shared expert (`intermediate_size=640`) |
| **Residual & PLE** | `4`-stream gated hyper-connections (`10,240`-dim stream width) + layer-2 per-layer `3`-gram embedding (`PLE`, `320M`-row FP8 hash table + dilated short conv) |
| **Speculative Head** | Built-in `1`-layer full-attention Multi-Token Prediction (`MTP` / `NEXTN`) head |
| **Checkpoints Evaluated** | `Qwen/Qwen3.8-Flash-Next-FP8` (`173 GiB`), `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`126 GiB`, `group_size=16`), `Qwen/Qwen3.8-Flash-Next` (`336 GiB` BF16) |
| **TPU Host** | Single TPU `v7x` host (`4` chips / `8` TensorCores, `192 GB` HBM/chip = `96 GB` HBM/TensorCore, `FORCE_SINGLE_HOST=1`) |

> [!IMPORTANT]
> **QSA Sparse Attention Disclosure:** Upstream Qwen Sparse Attention (`QSA`) relies on CUDA-only top-k sparse indexer kernels. All runs set `--json-model-override-args '{"indexer_n_heads": null, "text_config": {"indexer_n_heads": null}}'`, disabling the QSA top-k indexer so the `12` full-attention layers run **exact causal attention up to the `2,048`-token QSA budget** and **full dense causal attention (`Pallas RPA v3`) beyond `2,048` tokens** (`4K`, `64K`, and `128K`).

## TPU Bring-Up: Six Runtime Fixes Required on SGLang-TorchTPU `v0.2.0`

All patches are applied idempotently at container startup by [`scripts/q38_patches.py`](./scripts/q38_patches.py) and [`scripts/q38_qwen4_exp_tpu_ple.py`](./scripts/q38_qwen4_exp_tpu_ple.py), and verified bit-exact against upstream CPU reference execution via [`scripts/test_q38_ple_equivalence.py`](./scripts/test_q38_ple_equivalence.py):

1. **Native-Torch `RMSNormGated` Sigmoid Gate (`layernorm_gated.py`):** Qwen3.8 Gated-DeltaNet output normalization uses `rmsnorm(x) * sigmoid(z)`, whereas the TPU native-PyTorch fallback only implemented `silu`/`swish`. Added explicit `sigmoid` gating.
2. **TorchTPU-Traceable Static-Shape PLE & Exact `int32` Limb N-Gram Hash (`qwen4_exp.py`, `ple_state_pool.py`):**
   - Replaced `msgspec.Struct` `_PLEBatch` with `typing.NamedTuple` so `torch._dynamo` (`fullgraph=True`) can trace layer-2 PLE without `Unsupported object.__new__` graph breaks.
   - Eliminated host-side non-padded token slicing so every prompt length executes on static padded token buckets without triggering runtime recompilations under `fail_on_recompile`.
   - Replaced `int64` multiply/XOR/modulo/cumsum n-gram hashing (which aborts the TPU XLA compiler with `Unsupported conversion from vmreg/vreg to U64`) with an **exact `6 x 12`-bit `int32` limb multiplication + nibble-wise Horner modular reduction**, verified across `200,000` random windows with `0` integer mismatches and verified at weight-load time against checkpoint buffers.
   - Stored `NGramPool` histories as `torch.int32`, replaced slice-view `.copy_()` mutations with root-tensor `int32` `index_put_` writes, and masked fresh requests (`extend_prefix_lens == 0`) to zero short-conv state and `EOS` n-gram history inside the compiled graph.
3. **Whole-Model `DecoderTpuCompilePolicy` & `DP=2` Idle-Extend Guard (`server_args.py`, `qwen4_exp.py`):** Registered `Qwen4ExpForConditionalGeneration` in `LANGUAGE_MODEL_ONLY_ARCHITECTURES`, promoted text-only execution from `Qwen3VLDecoderTpuCompilePolicy` to whole-model `DecoderTpuCompilePolicy` (compiling `LogitsProcessor` + `lm_head` inside XLA), and routed `_prepare_ple_batch` on the active `forward_batch.forward_mode` (with an `out_cache_loc != 0` guard on `flat_extend`) so `DP=2 dp-attention` idle/peer-decode ranks materialized by `prepare_mlp_sync_batch` execute the exact SPMD `EXTEND` graph without mutating slot `0`.
4. **Partial-RoPE & Hyper-Connection Nested `torch.compile` Fix (`rotary_embedding/base.py`, `mrope.py`, `hyperconnection.py`):** Under `direct_compile` (`make_fx` for MTP `TARGET_VERIFY` and `DRAFT`), nested `torch.compile` wrappers on `apply_rotary_emb` (`partial_rotary_factor=0.25`) and `GatedResidual._mix_compute` / `_combine_compute` (`hc_count=4`) triggered `split_compiler` placeholder materialization failures. Disabled inner `torch.compile` on TPU (so the outer compiler fuses them directly) and pre-cast `cos_sin_cache` to `bfloat16` after weight loading.
5. **`MTP` (`NEXTN`) `hc_count=4` Hyper-Connection & `direct_compile` PLE Pool Lifting (`qwen4_exp_mtp.py`, `hybrid_linear_direct_compile.py`, `mtp_draft_worker.py`, `hybrid_linear_attn_backend.py`):** Replaced `int64` `cumsum` in `Qwen4ExpForCausalLMMTP` with `int32` prefix-sum indexing; lifted `ShortConvPool` and `NGramPool` buffers as donated inputs in `HybridLinearKVPoolState`; extended `TpuMtpDraftWorker` for `hc_count=4` (`10,240`-dim unmixed carry + `_mix_streams` before `lm_head`); and replaced host-synchronizing `valid.nonzero()` in `_update_ple_state_after_mtp_verify` with static-shape `int32` gather/blend/scatter.
6. **NVFP4 (`modelopt_fp4`, `group_size=16`) Fused-RS Reduction & JAX `0.11` Migration (`modelopt_quant.py`, `qwen2_moe.py`, `gmm_fused_rs_nodedup.py`):** Set `TpuModelOptNvFp4FusedMoEMethod.use_flashinfer_trtllm_weight_layout = False`, opted `Qwen2MoeSparseMoeBlock` into `allow_backend_reduction = True` for FP4 configs, pre-scaled already-TP-summed Pallas fused-RS routed outputs by `1 / tp_size` (`0.125 = 2^-3` for `TP=8`) before adding the unreduced `shared_expert` partial sum, and migrated removed `pltpu.{semaphore_signal,semaphore_wait,DeviceIdType}` calls in `gmm_fused_rs_nodedup.py` to `pl.*` for JAX `0.11.2`.

## Precision, KV-Cache & MoE Block-Scale Serving Findings

- **`bfloat16` KV Cache Required (`kv_cache_dtype="bfloat16"`):** `Qwen/Qwen3.8-Flash-Next-FP8` does not ship calibrated `k_scale`/`v_scale` factors (`1.0`). Writing unscaled KV activations (`|v| > 448`) into `float8_e4m3fn` overflows to `NaN` and poisons 256-token KV pages via `0.0 * NaN = NaN` inside RPA v3 `jnp.matmul(p, v)`. Because only `12` of `48` layers are full attention (`2` KV heads -> `1` KV head/rank on `TP=8` = `12,288 B/tok/rank`), `bfloat16` KV cache for `1,310,720` tokens consumes only `15.0 GB/rank` (`2.2 GB/rank` for the 36-layer GDN state pool at `160` slots), leaving `>50 GB/rank` free HBM on FP8.
- **`blockwise` vs `channelwise` TPU FP8 Runtime (`--tpu-fp8-runtime=blockwise`):** Collapsing `[128, 128]` block-FP8 weights into `channelwise` runtime scales triggers `NaN` logits on `EXTEND` batches at `bs=64` under `torch.compile`. Preserving native `128x128` blockwise FP8 scales (`--tpu-fp8-runtime=blockwise` + `SGLANG_TPU_FP8_MOE_FP32_SCALES=1`) matches eager logprobs to 4 decimal places (`0` `NaN`s across `GSM8K` and `bs=64` concurrency).
- **`TP=8 x EP=1` MoE Block-Alignment (`moe_intermediate_size=640`):** Splitting `640` intermediate channels across `TP=8` (`EP=1`) yields `80` channels/rank (`80 % 128 != 0`), which `[128, 128]` block-FP8 scales cannot partition across `128`-row scale blocks and `NVFP4` pads `6.4x` (`80 -> 512`). Unquantized `BF16` (`q38_bf16_tp8_ep1`) natively zero-pads `80 -> 128` lanes (`1.6x`) for Pallas `gmm_v2`, enabling clean single-host `TP=8 x EP=1` vs `TP=8 x EP=8` comparison.

## Key Single-Host TPU `v7x` Results (`FP8 TP=8 x EP=8`, `1K`–`128K` Context)

- **Interactive & High-Concurrency Throughput (`1024` in / `1024` out):** `41.5 output tok/s` at `c=1` (`23.97 ms` median `TPOT`), scaling to **`1,301.4 output tok/s`** (`325.4 tok/s/chip`, `2,602.9 total tok/s`) at `c=64` (`38.73 ms` median `TPOT`).
- **`4K` Context (`4096` in / `1024` out):** `41.1 output tok/s` at `c=1` (`24.08 ms` median `TPOT`), reaching **`1,050.8 output tok/s`** (`5,253.9 total tok/s`) at `c=64`.
- **Near-Flat `TPOT` Across `128x` Context Scaling (`64K` & `128K`):** Because `36` of `48` layers are constant-state Gated-DeltaNet linear attention and only `12` layers attend over the KV cache, single-stream median `TPOT` increases by just **`+3.5%` from `1K` to `128K`** (`23.97 ms` at `1K` -> `24.42 ms` at `64K` -> **`24.80 ms` at `128K`**, with `3.87 s` `64K` `TTFT` and `8.21 s` `128K` `TTFT`). A single `4`-chip host sustains **`139.2 output tok/s` at `64K` (`c=8`, `532K` tokens in flight)** and **`44.2 output tok/s` at `128K` (`c=4`, `526K` tokens in flight)** with zero OOMs or recompiles.

## Few-Shot GSM8K Accuracy & MTP Speculative Acceptance Gate

5-shot greedy GSM8K evaluation run directly against each live TPU server:

| Variant | Precision & Parallelism | Harness | Questions | Accuracy | Invalid | GSM8K Output tok/s | Latency (s) | MTP Accept Len |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | raw `/generate` (`stop=["Question"]`) | 200 | **17.0%** | 65.0% | 581.6 | 31.96 | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **98.5%** | 0.0% | 498.4 | 58.48 | - |

> [!NOTE]
> On thinking-enabled checkpoints (`Qwen3.8-Flash-Next`), SGLang's legacy raw-completion `few_shot_gsm8k` script passes `stop=["Question"]` without chat templating (`enable_thinking=false`), truncating Chain-of-Thought traces whenever the model echoes the word `Question` (`65.0%` unparsed `invalid` above; among completed answers, `34 / 70` match ground truth before chat templating). Chat-API evaluations (`sglang.test.run_eval --eval-name gsm8k --api chat` with `enable_thinking=false`) are reported in subsequent rows.

## Serving Throughput, Latency, Parallelism & Long-Context Benchmarks

Fixed-length ShareGPT-sampled requests (`--dataset-name random --random-range-ratio 1.0`, EOS ignored) on one TPU `v7x` host (`4` chips = `8` TensorCores):

| Variant | Layout / Precision | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | MTP Accept Len | vs FP8 Non-Spec |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 1 | 0.041 | **41.5** | 10.4 | 83.0 | 128 / 144 | 23.97 / 24.05 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 4 | 0.156 | **159.2** | 39.8 | 318.5 | 280 / 359 | 24.80 / 25.14 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 16 | 0.531 | **543.7** | 135.9 | 1,087.3 | 923 / 1,005 | 28.63 / 29.29 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 32 | 0.941 | **963.3** | 240.8 | 1,926.6 | 1,363 / 1,927 | 31.91 / 32.79 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 64 | 1.271 | **1,301.4** | 325.4 | 2,602.9 | 2,238 / 7,173 | 38.73 / 40.47 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 1 | 0.040 | **41.1** | 10.3 | 205.5 | 279 / 281 | 24.08 / 24.09 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 4 | 0.152 | **155.2** | 38.8 | 776.0 | 710 / 945 | 25.10 / 25.51 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 16 | 0.484 | **495.7** | 123.9 | 2,478.7 | 2,254 / 3,635 | 30.17 / 31.93 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 32 | 0.799 | **818.6** | 204.7 | 4,093.2 | 4,137 / 7,245 | 34.97 / 38.86 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 64 | 1.026 | **1,050.8** | 262.7 | 5,254.0 | 8,892 / 17,896 | 43.89 / 52.14 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 1 | 0.035 | **35.5** | 8.9 | 2,304.8 | 3,874 / 3,886 | 24.42 / 24.42 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 4 | 0.095 | **97.4** | 24.4 | 6,334.2 | 9,514 / 15,140 | 31.77 / 37.30 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 8 | 0.136 | **139.2** | 34.8 | 9,050.2 | 17,026 / 30,179 | 40.85 / 53.71 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `131072` / `512` | 1 | 0.048 | **24.5** | 6.1 | 6,294.6 | 8,212 / 8,234 | 24.80 / 24.80 | - | - |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `131072` / `512` | 4 | 0.086 | **44.2** | 11.0 | 11,348.2 | 20,226 / 32,212 | 51.16 / 74.73 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 1 | 0.035 | **35.9** | 9.0 | 71.8 | 179 / 181 | 27.70 / 27.71 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 4 | 0.133 | **135.8** | 34.0 | 271.6 | 484 / 595 | 28.96 / 29.33 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 16 | 0.465 | **475.7** | 118.9 | 951.5 | 598 / 1,034 | 33.07 / 33.42 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 32 | 0.850 | **870.5** | 217.6 | 1,741.0 | 1,191 / 1,907 | 35.63 / 36.33 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 64 | 1.166 | **1,194.2** | 298.5 | 2,388.3 | 2,212 / 7,700 | 41.76 / 43.68 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 1 | 0.035 | **35.5** | 8.9 | 177.4 | 486 / 490 | 27.72 / 27.74 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 4 | 0.130 | **133.0** | 33.2 | 665.0 | 906 / 1,295 | 29.48 / 29.86 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 16 | 0.426 | **436.4** | 109.1 | 2,181.8 | 2,217 / 3,982 | 34.74 / 36.33 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 32 | 0.732 | **749.1** | 187.3 | 3,745.3 | 3,973 / 7,466 | 38.71 / 42.21 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 64 | 0.957 | **979.7** | 244.9 | 4,898.5 | 7,896 / 18,342 | 47.93 / 54.99 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `65536` / `1024` | 1 | 0.031 | **31.5** | 7.9 | 2,044.5 | 3,813 / 3,818 | 28.08 / 28.09 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `65536` / `1024` | 4 | 0.087 | **88.9** | 22.2 | 5,779.1 | 9,374 / 14,943 | 35.85 / 41.30 | - | - |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `131072` / `512` | 1 | 0.044 | **22.6** | 5.6 | 5,803.6 | 8,108 / 8,114 | 28.46 / 28.47 | - | - |

## Reproducing the Sweep on GKE (`tpu7x`, `2x2x2` Single Host)

```bash
# 1. Create ConfigMap with runtime patches, equivalence test, and sweep harness
kubectl create configmap q38-sglang-sweep-scripts \
  --from-file=scripts/q38_patches.py \
  --from-file=scripts/q38_qwen4_exp_tpu_ple.py \
  --from-file=scripts/test_q38_ple_equivalence.py \
  --from-file=scripts/run_q38_sglang_sweep.py \
  --from-file=scripts/plan_q38_full.json \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Launch the single-host TPU v7x sweep job
kubectl apply -f k8s/q38-sglang-v7x-sweep-job.yaml
```

## Repository Layout

- [`scripts/q38_patches.py`](./scripts/q38_patches.py) — Idempotent runtime patcher for Gated-DeltaNet sigmoid norm, PLE, NGramPool `int32`, MTP `int32` prefix-sum, and NVFP4 Pallas fused-RS MoE.
- [`scripts/q38_qwen4_exp_tpu_ple.py`](./scripts/q38_qwen4_exp_tpu_ple.py) — Static-shape TorchTPU PLE implementation and exact base-`2^12` `int32` limb n-gram hash.
- [`scripts/test_q38_ple_equivalence.py`](./scripts/test_q38_ple_equivalence.py) — CPU bit-exactness verification suite comparing patched TPU PLE against upstream across `EXTEND`, `DECODE`, `TARGET_VERIFY`, and `200,000` random hash windows.
- [`scripts/run_q38_sglang_sweep.py`](./scripts/run_q38_sglang_sweep.py) — Automated single-host recipe generator, server lifecycle manager, `GSM8K` evaluator, and `sglang.benchmark.serving` runner.
- [`scripts/plan_q38_full.json`](./scripts/plan_q38_full.json) — 6-variant matrix specification (`FP8`, `MTP3`, `EP1`, `DP2-attention`, `NVFP4`, `BF16`, `1K`–`128K` context).
- [`k8s/q38-sglang-v7x-sweep-job.yaml`](./k8s/q38-sglang-v7x-sweep-job.yaml) — GKE TPU `v7x` single-host (`2x2x2`, `4` chips) Job manifest.
- [`results/`](./results/) — Sanitized `SUMMARY.md` and `summary.json` benchmark outputs.
