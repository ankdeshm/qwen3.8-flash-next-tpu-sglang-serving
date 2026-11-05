# `Qwen3.8-Flash-Next` Serving on Cloud TPU `v7x` with SGLang-TorchTPU

Single-host serving recipes, precision comparisons (`FP8` blockwise vs `BF16`, plus `NVFP4` and `MTP` evaluations), parallelism ablations (`TP=8 × EP=8`, `TP=8 × EP=1`, `DP=2 dp-attention × EP=8`), 5-shot `GSM8K` quality evaluations, and long-context (`64K` & `128K`) benchmarks for **`Qwen3.8-Flash-Next`** (`Qwen4ExpForConditionalGeneration`) on one **Cloud TPU `v7x` host** (`2x2x2` topology, `4` physical chips = `8` TensorCore ranks, `768 GB` HBM) using **SGLang-TorchTPU `v0.2.0`**.

Full per-cell tables: [`results/SUMMARY.md`](./results/SUMMARY.md) · machine-readable telemetry: [`results/summary.json`](./results/summary.json).

**Conventions:** `c` = concurrent requests; `TTFT` = time to first token and `TPOT` = time per output token (both p50 unless stated); `Chips` = physical TPU `v7x` chips (`4` chips = `8` TensorCores). Every request decodes the exact stated output token count (`EOS` ignored). Only output-verified configurations are reported in throughput tables.

| Attribute | Specification |
|---|---|
| **Architecture** | `Qwen4ExpForConditionalGeneration` (`48` layers = `36` Gated-DeltaNet linear attention + `12` full causal attention) |
| **MoE Routing** | `512` routed experts (`top-10` active, `moe_intermediate_size=640`) + `1` gated shared expert (`intermediate_size=640`) |
| **Residual & PLE** | `4`-stream gated hyper-connections (`10,240`-dim carry) + layer-2 per-layer `3`-gram embedding (`PLE`, `320M`-row FP8 hash table + dilated short conv) |
| **Speculative Head** | Built-in `1`-layer full-attention Multi-Token Prediction (`MTP` / `NEXTN`) head |
| **Checkpoints Tested** | [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) (`173 GiB`), [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`336 GiB` BF16), [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (`126 GiB`, `modelopt_fp4`) |
| **TPU Host** | 1× TPU `v7x` host (`4` chips / `8` TensorCores, `192 GB` HBM/chip = `96 GB` HBM/TensorCore, `FORCE_SINGLE_HOST=1`) |

---

## 1. Headline Results on Single-Host TPU `v7x` (`4` Chips / `8` TensorCores)

### 1.1 Interactive & Batch Serving (`1024` in / `1024` out, EOS ignored)

| Variant | Precision & Topology | 5-Shot Chat GSM8K | `c=1`: Out tok/s (`TPOT p50`) | `c=4` | `c=16` | `c=32` | `c=64` Peak (`tok/s/chip`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`q38_fp8_tp8_ep8`** | `FP8` W8A8 (`128×128` blockwise), `TP=8 × EP=8`, `BF16` KV | **99.0%**¹ | **41.5** (`23.97 ms`) | **159.2** (`24.80 ms`) | **543.7** (`28.63 ms`) | **963.3** (`31.91 ms`) | **1,301.4** (`325.4`/chip, `38.73 ms`) |
| **`q38_fp8_dp2attn_ep8`** | `FP8` W8A8 blockwise, `TP=8 × DP=2` (`dp-attention` + `DP` LM head) `× EP=8`, `BF16` KV | **99.0%** | 30.8 (`32.25 ms`) | 121.2 (`32.69 ms`) | 435.1 (`35.98 ms`) | 758.0 (`40.67 ms`) | **1,273.4** (`318.4`/chip, `46.97 ms`) |
| **`q38_bf16_tp8_ep8`** | `BF16` unquantized weights + `BF16` KV, `TP=8 × EP=8` | **98.5%** | 35.9 (`27.70 ms`) | 135.8 (`28.96 ms`) | 475.7 (`33.07 ms`) | 870.5 (`35.63 ms`) | **1,194.2** (`298.5`/chip, `41.76 ms`) |
| **`q38_bf16_tp8_ep1`** | `BF16` unquantized weights + `BF16` KV, `TP=8 × EP=1` (`80→128` `gmm_v2` pad) | **98.5%** | 34.9 (`28.45 ms`) | 131.1 (`29.79 ms`) | 417.4 (`37.25 ms`) | 708.8 (`43.23 ms`) | **946.7** (`236.7`/chip, `54.68 ms`) |
| *`q38_fp8_tp8_ep8_mtp3`* | *`FP8` W8A8 blockwise, `TP=8 × EP=8`, `BF16` KV + `MTP-3` (`NEXTN`)* | *49.0%*² | — | — | — | — | — |
| *`q38_nvfp4_tp8_ep8`* | *`NVFP4` (`modelopt_fp4`, `group_size=16`), `TP=8 × EP=8`, `BF16` KV* | *0.0% (`NaN`)*³ | — | — | — | — | — |

¹ Blockwise `FP8` Chat-API `GSM8K` (`enable_thinking=false`) is **`99.0%`** (`198/200`, measured on `q38_fp8_dp2attn_ep8`); the initial `q38_fp8_tp8_ep8` run used SGLang's legacy raw `/generate` script (`stop=["Question"]` without `enable_thinking=false`), which truncated `65.0%` of Chain-of-Thought responses (Section 2).
² MTP-3 speculative decoding is not yet output-equivalent on this stack (5-shot GSM8K `49.0%` vs `99.0%` without MTP) because PLE `ShortConvPool`/`NGramPool` state is not rolled back on rejected draft tokens; throughput is not reported until this is resolved ([Supported Configurations & Known Limitations](#known-limitations)).
³ `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`modelopt_fp4`) produces `NaN` logprobs (`0.0%` GSM8K) in `sglang-torchtpu v0.2.0` due to unmapped `512`-expert `Qwen4Exp` `modelopt_fp4` scales; throughput is not reported ([Supported Configurations & Known Limitations](#known-limitations)).

### 1.2 Long-Context Scaling (`4K` → `128K` Input Tokens, Radix Cache Off)

| Variant | `4096` / `1024` (`c=1` / Peak) | `65536` (`64K`) / `1024` (`c=1` / Peak) | `131072` (`128K`) / `512` (`c=1` / Peak) |
|---|---|---|---|
| **`q38_fp8_tp8_ep8`** | **`41.1`** (`24.08 ms`) / **`1,050.8`** (`c=64`, `5,254` tot tok/s) | **`35.5`** (`24.42 ms`, TTFT `3.87 s`) / **`139.2`** (`c=8`, `9,050` tot tok/s) | **`24.5`** (`24.80 ms`, TTFT `8.21 s`) / **`44.2`** (`c=4`, **`11,348` tot tok/s**) |
| **`q38_fp8_dp2attn_ep8`** | `30.3` (`32.28 ms`) / **`961.4`** (`c=64`, `4,807` tot tok/s) | `23.0` (`32.63 ms`, TTFT `11.16 s`) / **`70.3`** (`c=4`, `4,570` tot tok/s) | — |
| **`q38_bf16_tp8_ep8`** | `35.5` (`27.72 ms`) / **`979.7`** (`c=64`, `4,899` tot tok/s) | `31.5` (`28.08 ms`, TTFT `3.81 s`) / **`88.9`** (`c=4`, `5,779` tot tok/s) | **`22.6`** (`28.46 ms`, TTFT `8.11 s`, `5,804` tot tok/s) / — |
| **`q38_bf16_tp8_ep1`** | `34.3` (`28.48 ms`) / **`758.1`** (`c=64`, `3,791` tot tok/s) | `29.2` (`28.84 ms`, TTFT `5.52 s`, `1,900` tot tok/s) / — | **`19.3`** (`29.22 ms`, TTFT `11.51 s`, `4,973` tot tok/s) / — |

### Key Findings

1. **Blockwise `FP8` (`TP=8 × EP=8`) Matches `BF16` Quality (`99.0%` vs `98.5%` GSM8K) with Higher Throughput and `2×` Smaller Weight Footprint:**
   - Preserving native `128 × 128` blockwise FP8 scales (`--tpu-fp8-runtime=blockwise` + `SGLANG_TPU_FP8_MOE_FP32_SCALES=1`, `173 GiB` checkpoint) achieves **`99.0%` 5-shot Chat `GSM8K`** (`0.0%` invalid) versus **`98.5%`** on unquantized `BF16` (`336 GiB`), while increasing `c=1` output throughput by **`+15.6%`** (`41.5` vs `35.9` tok/s, `23.97 ms` vs `27.70 ms` TPOT) and `c=64` batch throughput by **`+9.0%`** (`1,301.4` vs `1,194.2` out tok/s at `1K/1K`).
2. **Near-Flat Decode `TPOT` Across `1K` → `128K` Context (`+3.5%` Over `128×` Prompt Scaling):**
   - Because `36` of `48` decoder layers use constant-state Gated-DeltaNet linear attention and only `12` layers attend over the KV cache (`1` KV head/rank at `TP=8` = `12,288 B/tok/rank` in `BF16`), single-stream `TPOT` on `q38_fp8_tp8_ep8` moves by only **`+3.5%` from `1K` to `128K`** (`23.97 ms` at `1K` → `24.08 ms` at `4K` → `24.42 ms` at `64K` → **`24.80 ms` at `128K`**). One `4`-chip `v7x` host sustains **`139.2` out tok/s (`9,050` total tok/s) at `64K` (`c=8`)** and **`44.2` out tok/s (`11,348` total tok/s) at `128K` (`c=4`)** with dense causal attention (`indexer_n_heads: null`).
3. **Expert Parallelism (`EP=8`) Outperforms Expert Tensor Parallelism (`EP=1`) by `+26.1%`–`+29.2%` on `512` Fine-Grained Experts:**
   - Both `q38_bf16_tp8_ep8` and `q38_bf16_tp8_ep1` reach identical **`98.5%` `GSM8K` accuracy** (`197/200`). Because `Qwen3.8-Flash-Next` uses `moe_intermediate_size=640`, sharding each expert across `TP=8` (`EP=1`) produces `80` channels/rank, which Pallas `gmm_v2` pads `1.6×` to `128` lanes across all `512` experts; `EP=8` keeps `64` whole experts per rank at unpadded `640` width and delivers **`+26.1%` higher `1K/1K c=64` throughput** (`1,194.2` vs `946.7` tok/s) and **`+29.2%` higher `4K/1K c=64` throughput** (`979.7` vs `758.1` tok/s).
4. **`DP=2` Attention Data Parallelism (`q38_fp8_dp2attn_ep8`) Converges Within `2.2%` of Pure `TP=8` at `c=64`:**
   - Partitioning attention into `2 × attn-TP=4` with a `DP` LM head (`--dp-size 2 --enable-dp-attention --enable-dp-lm-head`) introduces extra collective transitions across the `48` hybrid layers at low concurrency (`30.8` vs `41.5` tok/s at `c=1`), and reaches **`1,273.4` out tok/s at `1K/1K c=64`** (`97.8%` of `TP=8`).

---

## 2. Few-Shot `GSM8K` Quality Gate

5-shot greedy `GSM8K` (`200` questions, `max_new_tokens=1024`) run directly against each live TPU server:

| Variant | Precision & Parallelism | Harness | Questions | Accuracy | Invalid | GSM8K Out tok/s | Latency (s) |
|---|---|---|---:|---:|---:|---:|---:|
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128×128 blockwise), TP=8 × EP=8, BF16 KV | raw `/generate` (`stop=["Question"]`) | 200 | **17.0%**¹ | 65.0% | 581.6 | 31.96 |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 × DP=2 (`dp-attention`) × EP=8, BF16 KV | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **99.0%** | 0.0% | 288.1 | 100.04 |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 × EP=8 | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **98.5%** | 0.0% | 498.4 | 58.48 |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 × EP=1 (`80→128` pad) | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **98.5%** | 0.0% | 408.1 | 70.99 |
| `q38_fp8_tp8_ep8_mtp3` | FP8 W8A8 (blockwise), TP=8 × EP=8, BF16 KV + MTP-3 (`NEXTN`) | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | 49.0% (not output-equivalent)² | 0.0% | — | — |
| `q38_nvfp4_tp8_ep8` | NVFP4 (`modelopt_fp4`, `group_size=16`), TP=8 × EP=8, BF16 KV | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | 0.0% (`NaN`)³ | 0.0% | — | — |

¹ On thinking-capable `Qwen3.8-Flash-Next` checkpoints, SGLang's raw-completion `few_shot_gsm8k` client sends `stop=["Question"]` without chat templating (`enable_thinking=false`), truncating Chain-of-Thought output whenever the model emits `Question` (`65.0%` unparsed `invalid`). Evaluating via `/v1/chat/completions` with `enable_thinking=false` resolves this (`99.0%` on `FP8`, `98.5%` on `BF16`, `0.0%` invalid).
² MTP-3 speculative decoding is not yet output-equivalent on this stack (5-shot GSM8K `49.0%` vs `99.0%` without MTP) because PLE `ShortConvPool`/`NGramPool` state is not rolled back on rejected draft tokens; throughput is not reported until this is resolved.
³ `NVFP4` outputs are `NaN` logprobs; throughput is not reported.

---

## 3. Serving Throughput, Latency & Long-Context Benchmarks

Fixed-length ShareGPT-sampled requests (`--dataset-name random --random-range-ratio 1.0`, greedy, `EOS` ignored, radix cache disabled) on one TPU `v7x` host (`4` chips = `8` TensorCores) across the four output-verified configurations:

| Variant | Layout / Precision | In / Out | Conc. | Out tok/s | Out tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | vs `FP8 TP8 EP8` |
|---|---|---|---:|---:|---:|---:|---|---|---:|
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `1024` / `1024` | 1 | **41.5** | 10.4 | 83.0 | 128 / 144 | 23.97 / 24.05 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `1024` / `1024` | 4 | **159.2** | 39.8 | 318.5 | 280 / 359 | 24.80 / 25.14 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `1024` / `1024` | 16 | **543.7** | 135.9 | 1,087.3 | 923 / 1,005 | 28.63 / 29.29 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `1024` / `1024` | 32 | **963.3** | 240.8 | 1,926.6 | 1,363 / 1,927 | 31.91 / 32.79 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `1024` / `1024` | 64 | **1,301.4** | 325.4 | 2,602.9 | 2,238 / 7,173 | 38.73 / 40.47 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `4096` / `1024` | 1 | **41.1** | 10.3 | 205.5 | 279 / 281 | 24.08 / 24.09 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `4096` / `1024` | 4 | **155.2** | 38.8 | 776.0 | 710 / 945 | 25.10 / 25.51 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `4096` / `1024` | 16 | **495.7** | 123.9 | 2,478.7 | 2,254 / 3,635 | 30.17 / 31.93 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `4096` / `1024` | 32 | **818.6** | 204.7 | 4,093.2 | 4,137 / 7,245 | 34.97 / 38.86 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `4096` / `1024` | 64 | **1,050.8** | 262.7 | 5,254.0 | 8,892 / 17,896 | 43.89 / 52.14 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `65536` / `1024` | 1 | **35.5** | 8.9 | 2,304.8 | 3,874 / 3,886 | 24.42 / 24.42 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `65536` / `1024` | 4 | **97.4** | 24.4 | 6,334.2 | 9,514 / 15,140 | 31.77 / 37.30 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `65536` / `1024` | 8 | **139.2** | 34.8 | 9,050.2 | 17,026 / 30,179 | 40.85 / 53.71 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `131072` / `512` | 1 | **24.5** | 6.1 | 6,294.6 | 8,212 / 8,234 | 24.80 / 24.80 | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 blockwise, TP=8 × EP=8, BF16 KV | `131072` / `512` | 4 | **44.2** | 11.0 | 11,348.2 | 20,226 / 32,212 | 51.16 / 74.73 | 1.00x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `1024` / `1024` | 1 | **30.8** | 7.7 | 61.6 | 259 / 310 | 32.25 / 32.26 | 0.74x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `1024` / `1024` | 4 | **121.2** | 30.3 | 242.3 | 299 / 434 | 32.69 / 32.86 | 0.76x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `1024` / `1024` | 16 | **435.1** | 108.8 | 870.3 | 897 / 1,563 | 35.98 / 36.36 | 0.80x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `1024` / `1024` | 32 | **758.0** | 189.5 | 1,516.1 | 1,561 / 3,165 | 40.67 / 41.74 | 0.79x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `1024` / `1024` | 64 | **1,273.4** | 318.4 | 2,546.8 | 3,357 / 5,700 | 46.97 / 49.60 | 0.98x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `4096` / `1024` | 1 | **30.3** | 7.6 | 151.6 | 715 / 769 | 32.28 / 32.29 | 0.74x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `4096` / `1024` | 16 | **390.6** | 97.7 | 1,953.1 | 3,358 / 5,765 | 38.04 / 40.45 | 0.79x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `4096` / `1024` | 64 | **961.4** | 240.4 | 4,807.0 | 11,990 / 21,544 | 55.15 / 65.85 | 0.91x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `65536` / `1024` | 1 | **23.0** | 5.7 | 1,493.6 | 11,157 / 11,171 | 32.63 / 32.63 | 0.65x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 blockwise, TP=8 × DP=2 dp-attn × EP=8, BF16 KV | `65536` / `1024` | 4 | **70.3** | 17.6 | 4,570.0 | 17,679 / 24,412 | 39.89 / 46.31 | 0.72x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `1024` / `1024` | 1 | **35.9** | 9.0 | 71.8 | 179 / 181 | 27.70 / 27.71 | 0.86x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `1024` / `1024` | 4 | **135.8** | 34.0 | 271.6 | 484 / 595 | 28.96 / 29.33 | 0.85x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `1024` / `1024` | 16 | **475.7** | 118.9 | 951.5 | 598 / 1,034 | 33.07 / 33.42 | 0.88x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `1024` / `1024` | 32 | **870.5** | 217.6 | 1,741.0 | 1,191 / 1,907 | 35.63 / 36.33 | 0.90x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `1024` / `1024` | 64 | **1,194.2** | 298.5 | 2,388.3 | 2,212 / 7,700 | 41.76 / 43.68 | 0.92x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `4096` / `1024` | 1 | **35.5** | 8.9 | 177.4 | 486 / 490 | 27.72 / 27.74 | 0.86x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `4096` / `1024` | 4 | **133.0** | 33.2 | 665.0 | 906 / 1,295 | 29.48 / 29.86 | 0.86x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `4096` / `1024` | 16 | **436.4** | 109.1 | 2,181.8 | 2,217 / 3,982 | 34.74 / 36.33 | 0.88x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `4096` / `1024` | 32 | **749.1** | 187.3 | 3,745.3 | 3,973 / 7,466 | 38.71 / 42.21 | 0.92x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `4096` / `1024` | 64 | **979.7** | 244.9 | 4,898.5 | 7,896 / 18,342 | 47.93 / 54.99 | 0.93x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `65536` / `1024` | 1 | **31.5** | 7.9 | 2,044.5 | 3,813 / 3,818 | 28.08 / 28.09 | 0.89x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `65536` / `1024` | 4 | **88.9** | 22.2 | 5,779.1 | 9,374 / 14,943 | 35.85 / 41.30 | 0.91x |
| `q38_bf16_tp8_ep8` | BF16 weights + BF16 KV, TP=8 × EP=8 | `131072` / `512` | 1 | **22.6** | 5.6 | 5,803.6 | 8,108 / 8,114 | 28.46 / 28.47 | 0.92x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `1024` / `1024` | 1 | **34.9** | 8.7 | 69.8 | 237 / 240 | 28.45 / 28.45 | 0.84x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `1024` / `1024` | 4 | **131.1** | 32.8 | 262.3 | 692 / 862 | 29.79 / 30.35 | 0.82x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `1024` / `1024` | 16 | **417.4** | 104.4 | 834.9 | 1,110 / 1,520 | 37.25 / 38.11 | 0.77x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `1024` / `1024` | 32 | **708.8** | 177.2 | 1,417.6 | 1,994 / 2,821 | 43.23 / 44.53 | 0.74x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `1024` / `1024` | 64 | **946.7** | 236.7 | 1,893.5 | 3,290 / 10,332 | 54.68 / 57.22 | 0.73x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `4096` / `1024` | 1 | **34.3** | 8.6 | 171.6 | 688 / 690 | 28.48 / 28.48 | 0.84x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `4096` / `1024` | 16 | **373.8** | 93.4 | 1,868.9 | 3,279 / 5,897 | 40.02 / 42.28 | 0.75x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `4096` / `1024` | 64 | **758.1** | 189.5 | 3,790.7 | 11,751 / 26,509 | 63.05 / 73.86 | 0.72x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `65536` / `1024` | 1 | **29.2** | 7.3 | 1,899.5 | 5,519 / 5,539 | 28.84 / 28.84 | 0.82x |
| `q38_bf16_tp8_ep1` | BF16 weights + BF16 KV, TP=8 × EP=1 (80→128 pad) | `131072` / `512` | 1 | **19.3** | 4.8 | 4,972.5 | 11,512 / 11,575 | 29.22 / 29.22 | 0.79x |

---

<a id="methodology-notes"></a>
## 4. Methodology Notes

- **Workload & client:** `sglang.benchmark.serving` (`--backend sglang --dataset-name random --random-range-ratio 1.0`), sampling fixed-length ShareGPT token slices and decoding exact `output_len` tokens (`--ignore-eos`, greedy `temperature=0`). Prompt counts scale with concurrency (`4, 12, 48, 96, 192` prompts at `c=1, 4, 16, 32, 64` for `1K`/`4K`; `2, 8, 16` prompts at `c=1, 4, 8` for `64K`/`128K`). Treat `2`-prompt `c=1` long-context cells (`64K` and `128K`) as indicative.
- **Radix cache disabled (`--disable-radix-cache`):** All throughput and long-context cells run with prefix caching disabled so every request executes full prefill compute across all `48` hybrid layers.
- **Dense fallback for Qwen Sparse Attention (`QSA`):** Upstream QSA uses a non-TPU sparse top-k attention indexer kernel (`2,048`-token sparse window). All runs pass `--json-model-override-args '{"indexer_n_heads": null, "text_config": {"indexer_n_heads": null}}'`, so the `12` full-attention layers run causal attention up to `2,048` tokens and **full dense causal attention (`Pallas RPA v3`)** at `4K`, `64K`, and `128K`.
- **`bfloat16` KV cache & `blockwise` FP8 runtime:** `Qwen/Qwen3.8-Flash-Next-FP8` ships uncalibrated KV scales (`k_scale = v_scale = 1.0`), so all valid configurations use `--kv-cache-dtype bfloat16`. Because only `12` of `48` layers are full attention (`2` KV heads = `1` KV head/rank at `TP=8`, `12,288 B/tok/rank`), `1,310,720` BF16 KV tokens occupy `15.0 GB/rank` alongside `2.2 GB/rank` for the `36`-layer Gated-DeltaNet state pool (`160` slots). FP8 weights run with `--tpu-fp8-runtime=blockwise` and `SGLANG_TPU_FP8_MOE_FP32_SCALES=1` to preserve checkpoint `128 × 128` block scales.
- **`GSM8K` harness:** `q38_fp8_tp8_ep8` was evaluated with SGLang's raw `/generate` script (`stop=["Question"]`), whereas all subsequent variants used `/v1/chat/completions` with `chat_template_kwargs={"enable_thinking": false}` (`200` questions, `5`-shot, greedy).

<a id="modifications"></a>
## 5. Modifications

Applied idempotently to the container's `sglang` / `sglang_tpu` packages before server launch by [`scripts/q38_patches.py`](./scripts/q38_patches.py) and [`scripts/q38_qwen4_exp_tpu_ple.py`](./scripts/q38_qwen4_exp_tpu_ple.py) (verified against the CPU reference across `EXTEND`, `DECODE`, `TARGET_VERIFY`, and `200,000` random hash windows by [`scripts/test_q38_ple_equivalence.py`](./scripts/test_q38_ple_equivalence.py)):

- **`layernorm_gated.py` (`RMSNormGated`):** Adds `sigmoid` gating (`rmsnorm(x) * sigmoid(z)`) to the native-PyTorch TPU path used by `Qwen3.8` Gated-DeltaNet output normalization.
- **`qwen4_exp.py` & `server_args.py` (Static-Shape TPU PLE & Whole-Model `DecoderTpuCompilePolicy`):** Registers `Qwen4ExpForConditionalGeneration` in `LANGUAGE_MODEL_ONLY_ARCHITECTURES` so `--language-model-only` compiles `LogitsProcessor` + `lm_head` in-graph; converts `_PLEBatch` from `msgspec.Struct` to `NamedTuple`; keeps static padded token shapes with flat token-major `EXTEND` layout and an `out_cache_loc != 0` guard for `DP=2 dp-attention` peer-decode ranks; zero-masks fresh-request short-conv state and sets `EOS` n-gram history in-graph; evaluates the `3`-gram hash via exact `6 × 12`-bit `int32` limb multiplication + Horner reduction; and updates `ShortConvPool` via root `index_put_`.
- **`ple_state_pool.py` (`NGramPool`):** Stores n-gram token histories as `torch.int32` and performs root `index_put_` updates in `set_intermediate_context`.
- **`qwen4_exp_mtp.py`, `rotary_embedding/{base,mrope}.py`, `hyperconnection.py`, & `sglang_tpu` MTP / `direct_compile`:** Uses `int32` prefix sums in `Qwen4ExpForCausalLMMTP`; preserves full `TARGET_VERIFY` rows in `_set_hc_logits_hidden_states`; disables nested `torch.compile` decorators on partial-RoPE `apply_rotary_emb` and `GatedResidual._mix_compute` / `_combine_compute` so outer `make_fx` (`direct_compile`) traces cleanly and pre-casts `cos_sin_cache` to `bfloat16`; lifts `req_index_to_mamba_index_mapping`, `ShortConvPool`, and `NGramPool` buffers in `HybridLinearKVPoolState`; extends `TpuMtpDraftWorker` for `hc_count=4` (`10,240`-dim stream carry + `_mix_streams` before `lm_head`); and commits verified PLE state with static-shape `int32` gather/scatter in `HybridLinearAttnBackend`.
- **`modelopt_quant.py`, `qwen2_moe.py`, & `fused_rs/{fused_moe,gmm_fused_rs_nodedup}.py`:** Sets `TpuModelOptNvFp4FusedMoEMethod.use_flashinfer_trtllm_weight_layout = False`, enables `allow_backend_reduction = True` on `Qwen2MoeSparseMoeBlock` for FP4 configs, scales TP-summed Pallas fused-RS routed outputs by `1 / tp_size` before adding the unreduced shared expert, widens non-SP fused-RS routing broadcast from `bfloat16` to `float32` for `512`-expert indices, and updates Pallas semaphore calls to `pl.*` for JAX `0.11+`.

<a id="known-limitations"></a>
## 6. Supported Configurations & Known Limitations

- **Recommended production configuration on 1-host TPU `v7x`:** `q38_fp8_tp8_ep8` (`Qwen/Qwen3.8-Flash-Next-FP8`, `--tp-size 8 --ep-size 8 --quantization fp8 --tpu-fp8-runtime blockwise --kv-cache-dtype bfloat16 --language-model-only`), which delivers full accuracy (`99.0%` Chat `GSM8K` on blockwise `FP8`) and highest output-verified throughput from `1K` to `128K` context.
- **Text-only serving (`--language-model-only`):** All benchmarks run text-only; multimodal vision encoding was not evaluated.
- **QSA sparse indexer disabled (`indexer_n_heads: null`):** Context lengths above `2,048` tokens execute dense causal attention (`Pallas RPA v3`) rather than QSA top-k sparse attention.
- **`BF16` KV cache requirement on uncalibrated checkpoints:** Using `--kv-cache-dtype fp8_e4m3` without calibrated `k_scale`/`v_scale` overflows `float8_e4m3fn` (`±448`) when `|v| > 448` and produces `NaN` attention outputs; use `--kv-cache-dtype bfloat16`.
- **`FP8` with `EP=1` (`TP=8 × EP=1`) not supported:** With `moe_intermediate_size=640`, `TP=8 × EP=1` yields `80` intermediate channels/rank (`80 % 128 != 0`), which cannot partition `128 × 128` block-FP8 scales evenly; `EP=1` is supported on unquantized `BF16` (`q38_bf16_tp8_ep1`, `80 → 128` lane pad).
- **MTP-3 speculative decoding (`NEXTN`):** MTP-3 speculative decoding is not yet output-equivalent on this stack (5-shot GSM8K `49.0%` vs `99.0%` without MTP) because PLE `ShortConvPool`/`NGramPool` state is not rolled back on rejected draft tokens; throughput is not reported until this is resolved.
- **`NVFP4` (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, `modelopt_fp4`) unsupported (`0.0%` GSM8K / `NaN` logprobs):** While the Pallas fused-RS `modelopt_fp4` kernel compiles and executes on TPU `v7x`, `Qwen4Exp` `FusedMoE` weight loading in `sglang-torchtpu v0.2.0` leaves `512`-expert `modelopt_fp4` input/group scales unmapped, producing `NaN` logprobs; throughput is not reported.

---

## 7. Reproducing the Benchmarks on GKE (`tpu7x`, `2x2x2` Single Host)

```bash
# 1. Create ConfigMap with runtime modifications, PLE equivalence check, and sweep harness
kubectl create configmap q38-sglang-sweep-scripts \
  --from-file=scripts/q38_patches.py \
  --from-file=scripts/q38_qwen4_exp_tpu_ple.py \
  --from-file=scripts/test_q38_ple_equivalence.py \
  --from-file=scripts/run_q38_sglang_sweep.py \
  --from-file=scripts/plan_q38_full.json \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Launch the single-host TPU v7x sweep job (edit image & GCS bucket placeholders first)
kubectl apply -f k8s/q38-sglang-v7x-sweep-job.yaml
```

## 8. Repository Layout

- [`scripts/q38_patches.py`](./scripts/q38_patches.py) — Idempotent container startup module applying the SGLang-TorchTPU `v0.2.0` `Qwen4Exp` modifications.
- [`scripts/q38_qwen4_exp_tpu_ple.py`](./scripts/q38_qwen4_exp_tpu_ple.py) — Static-shape TorchTPU layer-2 PLE implementation and exact `6 × 12`-bit `int32` limb `3`-gram hash.
- [`scripts/test_q38_ple_equivalence.py`](./scripts/test_q38_ple_equivalence.py) — CPU bit-exactness test suite comparing the TPU PLE implementation against the reference across `EXTEND`, `DECODE`, `TARGET_VERIFY`, and `200,000` random hash windows.
- [`scripts/run_q38_sglang_sweep.py`](./scripts/run_q38_sglang_sweep.py) — Single-host server lifecycle manager, `GSM8K` evaluator, and `sglang.benchmark.serving` runner.
- [`scripts/plan_q38_full.json`](./scripts/plan_q38_full.json) — Declarative 6-variant sweep plan (`FP8`, `MTP-3`, `DP=2 dp-attention`, `NVFP4`, `BF16 EP=8`, `BF16 EP=1`, `1K`–`128K` context).
- [`k8s/q38-sglang-v7x-sweep-job.yaml`](./k8s/q38-sglang-v7x-sweep-job.yaml) — Sanitized GKE TPU `v7x` single-host (`2x2x2`, `4` chips) Job manifest.
- [`results/`](./results/) — Consolidated [`SUMMARY.md`](./results/SUMMARY.md), [`summary.json`](./results/summary.json), and [`dev_bringup_summary.json`](./results/dev_bringup_summary.json).
