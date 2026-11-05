# Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration: 48 layers = 36 Gated-DeltaNet + 12 full attention, 512 experts top-10, hyper-connections x4, per-layer n-gram embedding, MTP head) on One TPU v7x Host (4 Chips / 8 TensorCores) with SGLang-TorchTPU v0.2.0

- **Updated (UTC):** `2026-11-04T04:02:05+00:00`
- **Hardware:** one TPU `v7x` host (`4` chips = `8` TensorCore ranks, `768 GB` HBM), `SGLang-TorchTPU` (`sglang-torchtpu-diffusion:v0.2.0`), single-host mode (`FORCE_SINGLE_HOST=1`).
- **Checkpoints:** `Qwen/Qwen3.8-Flash-Next-FP8, Qwen/Qwen3.8-Flash-Next (BF16), RadixArk/Qwen3.8-Flash-Next-NVFP4`.
- **QSA Note:** `indexer_n_heads: null` disables the non-TPU QSA sparse top-k indexer so the `12` full-attention layers run causal attention up to the `2,048`-token QSA window and full dense causal attention (`Pallas RPA v3`) beyond `2,048` tokens (`4K`, `64K`, `128K`).
- **Method:** fixed-length ShareGPT-sampled prompts (`--dataset-name random --random-range-ratio 1.0`), greedy decoding (`temperature=0`), exact `output_len` tokens decoded per request (EOS ignored); throughput and latencies are measured end-to-end at the client.

## 1. Few-Shot GSM8K Quality & Speculative Acceptance

| Variant | Layout / Precision | Harness | Questions | Accuracy | Invalid | GSM8K Out tok/s | Latency (s) | MTP Accept Len |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | raw `/generate` (`stop=["Question"]`) | 200 | **17.0%**¹ | 65.0% | 581.6 | 31.96 | — |
| `q38_fp8_tp8_ep8_mtp3` | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **49.0%**² | 0.0% | 841.8 | 48.46 | 3.08 |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **99.0%** | 0.0% | 288.1 | 100.04 | — |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **98.5%** | 0.0% | 498.4 | 58.48 | — |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **98.5%** | 0.0% | 408.1 | 70.99 | — |
| `q38_nvfp4_tp8_ep8` | NVFP4 experts (RadixArk modelopt_fp4, group_size=16) via Pallas fused-RS MoE, TP=8 x EP=8, BF16 KV | chat `/v1/chat/completions` (`enable_thinking=false`) | 200 | **0.0%** (`NaN` logprobs)³ | 0.0% | — | — | — |

¹ Legacy raw `/generate` GSM8K passes `stop=["Question"]` without chat templating (`enable_thinking=false`), truncating Chain-of-Thought responses when `Question` appears (`65.0%` unparsed `invalid`); Chat-API evaluation on the same blockwise `FP8` checkpoint (`q38_fp8_dp2attn_ep8`) achieves **`99.0%`** (`0.0%` invalid).
² `q38_fp8_tp8_ep8_mtp3` passes short greedy smoke probes (`0` NaN logprobs) and reaches `3.08` mean accepted tokens on GSM8K, but multi-step arithmetic accuracy drops to `49.0%` because layer-2 PLE `ShortConvPool` and `NGramPool` states are not rolled back across rejected draft tokens during `TARGET_VERIFY`.
³ `q38_nvfp4_tp8_ep8` (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, `modelopt_fp4`) produces `NaN` logprobs (`0.0%` GSM8K) because `512`-expert `Qwen4Exp` `modelopt_fp4` expert input/group scales are unmapped in `sglang-torchtpu v0.2.0`; invalid throughput cells are omitted below.

## 2. Serving Throughput, Latency & Long-Context Matrix (Valid Configurations)

| Variant | Layout / Precision | In / Out | Conc. | Prompts | Req/s | Output tok/s | Out tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | MTP Accept Len | vs `FP8 TP8 EP8` |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 1 | 4 | 0.041 | **41.5** | 10.4 | 83.0 | 128 / 144 | 23.97 / 24.05 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 4 | 12 | 0.156 | **159.2** | 39.8 | 318.5 | 280 / 359 | 24.80 / 25.14 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 16 | 48 | 0.531 | **543.7** | 135.9 | 1,087.3 | 923 / 1,005 | 28.63 / 29.29 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 32 | 96 | 0.941 | **963.3** | 240.8 | 1,926.6 | 1,363 / 1,927 | 31.91 / 32.79 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 64 | 192 | 1.271 | **1,301.4** | 325.4 | 2,602.9 | 2,238 / 7,173 | 38.73 / 40.47 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 1 | 4 | 0.040 | **41.1** | 10.3 | 205.5 | 279 / 281 | 24.08 / 24.09 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 4 | 12 | 0.152 | **155.2** | 38.8 | 776.0 | 710 / 945 | 25.10 / 25.51 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 16 | 48 | 0.484 | **495.7** | 123.9 | 2,478.7 | 2,254 / 3,635 | 30.17 / 31.93 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 32 | 96 | 0.799 | **818.6** | 204.7 | 4,093.2 | 4,137 / 7,245 | 34.97 / 38.86 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 64 | 192 | 1.026 | **1,050.8** | 262.7 | 5,254.0 | 8,892 / 17,896 | 43.89 / 52.14 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 1 | 2 | 0.035 | **35.5** | 8.9 | 2,304.8 | 3,874 / 3,886 | 24.42 / 24.42 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 4 | 8 | 0.095 | **97.4** | 24.4 | 6,334.2 | 9,514 / 15,140 | 31.77 / 37.30 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 8 | 16 | 0.136 | **139.2** | 34.8 | 9,050.2 | 17,026 / 30,179 | 40.85 / 53.71 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `131072` / `512` | 1 | 2 | 0.048 | **24.5** | 6.1 | 6,294.6 | 8,212 / 8,234 | 24.80 / 24.80 | — | 1.00x |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `131072` / `512` | 4 | 8 | 0.086 | **44.2** | 11.0 | 11,348.2 | 20,226 / 32,212 | 51.16 / 74.73 | — | 1.00x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `1024` / `1024` | 1 | 4 | 0.083 | **85.3** | 21.3 | 170.6 | 172 / 177 | 11.53 / 14.62 | 3.03 | 2.05x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `1024` / `1024` | 4 | 12 | 0.291 | **298.1** | 74.5 | 596.1 | 206 / 608 | 12.40 / 14.65 | 2.96 | 1.87x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `1024` / `1024` | 16 | 48 | 0.836 | **856.2** | 214.0 | 1,712.4 | 223 / 1,123 | 16.24 / 27.32 | 2.85 | 1.57x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `1024` / `1024` | 32 | 96 | 1.355 | **1,387.5** | 346.9 | 2,774.9 | 283 / 1,991 | 18.63 / 32.95 | 2.83 | 1.44x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `4096` / `1024` | 1 | 4 | 0.101 | **103.2** | 25.8 | 516.0 | 484 / 486 | 9.29 / 10.36 | 2.84 | 2.51x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `4096` / `1024` | 4 | 12 | 0.277 | **283.8** | 71.0 | 1,419.0 | 518 / 1,320 | 12.23 / 16.60 | 2.85 | 1.83x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `4096` / `1024` | 16 | 48 | 0.712 | **729.0** | 182.3 | 3,645.1 | 533 / 4,092 | 18.41 / 26.11 | 2.87 | 1.47x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `4096` / `1024` | 32 | 96 | 1.019 | **1,043.6** | 260.9 | 5,218.0 | 601 / 7,714 | 25.22 / 42.48 | 2.89 | 1.27x |
| `q38_fp8_tp8_ep8_mtp3`† | FP8 W8A8 (blockwise), TP=8 x EP=8, BF16 KV + MTP speculative decoding (NEXTN, 3 steps, topk 1, 4 draft tokens) | `65536` / `1024` | 1 | 2 | 0.080 | **82.3** | 20.6 | 5,349.3 | 3,925 / 3,935 | 8.31 / 8.81 | 2.89 | 2.32x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `1024` / `1024` | 1 | 4 | 0.030 | **30.8** | 7.7 | 61.6 | 259 / 310 | 32.25 / 32.26 | — | 0.74x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `1024` / `1024` | 4 | 12 | 0.118 | **121.2** | 30.3 | 242.3 | 299 / 434 | 32.69 / 32.86 | — | 0.76x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `1024` / `1024` | 16 | 48 | 0.425 | **435.1** | 108.8 | 870.3 | 897 / 1,563 | 35.98 / 36.36 | — | 0.80x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `1024` / `1024` | 32 | 96 | 0.740 | **758.0** | 189.5 | 1,516.1 | 1,561 / 3,165 | 40.67 / 41.74 | — | 0.79x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `1024` / `1024` | 64 | 192 | 1.244 | **1,273.4** | 318.4 | 2,546.8 | 3,357 / 5,700 | 46.97 / 49.60 | — | 0.98x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `4096` / `1024` | 1 | 4 | 0.030 | **30.3** | 7.6 | 151.6 | 715 / 769 | 32.28 / 32.29 | — | 0.74x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `4096` / `1024` | 16 | 48 | 0.381 | **390.6** | 97.7 | 1,953.1 | 3,358 / 5,765 | 38.04 / 40.45 | — | 0.79x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `4096` / `1024` | 64 | 192 | 0.939 | **961.4** | 240.4 | 4,807.0 | 11,990 / 21,544 | 55.15 / 65.85 | — | 0.91x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `65536` / `1024` | 1 | 2 | 0.022 | **23.0** | 5.7 | 1,493.6 | 11,157 / 11,171 | 32.63 / 32.63 | — | 0.65x |
| `q38_fp8_dp2attn_ep8` | FP8 W8A8 (blockwise), TP=8 x DP=2 (dp-attention, 2x attn-TP4 + DP LM head) x EP=8, BF16 KV | `65536` / `1024` | 4 | 8 | 0.069 | **70.3** | 17.6 | 4,570.0 | 17,679 / 24,412 | 39.89 / 46.31 | — | 0.72x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 1 | 4 | 0.035 | **35.9** | 9.0 | 71.8 | 179 / 181 | 27.70 / 27.71 | — | 0.86x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 4 | 12 | 0.133 | **135.8** | 34.0 | 271.6 | 484 / 595 | 28.96 / 29.33 | — | 0.85x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 16 | 48 | 0.465 | **475.7** | 118.9 | 951.5 | 598 / 1,034 | 33.07 / 33.42 | — | 0.88x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 32 | 96 | 0.850 | **870.5** | 217.6 | 1,741.0 | 1,191 / 1,907 | 35.63 / 36.33 | — | 0.90x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `1024` / `1024` | 64 | 192 | 1.166 | **1,194.2** | 298.5 | 2,388.3 | 2,212 / 7,700 | 41.76 / 43.68 | — | 0.92x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 1 | 4 | 0.035 | **35.5** | 8.9 | 177.4 | 486 / 490 | 27.72 / 27.74 | — | 0.86x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 4 | 12 | 0.130 | **133.0** | 33.2 | 665.0 | 906 / 1,295 | 29.48 / 29.86 | — | 0.86x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 16 | 48 | 0.426 | **436.4** | 109.1 | 2,181.8 | 2,217 / 3,982 | 34.74 / 36.33 | — | 0.88x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 32 | 96 | 0.732 | **749.1** | 187.3 | 3,745.3 | 3,973 / 7,466 | 38.71 / 42.21 | — | 0.92x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `4096` / `1024` | 64 | 192 | 0.957 | **979.7** | 244.9 | 4,898.5 | 7,896 / 18,342 | 47.93 / 54.99 | — | 0.93x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `65536` / `1024` | 1 | 2 | 0.031 | **31.5** | 7.9 | 2,044.5 | 3,813 / 3,818 | 28.08 / 28.09 | — | 0.89x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `65536` / `1024` | 4 | 8 | 0.087 | **88.9** | 22.2 | 5,779.1 | 9,374 / 14,943 | 35.85 / 41.30 | — | 0.91x |
| `q38_bf16_tp8_ep8` | BF16 unquantized weights + BF16 KV, TP=8 x EP=8 | `131072` / `512` | 1 | 2 | 0.044 | **22.6** | 5.6 | 5,803.6 | 8,108 / 8,114 | 28.46 / 28.47 | — | 0.92x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `1024` / `1024` | 1 | 4 | 0.034 | **34.9** | 8.7 | 69.8 | 237 / 240 | 28.45 / 28.45 | — | 0.84x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `1024` / `1024` | 4 | 12 | 0.128 | **131.1** | 32.8 | 262.3 | 692 / 862 | 29.79 / 30.35 | — | 0.82x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `1024` / `1024` | 16 | 48 | 0.408 | **417.4** | 104.4 | 834.9 | 1,110 / 1,520 | 37.25 / 38.11 | — | 0.77x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `1024` / `1024` | 32 | 96 | 0.692 | **708.8** | 177.2 | 1,417.6 | 1,994 / 2,821 | 43.23 / 44.53 | — | 0.74x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `1024` / `1024` | 64 | 192 | 0.925 | **946.7** | 236.7 | 1,893.5 | 3,290 / 10,332 | 54.68 / 57.22 | — | 0.73x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `4096` / `1024` | 1 | 4 | 0.034 | **34.3** | 8.6 | 171.6 | 688 / 690 | 28.48 / 28.48 | — | 0.84x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `4096` / `1024` | 16 | 48 | 0.365 | **373.8** | 93.4 | 1,868.9 | 3,279 / 5,897 | 40.02 / 42.28 | — | 0.75x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `4096` / `1024` | 64 | 192 | 0.740 | **758.1** | 189.5 | 3,790.7 | 11,751 / 26,509 | 63.05 / 73.86 | — | 0.72x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `65536` / `1024` | 1 | 2 | 0.029 | **29.2** | 7.3 | 1,899.5 | 5,519 / 5,539 | 28.84 / 28.84 | — | 0.82x |
| `q38_bf16_tp8_ep1` | BF16 unquantized weights + BF16 KV, TP=8 x EP=1 (expert tensor-parallel, 80->128 gmm_v2 lane pad) | `131072` / `512` | 1 | 2 | 0.038 | **19.3** | 4.8 | 4,972.5 | 11,512 / 11,575 | 29.22 / 29.22 | — | 0.79x |

† `q38_fp8_tp8_ep8_mtp3` uses greedy decoding (`temperature=0`) on synthetic fixed-length prompts (`2.83–3.03` mean accepted tokens) and exhibits `49.0%` GSM8K accuracy on long Chain-of-Thought arithmetic due to unrolled PLE side state across rejected draft tokens.
