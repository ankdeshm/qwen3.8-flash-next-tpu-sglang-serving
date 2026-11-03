# Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration: 48 layers = 36 Gated-DeltaNet + 12 full attention, 512 experts top-10, hyper-connections x4, per-layer n-gram embedding, MTP head) on One TPU v7x Host (4 Chips / 8 TensorCores) with SGLang-TorchTPU v0.2.0

- **Updated (UTC):** `2026-11-03T10:14:03.260160+00:00`
- **Hardware:** one TPU `v7x` host (`4` chips = `8` TensorCore ranks, `768 GB` HBM), engine `SGLang-TorchTPU` (`sglang-torchtpu-diffusion:v0.2.0`), single-host mode (`FORCE_SINGLE_HOST=1`).
- **Checkpoints:** `Qwen/Qwen3.8-Flash-Next-FP8, Qwen/Qwen3.8-Flash-Next (BF16), RadixArk/Qwen3.8-Flash-Next-NVFP4`.
- **QSA Disclosure:** `indexer_n_heads: null` disables the sparse QSA top-k indexer so the 12 full-attention layers execute exact causal attention up to the 2,048-token QSA budget and full dense causal attention beyond 2,048 tokens (`4K`, `64K`, `128K`).
- **Method:** fixed-length ShareGPT-sampled prompts (`--dataset-name random --random-range-ratio 1.0`), every request decodes exact `output_len` tokens (EOS ignored); throughput is measured end-to-end at the client.

## Few-Shot GSM8K Quality & Speculative Acceptance Gate

| Variant | Layout / Precision | Questions (Shots) | Accuracy | Invalid | Output tok/s | Latency (s) | MTP Accept Len |
|---|---|---:|---:|---:|---:|---:|---:|
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | 200 (5-shot) | **17.0%** | 65.0% | 581.6 | 31.96 | - |

## Serving Throughput, Latency & Long-Context Matrix

| Variant | Layout / Precision | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | MTP Accept Len | Speedup vs FP8 Non-Spec | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 1 | 0.041 | 41.5 | 10.4 | 83.0 | 128 / 144 | 23.97 / 24.05 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 4 | 0.156 | 159.2 | 39.8 | 318.5 | 280 / 359 | 24.80 / 25.14 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 16 | 0.531 | 543.7 | 135.9 | 1,087.3 | 923 / 1,005 | 28.63 / 29.29 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 32 | 0.941 | 963.3 | 240.8 | 1,926.6 | 1,363 / 1,927 | 31.91 / 32.79 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `1024` / `1024` | 64 | 1.271 | 1,301.4 | 325.4 | 2,602.9 | 2,238 / 7,173 | 38.73 / 40.47 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 1 | 0.040 | 41.1 | 10.3 | 205.5 | 279 / 281 | 24.08 / 24.09 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 4 | 0.152 | 155.2 | 38.8 | 776.0 | 710 / 945 | 25.10 / 25.51 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 16 | 0.484 | 495.7 | 123.9 | 2,478.7 | 2,254 / 3,635 | 30.17 / 31.93 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 32 | 0.799 | 818.6 | 204.7 | 4,093.2 | 4,137 / 7,245 | 34.97 / 38.86 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `4096` / `1024` | 64 | 1.026 | 1,050.8 | 262.7 | 5,254.0 | 8,892 / 17,896 | 43.89 / 52.14 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 1 | 0.035 | 35.5 | 8.9 | 2,304.8 | 3,874 / 3,886 | 24.42 / 24.42 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 4 | 0.095 | 97.4 | 24.4 | 6,334.2 | 9,514 / 15,140 | 31.77 / 37.30 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `65536` / `1024` | 8 | 0.136 | 139.2 | 34.8 | 9,050.2 | 17,026 / 30,179 | 40.85 / 53.71 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `131072` / `512` | 1 | 0.048 | 24.5 | 6.1 | 6,294.6 | 8,212 / 8,234 | 24.80 / 24.80 | - | - | `PASS` |
| `q38_fp8_tp8_ep8` | FP8 W8A8 (128x128 blockwise runtime scales), TP=8 x EP=8, BF16 KV | `131072` / `512` | 4 | 0.086 | 44.2 | 11.0 | 11,348.2 | 20,226 / 32,212 | 51.16 / 74.73 | - | - | `PASS` |
