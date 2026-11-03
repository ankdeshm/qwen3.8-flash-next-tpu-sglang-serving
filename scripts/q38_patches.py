#!/usr/bin/env python3
"""Runtime model-enablement patches for Qwen3.8-Flash-Next on SGLang-TorchTPU v0.2.0.

Applied in-place inside the container (site-packages) before any server starts; idempotent.
apply_all() returns the description of EVERY active patch (newly applied or already present),
which the sweep harness records in summary.json["runtime_patches"].
"""
from __future__ import annotations

from pathlib import Path
import sys

SITE = Path("/usr/local/lib/python3.12/site-packages/sglang")
HERE = Path(__file__).resolve().parent
GATED_NORM_PY = SITE / "srt/layers/attention/native_torch/layernorm_gated.py"
QWEN4_EXP_PY = SITE / "srt/models/qwen4_exp.py"
QWEN4_EXP_MTP_PY = SITE / "srt/models/qwen4_exp_mtp.py"
QWEN2_MOE_PY = SITE / "srt/models/qwen2_moe.py"
MODELOPT_PY = SITE / "srt/layers/quantization/modelopt_quant.py"
PLE_POOL_PY = SITE / "srt/mem_cache/ple_state_pool.py"
PLE_PATCH = HERE / "q38_qwen4_exp_tpu_ple.py"
PLE_MARKER = "# q38 TPU bring-up patch for Qwen4-Exp per-layer n-gram embeddings (PLE)."
MTP_MARKER = "# q38: int32 lower-triangular prefix sum instead of int64 cumsum (TPU XLA U64 abort)"
NVFP4_MARKER = "# q38: TPU NVFP4 MoE method never sets the flashinfer-TRTLLM layout flag"
QWEN2_MOE_NVFP4_MARKER = "# q38: opt Qwen2MoeSparseMoeBlock into TPU NVFP4 fused-RS backend reduction"
PLE_POOL_MARKER = "# q38: NGramPool stores token-id history as int32 (TPU graph has no s64 PLE state)"

DESC_GATED_NORM = (
    "layernorm_gated.py (native-torch RMSNormGated): add the sigmoid gate, rmsnorm(x)*sigmoid(z), used by the "
    "Qwen3.8 Gated-DeltaNet output norm (TPU fallback only implemented SiLU)"
)
DESC_PLE = (
    "qwen4_exp.py PLE (per-layer n-gram embedding, layer 2) made TorchTPU-traceable (block appended from "
    "q38_qwen4_exp_tpu_ple.py): _PLEBatch msgspec.Struct -> NamedTuple (Dynamo cannot construct msgspec "
    "Structs); padded static shapes (no slicing by host real-token counts -> no per-prompt-length recompiles); "
    "flat token-major EXTEND layout for n-gram windows / short-conv taps / state commit; fresh requests "
    "(extend_prefix_lens==0) start from zero short-conv state + EOS n-gram history in-graph (TpuMambaPool."
    "clear_slots never resets the PLE side-state pools -> cross-request state leak otherwise); int32-only "
    "integer math with an exact base-2^12-limb evaluation of the int64 n-gram hash (the TPU XLA backend "
    "aborts on the s64 multiply/xor/remainder/cumsum), hash constants checked against the loaded buffers; no "
    "CUDA/Triton fused kernels, no cummax/cumsum. Same math: CPU-verified bit-exact vs upstream (extend with "
    "padding/chunk continuation/EOS, decode with padded rows, MTP verify, 200k-window hash sweep)"
)
DESC_POOL = (
    "ple_state_pool.py NGramPool: int32 instead of int64 storage/indices (token ids < 2^31) so the TPU graph "
    "carries no s64 PLE state"
)
DESC_MTP = (
    "qwen4_exp_mtp.py Qwen4ExpForCausalLMMTP._set_hc_logits_hidden_states: replace int64 torch.cumsum with "
    "int32 lower-triangular prefix sum + clamp(min=0) so MTP draft-extend avoids TPU XLA U64 windowing abort"
)
DESC_NVFP4 = (
    "modelopt_quant.py + qwen2_moe.py: set TpuModelOptNvFp4FusedMoEMethod.use_flashinfer_trtllm_weight_layout = "
    "False, opt Qwen2MoeSparseMoeBlock into allow_backend_reduction=True for FP4 quant configs, and pre-scale "
    "already-TP-summed fused-RS routed outputs by 1/tp_size before adding the unreduced shared expert so the "
    "single downstream TP all-reduce sums routed + shared experts exactly once"
)


def _patch_gated_norm() -> str:
    src = GATED_NORM_PY.read_text()
    if '"sigmoid"' not in src:
        edits = (
            ('if activation not in ("silu", "swish"):', 'if activation not in ("silu", "swish", "sigmoid"):'),
            ("        self.eps = eps\n", "        self.eps = eps\n        self.activation = activation\n"),
            (
                "out = out * torch.nn.functional.silu(z.float())",
                'out = out * (torch.sigmoid(z.float()) if self.activation == "sigmoid" '
                "else torch.nn.functional.silu(z.float()))",
            ),
        )
        for old, new in edits:
            assert src.count(old) == 1, f"unexpected layernorm_gated.py layout near: {old!r}"
            src = src.replace(old, new, 1)
        GATED_NORM_PY.write_text(src)
    return DESC_GATED_NORM


def _patch_qwen4_ple() -> str:
    src = QWEN4_EXP_PY.read_text()
    patch = PLE_PATCH.read_text()
    if PLE_MARKER in src and src.endswith(patch):
        return DESC_PLE
    if PLE_MARKER in src:  # an older revision of the block is appended: strip it back to upstream
        head = src[: src.index(PLE_MARKER)]
        src = head[: head.rstrip("\n").rfind("\n# ====")]
    assert "class _PLEBatch(msgspec.Struct, frozen=True):" in src, "unexpected qwen4_exp.py layout"
    assert PLE_MARKER not in src
    QWEN4_EXP_PY.write_text(src.rstrip("\n") + "\n" + patch)
    return DESC_PLE


def _patch_ngram_pool_int32() -> str:
    src = PLE_POOL_PY.read_text()
    if PLE_POOL_MARKER not in src:
        start = src.index("class NGramPool:")
        head, body = src[:start], src[start:]
        assert body.count("torch.long") >= 2, "unexpected ple_state_pool.py layout"
        PLE_POOL_PY.write_text(head + PLE_POOL_MARKER + "\n" + body.replace("torch.long", "torch.int32"))
    return DESC_POOL


def _patch_qwen4_mtp_int32() -> str:
    src = QWEN4_EXP_MTP_PY.read_text()
    if MTP_MARKER not in src:
        old = (
            "            last_index = (\n"
            "                torch.cumsum(forward_batch.extend_seq_lens.to(torch.int64), dim=0) - 1\n"
            "            )\n"
            "            hc_hidden_states = hc_hidden_states[last_index]"
        )
        new = (
            f"            {MTP_MARKER}\n"
            "            _lens_i32 = forward_batch.extend_seq_lens.to(torch.int32)\n"
            "            _seq_idx = torch.arange(_lens_i32.shape[0], device=_lens_i32.device, dtype=torch.int32)\n"
            "            _lower = (_seq_idx.unsqueeze(1) >= _seq_idx.unsqueeze(0)).to(torch.int32)\n"
            "            last_index = ((_lower * _lens_i32.unsqueeze(0)).sum(dim=1, dtype=torch.int32) - 1).clamp(min=0)\n"
            "            hc_hidden_states = hc_hidden_states[last_index]"
        )
        assert src.count(old) == 1, "unexpected qwen4_exp_mtp.py layout"
        QWEN4_EXP_MTP_PY.write_text(src.replace(old, new, 1))
    return DESC_MTP


def _patch_nvfp4() -> str:
    src = MODELOPT_PY.read_text()
    if NVFP4_MARKER not in src:
        assert "class TpuModelOptNvFp4FusedMoEMethod(ModelOptNvFp4FusedMoEMethod):" in src
        MODELOPT_PY.write_text(
            src.rstrip("\n")
            + "\n\n\n" + NVFP4_MARKER + "\n"
            "# (its __init__ skips the parent's); FusedMoE.weight_loader reads it -> AttributeError.\n"
            "# TPU keeps the canonical w13 layout, i.e. False.\n"
            "TpuModelOptNvFp4FusedMoEMethod.use_flashinfer_trtllm_weight_layout = False\n"
        )
    moe_src = QWEN2_MOE_PY.read_text()
    if QWEN2_MOE_NVFP4_MARKER not in moe_src:
        old_init = (
            "            enable_qwen35_fp8_deferred_finalize=(\n"
            '                config.model_type == "qwen3_5_moe_text" and uses_cutedsl_ar_fusion()\n'
            "            ),\n"
            "        )\n"
        )
        new_init = (
            old_init
            + f"        {QWEN2_MOE_NVFP4_MARKER}\n"
            '        if quant_config is not None and "fp4" in quant_config.get_name().lower():\n'
            "            self.experts.allow_backend_reduction = True\n"
        )
        old_ret = "        return self.experts(hidden_states, topk_output)\n"
        new_ret = (
            "        _routed = self.experts(hidden_states, topk_output)\n"
            '        if getattr(self.experts, "handles_moe_reduction", False) and self.tp_size > 1:\n'
            "            _routed = (_routed.float() * (1.0 / float(self.tp_size))).to(_routed.dtype)\n"
            "        return _routed\n"
        )
        old_m0 = "            final_hidden_states = self.experts(hidden_states, topk_output)\n"
        new_m0 = (
            "            final_hidden_states = self.experts(hidden_states, topk_output)\n"
            '            if getattr(self.experts, "handles_moe_reduction", False) and self.tp_size > 1:\n'
            "                final_hidden_states = (final_hidden_states.float() * (1.0 / float(self.tp_size))).to(final_hidden_states.dtype)\n"
        )
        assert moe_src.count(old_init) == 1, "unexpected qwen2_moe.py init layout"
        assert moe_src.count(old_ret) == 1, "unexpected qwen2_moe.py _forward_router_experts layout"
        assert moe_src.count(old_m0) == 1, "unexpected qwen2_moe.py M=0 layout"
        moe_src = moe_src.replace(old_init, new_init, 1).replace(old_ret, new_ret, 1).replace(old_m0, new_m0, 1)
        QWEN2_MOE_PY.write_text(moe_src)
    return DESC_NVFP4


def apply_all() -> list[str]:
    active = [
        _patch_gated_norm(),
        _patch_qwen4_ple(),
        _patch_ngram_pool_int32(),
        _patch_qwen4_mtp_int32(),
        _patch_nvfp4(),
    ]
    # drop stale bytecode so every spawned rank re-reads the patched sources
    for py in (GATED_NORM_PY, QWEN4_EXP_PY, QWEN4_EXP_MTP_PY, QWEN2_MOE_PY, MODELOPT_PY, PLE_POOL_PY):
        for pyc in py.parent.glob(f"__pycache__/{py.stem}.*.pyc"):
            pyc.unlink(missing_ok=True)
    return active


if __name__ == "__main__":
    for line in apply_all():
        print("q38 patch active:", line)
    sys.exit(0)
