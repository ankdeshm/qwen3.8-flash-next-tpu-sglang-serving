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
SERVER_ARGS_PY = SITE / "srt/server_args.py"
GATED_NORM_PY = SITE / "srt/layers/attention/native_torch/layernorm_gated.py"
QWEN4_EXP_PY = SITE / "srt/models/qwen4_exp.py"
QWEN4_EXP_MTP_PY = SITE / "srt/models/qwen4_exp_mtp.py"
QWEN2_MOE_PY = SITE / "srt/models/qwen2_moe.py"
MODELOPT_PY = SITE / "srt/layers/quantization/modelopt_quant.py"
PLE_POOL_PY = SITE / "srt/mem_cache/ple_state_pool.py"
ROTARY_BASE_PY = SITE / "srt/layers/rotary_embedding/base.py"
ROTARY_MROPE_PY = SITE / "srt/layers/rotary_embedding/mrope.py"
HYPERCONN_PY = SITE / "srt/layers/hyperconnection.py"
HYBRID_ATTN_PY = SITE / "srt/layers/attention/hybrid_linear_attn_backend.py"
TPU_ROOTS = [
    Path("/workspace/sglang/sglang_tpu"),
    Path("/usr/local/lib/python3.12/site-packages/sglang_tpu"),
]
PLE_PATCH = HERE / "q38_qwen4_exp_tpu_ple.py"
PLE_MARKER = "# q38 TPU bring-up patch for Qwen4-Exp per-layer n-gram embeddings (PLE)."
LM_ONLY_MARKER = "# q38: register Qwen4ExpForConditionalGeneration in LANGUAGE_MODEL_ONLY_ARCHITECTURES"
MTP_MARKER = "# q38: int32 lower-triangular prefix sum instead of int64 cumsum (TPU XLA int32 compatibility)"
MTP_COS_SIN_MARKER = "# q38: pre-cast MTP rotary cos_sin_cache to bfloat16 before direct_compile"
NVFP4_MARKER = "# q38: TPU NVFP4 MoE method never sets the flashinfer-TRTLLM layout flag"
QWEN2_MOE_NVFP4_MARKER = "# q38: opt Qwen2MoeSparseMoeBlock into TPU NVFP4 fused-RS backend reduction"
PLE_POOL_MARKER = "# q38: NGramPool stores token-id history as int32 (TPU graph has no s64 PLE state)"
PLE_POOL_INTER_MARKER = "# q38: index_put_ on root intermediate_context instead of slice .copy_()"
ROTARY_MARKER = "# q38: disable inner torch.compile(dynamic=True) on apply_rotary_emb for TPU direct_compile"
HYPERCONN_MARKER = "# q38: disable inner torch.compile on GatedResidual._mix_compute/_combine_compute for TPU direct_compile"
DIRECT_PLE_MARKER = "# q38: lift PLE short_conv_pool and ngram_pool buffers in HybridLinearKVPoolState"
MTP_HC_MARKER = "# q38: Qwen4-Exp hc_count>1 carry_size (input_embeds_hidden_size) + _mix_streams before lm_head"
PLE_VERIFY_SCATTER_MARKER = "# q38: static-shape int32 TPU scatter for PLE side states after MTP verify"

DESC_GATED_NORM = (
    "layernorm_gated.py (native-torch RMSNormGated): add the sigmoid gate, rmsnorm(x)*sigmoid(z), used by the "
    "Qwen3.8 Gated-DeltaNet output norm (TPU fallback only implemented SiLU)"
)
DESC_PLE = (
    "qwen4_exp.py + server_args.py PLE & whole-model language_model_only DecoderTpuCompilePolicy (block appended from "
    "q38_qwen4_exp_tpu_ple.py): register Qwen4ExpForConditionalGeneration in LANGUAGE_MODEL_ONLY_ARCHITECTURES; "
    "_PLEBatch msgspec.Struct -> NamedTuple; padded static shapes; flat token-major EXTEND layout; fresh requests "
    "zero-masked in-graph; exact base-2^12 int32 limb n-gram hash; root short_conv_pool index_put_ writes; and "
    "whole-model Qwen4ExpForConditionalGeneration.forward compiling LogitsProcessor + lm_head in-graph when "
    "language_model_only=True (enables DP=2 dp-attention and MTP hc_hidden_states output leaves)"
)
DESC_POOL = (
    "ple_state_pool.py NGramPool: int32 storage/indices and root index_put_ in set_intermediate_context (no s64 "
    "state or slice-view .copy_() under TPU direct_compile)"
)
DESC_MTP = (
    "qwen4_exp_mtp.py + rotary_embedding/{base,mrope}.py + hyperconnection.py + sglang_tpu MTP/direct_compile: int32 "
    "prefix sum in Qwen4ExpForCausalLMMTP; disable nested torch.compile on partial-RoPE apply_rotary_emb and "
    "GatedResidual._mix_compute/_combine_compute so direct_compile (make_fx) traces without placeholder materialization "
    "errors; lift req_index_to_mamba_index_mapping + ShortConvPool + NGramPool buffers in HybridLinearKVPoolState; "
    "support hc_count=4 (10,240-dim carry + _mix_streams before lm_head) in TpuMtpDraftWorker; and static-shape int32 "
    "PLE verify commit in HybridLinearAttnBackend"
)
DESC_NVFP4 = (
    "modelopt_quant.py + qwen2_moe.py + fused_rs/{fused_moe,gmm_fused_rs_nodedup}.py: set "
    "TpuModelOptNvFp4FusedMoEMethod.use_flashinfer_trtllm_weight_layout = False, opt Qwen2MoeSparseMoeBlock into "
    "allow_backend_reduction=True for FP4 quant configs, pre-scale already-TP-summed fused-RS routed outputs by "
    "1/tp_size before adding the unreduced shared expert, widen non-SP fused-RS routing broadcast from bfloat16 "
    "(exact up to 256 experts; float32 preserves exact 0..511 indices) to float32 for 512-expert "
    "models, and migrate deprecated pltpu.{semaphore_signal,semaphore_wait,DeviceIdType} calls to pl.* for JAX 0.11+"
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
    sa_src = SERVER_ARGS_PY.read_text()
    if LM_ONLY_MARKER not in sa_src:
        old_sa = (
            "    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (\n"
            '        "MuseGlimmerForConditionalGeneration",\n'
            '        "Cosmos3ForConditionalGeneration",\n'
            '        "Cosmos3EdgeForConditionalGeneration",\n'
            '        "Glm5NextForConditionalGeneration",\n'
            "    )"
        )
        new_sa = (
            f"    {LM_ONLY_MARKER}\n"
            "    LANGUAGE_MODEL_ONLY_ARCHITECTURES = (\n"
            '        "MuseGlimmerForConditionalGeneration",\n'
            '        "Cosmos3ForConditionalGeneration",\n'
            '        "Cosmos3EdgeForConditionalGeneration",\n'
            '        "Glm5NextForConditionalGeneration",\n'
            '        "Qwen4ExpForConditionalGeneration",\n'
            "    )"
        )
        assert sa_src.count(old_sa) == 1, "unexpected server_args.py LANGUAGE_MODEL_ONLY_ARCHITECTURES layout"
        SERVER_ARGS_PY.write_text(sa_src.replace(old_sa, new_sa, 1))
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
    changed = False
    if PLE_POOL_MARKER not in src:
        start = src.index("class NGramPool:")
        head, body = src[:start], src[start:]
        assert body.count("torch.long") >= 2, "unexpected ple_state_pool.py layout"
        src = head + PLE_POOL_MARKER + "\n" + body.replace("torch.long", "torch.int32")
        changed = True
    if PLE_POOL_INTER_MARKER not in src:
        old_inter = (
            "            if indices is None:\n"
            "                self.intermediate_context[: context.shape[0], : context.shape[1]].copy_(\n"
            "                    context\n"
            "                )\n"
            "            else:\n"
            "                self.intermediate_context[\n"
            "                    indices.to(device=self.context.device, dtype=torch.int32),\n"
            "                    : context.shape[1],\n"
            "                ] = context"
        )
        new_inter = (
            f"            {PLE_POOL_INTER_MARKER}\n"
            "            _dst_rows = (\n"
            "                torch.arange(context.shape[0], device=self.context.device, dtype=torch.int32)\n"
            "                if indices is None\n"
            "                else indices.to(device=self.context.device, dtype=torch.int32)\n"
            "            )\n"
            "            if context.shape[1] == self.intermediate_context.shape[1]:\n"
            "                self.intermediate_context[_dst_rows] = context\n"
            "            else:\n"
            "                _cols = torch.arange(context.shape[1], device=self.context.device, dtype=torch.int32)\n"
            "                self.intermediate_context[_dst_rows.unsqueeze(1), _cols.unsqueeze(0)] = context"
        )
        assert src.count(old_inter) == 1, "unexpected NGramPool.set_intermediate_context layout"
        src = src.replace(old_inter, new_inter, 1)
        changed = True
    if changed:
        PLE_POOL_PY.write_text(src)
    return DESC_POOL


def _patch_qwen4_mtp_and_tpu_spec() -> str:
    # 1) qwen4_exp_mtp.py: int32 prefix sum + bfloat16 cos_sin_cache after load_weights
    src = QWEN4_EXP_MTP_PY.read_text()
    changed_mtp = False
    MTP_FULL_CAPTURE_MARKER = "# q38: preserve FULL verify rows in _set_hc_logits_hidden_states"
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
        src = src.replace(old, new, 1)
        changed_mtp = True
    if MTP_FULL_CAPTURE_MARKER not in src:
        old_cap = (
            "        if hc_hidden_states is None:\n"
            "            return\n"
        )
        new_cap = (
            "        if hc_hidden_states is None:\n"
            "            return\n"
            f"        {MTP_FULL_CAPTURE_MARKER}\n"
            "        if not forward_batch.capture_hidden_mode.need_capture():\n"
            "            return\n"
        )
        old_elif = (
            "        elif (\n"
            "            forward_batch.extend_seq_lens is not None\n"
            "            and hc_hidden_states.shape[0] != forward_batch.extend_seq_lens.shape[0]\n"
            "        ):"
        )
        new_elif = (
            "        elif (\n"
            "            forward_batch.capture_hidden_mode.is_last()\n"
            "            and forward_batch.extend_seq_lens is not None\n"
            "            and hc_hidden_states.shape[0] != forward_batch.extend_seq_lens.shape[0]\n"
            "        ):"
        )
        assert src.count(old_cap) == 1, "unexpected qwen4_exp_mtp.py hc_hidden_states guard"
        assert src.count(old_elif) == 1, "unexpected qwen4_exp_mtp.py elif extend_seq_lens guard"
        src = src.replace(old_cap, new_cap, 1).replace(old_elif, new_elif, 1)
        changed_mtp = True
    if MTP_COS_SIN_MARKER not in src:
        tail = (
            f"\n\n{MTP_COS_SIN_MARKER}\n"
            "_q38_orig_mtp_load_weights = Qwen4ExpForCausalLMMTP.load_weights\n\n\n"
            "def _q38_mtp_load_weights(self, weights, *args, **kwargs):\n"
            "    res = _q38_orig_mtp_load_weights(self, weights, *args, **kwargs)\n"
            "    for mod in self.modules():\n"
            '        csc = getattr(mod, "cos_sin_cache", None)\n'
            '        if isinstance(csc, torch.Tensor) and csc.device.type != "meta" and csc.dtype != torch.bfloat16:\n'
            "            mod.cos_sin_cache = csc.to(device=csc.device, dtype=torch.bfloat16)\n"
            "    return res\n\n\n"
            "Qwen4ExpForCausalLMMTP.load_weights = _q38_mtp_load_weights\n"
        )
        src = src.rstrip("\n") + tail
        changed_mtp = True
    if changed_mtp:
        QWEN4_EXP_MTP_PY.write_text(src)

    # 2) rotary_embedding/{base,mrope}.py: disable nested torch.compile(dynamic=True) on TPU
    rbase = ROTARY_BASE_PY.read_text()
    if ROTARY_MARKER not in rbase:
        old_rb = (
            "            self._apply_rotary_emb_wrapped = torch.compile(\n"
            "                dynamic=True,\n"
            "                disable=_is_npu,\n"
            "            )(apply_rotary_emb)"
        )
        new_rb = (
            f"            {ROTARY_MARKER}\n"
            "            from sglang.srt.utils import is_tpu as _q38_is_tpu\n"
            "            self._apply_rotary_emb_wrapped = torch.compile(\n"
            "                dynamic=True,\n"
            "                disable=_is_npu or _q38_is_tpu(),\n"
            "            )(apply_rotary_emb)"
        )
        assert rbase.count(old_rb) == 1, "unexpected rotary_embedding/base.py layout"
        ROTARY_BASE_PY.write_text(rbase.replace(old_rb, new_rb, 1))

    rmrope = ROTARY_MROPE_PY.read_text()
    if ROTARY_MARKER not in rmrope:
        old_rm = "        self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(apply_rotary_emb)"
        new_rm = (
            f"        {ROTARY_MARKER}\n"
            "        from sglang.srt.utils import is_tpu as _q38_is_tpu\n"
            "        self._apply_rotary_emb_wrapped = (\n"
            "            apply_rotary_emb if _q38_is_tpu() else torch.compile(dynamic=True)(apply_rotary_emb)\n"
            "        )"
        )
        assert rmrope.count(old_rm) == 1, "unexpected rotary_embedding/mrope.py layout"
        ROTARY_MROPE_PY.write_text(rmrope.replace(old_rm, new_rm, 1))

    hc_src = HYPERCONN_PY.read_text()
    if HYPERCONN_MARKER not in hc_src:
        old_hc = (
            "        self._mix_compute = torch.compile(_mix_compute)\n"
            "        self._combine_compute = torch.compile(_combine_compute)"
        )
        new_hc = (
            f"        {HYPERCONN_MARKER}\n"
            "        from sglang.srt.utils import is_tpu as _q38_is_tpu\n"
            "        self._mix_compute = _mix_compute if _q38_is_tpu() else torch.compile(_mix_compute)\n"
            "        self._combine_compute = _combine_compute if _q38_is_tpu() else torch.compile(_combine_compute)"
        )
        assert hc_src.count(old_hc) == 1, "unexpected hyperconnection.py layout"
        HYPERCONN_PY.write_text(hc_src.replace(old_hc, new_hc, 1))

    # 3) hybrid_linear_attn_backend.py: static-shape int32 TPU scatter for PLE side states after MTP verify
    hattn = HYBRID_ATTN_PY.read_text()
    if PLE_VERIFY_SCATTER_MARKER not in hattn:
        old_ha = (
            "        for state, intermediate_state in state_pairs:\n"
            "            self._scatter_speculative_state_with_mask(\n"
            "                state,\n"
            "                intermediate_state,\n"
            "                state_indices_tensor,\n"
            "                last_correct_step_indices,\n"
            "                src_indices_raw,\n"
            "            )\n"
            "            if mamba_track_indices is not None:\n"
            "                self._scatter_speculative_state_with_mask(\n"
            "                    state,\n"
            "                    intermediate_state,\n"
            "                    mamba_track_indices,\n"
            "                    mamba_steps_to_track,\n"
            "                    src_indices_raw,\n"
            "                )"
        )
        new_ha = (
            f"        {PLE_VERIFY_SCATTER_MARKER}\n"
            "        from sglang.srt.utils import is_tpu as _q38_is_tpu\n"
            "        if _q38_is_tpu():\n"
            "            def _tpu_scatter_ple(dst_root, src_root, has_layer_dim, dst_idx_raw, step_idx_raw):\n"
            "                if dst_root is None or src_root is None or step_idx_raw.numel() == 0:\n"
            "                    return\n"
            "                dev = dst_root.device\n"
            "                b = step_idx_raw.shape[0]\n"
            "                dst_i = dst_idx_raw.to(device=dev, dtype=torch.int32)\n"
            "                stp_i = step_idx_raw.to(device=dev, dtype=torch.int32)\n"
            "                src_i = (\n"
            "                    torch.arange(b, device=dev, dtype=torch.int32)\n"
            "                    if src_indices_raw is None\n"
            "                    else src_indices_raw.to(device=dev, dtype=torch.int32)\n"
            "                )\n"
            "                s_dst = dst_root.shape[1] if has_layer_dim else dst_root.shape[0]\n"
            "                s_src = src_root.shape[1] if has_layer_dim else src_root.shape[0]\n"
            "                w_src = src_root.shape[2] if has_layer_dim else src_root.shape[1]\n"
            "                valid = (stp_i >= 0) & (stp_i < w_src) & (dst_i > 0) & (dst_i < s_dst) & (src_i >= 0) & (src_i < s_src)\n"
            "                safe_dst = torch.where(valid, dst_i, torch.zeros_like(dst_i))\n"
            "                safe_src = src_i.clamp(0, s_src - 1)\n"
            "                safe_stp = stp_i.clamp(0, w_src - 1)\n"
            "                if has_layer_dim:\n"
            "                    cand = src_root[:, safe_src, safe_stp]\n"
            "                    prev = dst_root[:, safe_dst]\n"
            "                    vshape = [1, b] + [1] * (cand.ndim - 2)\n"
            "                    dst_root[:, safe_dst] = torch.where(valid.view(*vshape), cand, prev)\n"
            "                else:\n"
            "                    cand = src_root[safe_src, safe_stp]\n"
            "                    prev = dst_root[safe_dst]\n"
            "                    vshape = [b] + [1] * (cand.ndim - 1)\n"
            "                    dst_root[safe_dst] = torch.where(valid.view(*vshape), cand, prev)\n"
            "            if short_conv_pool.conv_state is not None and short_conv_pool.intermediate_conv_state is not None:\n"
            "                _tpu_scatter_ple(short_conv_pool.conv_state, short_conv_pool.intermediate_conv_state, True, state_indices_tensor, last_correct_step_indices)\n"
            "                if mamba_track_indices is not None:\n"
            "                    _tpu_scatter_ple(short_conv_pool.conv_state, short_conv_pool.intermediate_conv_state, True, mamba_track_indices, mamba_steps_to_track)\n"
            "            if ngram_pool.context is not None and ngram_pool.intermediate_context is not None:\n"
            "                _tpu_scatter_ple(ngram_pool.context, ngram_pool.intermediate_context, False, state_indices_tensor, last_correct_step_indices)\n"
            "                if mamba_track_indices is not None:\n"
            "                    _tpu_scatter_ple(ngram_pool.context, ngram_pool.intermediate_context, False, mamba_track_indices, mamba_steps_to_track)\n"
            "            return\n"
            + old_ha
        )
        assert hattn.count(old_ha) == 1, "unexpected hybrid_linear_attn_backend.py PLE verify layout"
        HYBRID_ATTN_PY.write_text(hattn.replace(old_ha, new_ha, 1))

    # 4) sglang_tpu: HybridLinearKVPoolState PLE pool lifting + TpuMtpDraftWorker hc_count>1 support
    for tpu_root in TPU_ROOTS:
        hldc_py = tpu_root / "compilation/hybrid_linear_direct_compile.py"
        if hldc_py.exists():
            hsrc = hldc_py.read_text()
            if DIRECT_PLE_MARKER not in hsrc:
                old_cls = (
                    "class HybridLinearKVPoolState:\n"
                    '    """KVPool over the full-attention KV buffers followed by the linear-attention state."""\n\n'
                    "    def __init__(\n"
                    "        self, *, full_kv_pool: MHATokenToKVPool, mamba_pool: TpuMambaPool\n"
                    "    ) -> None:\n"
                    "        self._full_kv_pool = full_kv_pool\n"
                    "        self._mamba_pool = mamba_pool\n\n"
                    "    def get_state(self) -> list[torch.Tensor]:\n"
                    "        return [*self._full_kv_pool.get_state(), *self._mamba_pool.get_state()]\n\n"
                    "    def set_state(self, state: list[torch.Tensor]) -> None:\n"
                    "        num_full = len(self._full_kv_pool.get_state())\n"
                    "        self._full_kv_pool.set_state(state[:num_full])\n"
                    "        self._mamba_pool.set_state(state[num_full:])\n\n\n"
                    "def hybrid_linear_kv_pool_state(\n"
                    "    *, kv_pool: HybridLinearKVPool, req_to_token_pool: ReqToTokenPool\n"
                    ") -> HybridLinearKVPoolState:\n"
                    "    # The GDN layers read state through the request pool, so lift that one.\n"
                    "    assert isinstance(req_to_token_pool, HybridReqToTokenPool)\n"
                    "    mamba_pool = req_to_token_pool.mamba_pool\n"
                    "    assert isinstance(mamba_pool, TpuMambaPool), type(mamba_pool)\n"
                    "    full_kv_pool = kv_pool.full_kv_pool\n"
                    "    assert isinstance(full_kv_pool, MHATokenToKVPool), type(full_kv_pool)\n"
                    "    return HybridLinearKVPoolState(full_kv_pool=full_kv_pool, mamba_pool=mamba_pool)"
                )
                new_cls = (
                    f"{DIRECT_PLE_MARKER} v2\n"
                    "class HybridLinearKVPoolState:\n"
                    '    """KVPool over full-attention KV buffers, linear-attention state, and optional PLE side-state pools."""\n\n'
                    "    def __init__(\n"
                    "        self,\n"
                    "        *,\n"
                    "        full_kv_pool: MHATokenToKVPool,\n"
                    "        mamba_pool: TpuMambaPool,\n"
                    "        req_to_token_pool: Any = None,\n"
                    "        short_conv_pool: Any = None,\n"
                    "        ngram_pool: Any = None,\n"
                    "    ) -> None:\n"
                    "        self._full_kv_pool = full_kv_pool\n"
                    "        self._mamba_pool = mamba_pool\n"
                    "        self._req_to_token_pool = req_to_token_pool\n"
                    "        self._short_conv_pool = short_conv_pool\n"
                    "        self._ngram_pool = ngram_pool\n\n"
                    "    def _extra_slots(self) -> list[tuple[Any, str]]:\n"
                    "        slots: list[tuple[Any, str]] = []\n"
                    "        if self._req_to_token_pool is not None:\n"
                    '            if getattr(self._req_to_token_pool, "req_index_to_mamba_index_mapping", None) is not None:\n'
                    '                slots.append((self._req_to_token_pool, "req_index_to_mamba_index_mapping"))\n'
                    "        if self._short_conv_pool is not None:\n"
                    '            for attr in ("conv_state", "intermediate_conv_state"):\n'
                    "                if getattr(self._short_conv_pool, attr, None) is not None:\n"
                    "                    slots.append((self._short_conv_pool, attr))\n"
                    "        if self._ngram_pool is not None:\n"
                    '            for attr in ("context", "intermediate_context"):\n'
                    "                if getattr(self._ngram_pool, attr, None) is not None:\n"
                    "                    slots.append((self._ngram_pool, attr))\n"
                    "        return slots\n\n"
                    "    def get_state(self) -> list[torch.Tensor]:\n"
                    "        extra = [getattr(obj, attr) for obj, attr in self._extra_slots()]\n"
                    "        return [*self._full_kv_pool.get_state(), *self._mamba_pool.get_state(), *extra]\n\n"
                    "    def set_state(self, state: list[torch.Tensor]) -> None:\n"
                    "        num_full = len(self._full_kv_pool.get_state())\n"
                    "        num_mamba = len(self._mamba_pool.get_state())\n"
                    "        self._full_kv_pool.set_state(state[:num_full])\n"
                    "        self._mamba_pool.set_state(state[num_full : num_full + num_mamba])\n"
                    "        for (obj, attr), tensor in zip(self._extra_slots(), state[num_full + num_mamba :], strict=True):\n"
                    "            setattr(obj, attr, tensor)\n\n\n"
                    "def hybrid_linear_kv_pool_state(\n"
                    "    *, kv_pool: HybridLinearKVPool, req_to_token_pool: ReqToTokenPool\n"
                    ") -> HybridLinearKVPoolState:\n"
                    "    # The GDN layers read state through the request pool, so lift that one.\n"
                    "    assert isinstance(req_to_token_pool, HybridReqToTokenPool)\n"
                    "    mamba_pool = req_to_token_pool.mamba_pool\n"
                    "    assert isinstance(mamba_pool, TpuMambaPool), type(mamba_pool)\n"
                    "    full_kv_pool = kv_pool.full_kv_pool\n"
                    "    assert isinstance(full_kv_pool, MHATokenToKVPool), type(full_kv_pool)\n"
                    "    return HybridLinearKVPoolState(\n"
                    "        full_kv_pool=full_kv_pool,\n"
                    "        mamba_pool=mamba_pool,\n"
                    "        req_to_token_pool=req_to_token_pool,\n"
                    '        short_conv_pool=getattr(req_to_token_pool, "short_conv_pool", None),\n'
                    '        ngram_pool=getattr(req_to_token_pool, "ngram_pool", None),\n'
                    "    )"
                )
                assert hsrc.count(old_cls) == 1, f"unexpected {hldc_py} layout"
                hldc_py.write_text(hsrc.replace(old_cls, new_cls, 1))
            elif f"{DIRECT_PLE_MARKER} v2" not in hsrc:
                idx = hsrc.index(DIRECT_PLE_MARKER)
                end_idx = hsrc.index("\n\ndef has_linear_attention", idx)
                new_v2 = (
                    f"{DIRECT_PLE_MARKER} v2\n"
                    "class HybridLinearKVPoolState:\n"
                    '    """KVPool over full-attention KV buffers, linear-attention state, and optional PLE side-state pools."""\n\n'
                    "    def __init__(\n"
                    "        self,\n"
                    "        *,\n"
                    "        full_kv_pool: MHATokenToKVPool,\n"
                    "        mamba_pool: TpuMambaPool,\n"
                    "        req_to_token_pool: Any = None,\n"
                    "        short_conv_pool: Any = None,\n"
                    "        ngram_pool: Any = None,\n"
                    "    ) -> None:\n"
                    "        self._full_kv_pool = full_kv_pool\n"
                    "        self._mamba_pool = mamba_pool\n"
                    "        self._req_to_token_pool = req_to_token_pool\n"
                    "        self._short_conv_pool = short_conv_pool\n"
                    "        self._ngram_pool = ngram_pool\n\n"
                    "    def _extra_slots(self) -> list[tuple[Any, str]]:\n"
                    "        slots: list[tuple[Any, str]] = []\n"
                    "        if self._req_to_token_pool is not None:\n"
                    '            if getattr(self._req_to_token_pool, "req_index_to_mamba_index_mapping", None) is not None:\n'
                    '                slots.append((self._req_to_token_pool, "req_index_to_mamba_index_mapping"))\n'
                    "        if self._short_conv_pool is not None:\n"
                    '            for attr in ("conv_state", "intermediate_conv_state"):\n'
                    "                if getattr(self._short_conv_pool, attr, None) is not None:\n"
                    "                    slots.append((self._short_conv_pool, attr))\n"
                    "        if self._ngram_pool is not None:\n"
                    '            for attr in ("context", "intermediate_context"):\n'
                    "                if getattr(self._ngram_pool, attr, None) is not None:\n"
                    "                    slots.append((self._ngram_pool, attr))\n"
                    "        return slots\n\n"
                    "    def get_state(self) -> list[torch.Tensor]:\n"
                    "        extra = [getattr(obj, attr) for obj, attr in self._extra_slots()]\n"
                    "        return [*self._full_kv_pool.get_state(), *self._mamba_pool.get_state(), *extra]\n\n"
                    "    def set_state(self, state: list[torch.Tensor]) -> None:\n"
                    "        num_full = len(self._full_kv_pool.get_state())\n"
                    "        num_mamba = len(self._mamba_pool.get_state())\n"
                    "        self._full_kv_pool.set_state(state[:num_full])\n"
                    "        self._mamba_pool.set_state(state[num_full : num_full + num_mamba])\n"
                    "        for (obj, attr), tensor in zip(self._extra_slots(), state[num_full + num_mamba :], strict=True):\n"
                    "            setattr(obj, attr, tensor)\n\n\n"
                    "def hybrid_linear_kv_pool_state(\n"
                    "    *, kv_pool: HybridLinearKVPool, req_to_token_pool: ReqToTokenPool\n"
                    ") -> HybridLinearKVPoolState:\n"
                    "    # The GDN layers read state through the request pool, so lift that one.\n"
                    "    assert isinstance(req_to_token_pool, HybridReqToTokenPool)\n"
                    "    mamba_pool = req_to_token_pool.mamba_pool\n"
                    "    assert isinstance(mamba_pool, TpuMambaPool), type(mamba_pool)\n"
                    "    full_kv_pool = kv_pool.full_kv_pool\n"
                    "    assert isinstance(full_kv_pool, MHATokenToKVPool), type(full_kv_pool)\n"
                    "    return HybridLinearKVPoolState(\n"
                    "        full_kv_pool=full_kv_pool,\n"
                    "        mamba_pool=mamba_pool,\n"
                    "        req_to_token_pool=req_to_token_pool,\n"
                    '        short_conv_pool=getattr(req_to_token_pool, "short_conv_pool", None),\n'
                    '        ngram_pool=getattr(req_to_token_pool, "ngram_pool", None),\n'
                    "    )"
                )
                hldc_py.write_text(hsrc[:idx] + new_v2 + hsrc[end_idx:])

        mtp_dw_py = tpu_root / "experimental/speculative/mtp/mtp_draft_worker.py"
        if mtp_dw_py.exists():
            msrc = mtp_dw_py.read_text()
            if MTP_HC_MARKER not in msrc:
                old_carry = (
                    "        self.carry_size = int(draft_model.config.hidden_size)\n"
                    "        self.target_feature_size = self.carry_size"
                )
                new_carry = (
                    f"        {MTP_HC_MARKER}\n"
                    '        self.carry_size = int(getattr(draft_model, "input_embeds_hidden_size", draft_model.config.hidden_size))\n'
                    "        self.target_feature_size = self.carry_size"
                )
                old_logits = (
                    "    def first_draft_logits(self, carry: torch.Tensor) -> torch.Tensor:\n"
                    "        # The carry is the logits input the draft's final norm already produced.\n"
                    "        lm_head = self._draft_model.lm_head\n"
                    "        with torch.no_grad():\n"
                    "            return gather_and_crop_vocab(\n"
                    "                project_through_lm_head(carry, lm_head), lm_head\n"
                    "            )"
                )
                new_logits = (
                    "    def first_draft_logits(self, carry: torch.Tensor) -> torch.Tensor:\n"
                    "        # When hc_count > 1 (Qwen4-Exp), carry is the unmixed hc_hidden_states;\n"
                    "        # mix down to hidden_size via _mix_streams before projecting through lm_head.\n"
                    "        lm_head = self._draft_model.lm_head\n"
                    "        with torch.no_grad():\n"
                    "            logits_in = (\n"
                    "                self._draft_model.model._mix_streams(carry)\n"
                    "                if carry.shape[-1] != self._draft_model.config.hidden_size\n"
                    "                else carry\n"
                    "            )\n"
                    "            return gather_and_crop_vocab(\n"
                    "                project_through_lm_head(logits_in, lm_head), lm_head\n"
                    "            )"
                )
                assert msrc.count(old_carry) == 1, f"unexpected {mtp_dw_py} carry layout"
                assert msrc.count(old_logits) == 1, f"unexpected {mtp_dw_py} first_draft_logits layout"
                mtp_dw_py.write_text(msrc.replace(old_carry, new_carry, 1).replace(old_logits, new_logits, 1))

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
    # JAX 0.11+ removed pltpu.semaphore_signal, pltpu.semaphore_wait, and pltpu.DeviceIdType (moved to pl.*).
    # Also, fused_rs/fused_moe.py broadcasts non-SP routing via bfloat16 (exact only up to 256 experts); for 512-expert
    # Qwen3.8-Flash-Next, bfloat16(511) == 512.0 exceeds bfloat16 integer precision; float32 preserves all 512 expert indices.
    for tpu_root in TPU_ROOTS:
        gmm_rs_py = tpu_root / "moe/pallas/fused_rs/gmm_fused_rs_nodedup.py"
        if gmm_rs_py.exists():
            gsrc = gmm_rs_py.read_text()
            gnew = (
                gsrc.replace("pltpu.semaphore_signal", "pl.semaphore_signal")
                .replace("pltpu.semaphore_wait", "pl.semaphore_wait")
                .replace("pltpu.DeviceIdType", "pl.DeviceIdType")
            )
            if gnew != gsrc:
                gmm_rs_py.write_text(gnew)
        fmoe_rs_py = tpu_root / "moe/pallas/fused_rs/fused_moe.py"
        if fmoe_rs_py.exists():
            fsrc = fmoe_rs_py.read_text()
            old_bf16_route = "(topk_indices.astype(jnp.bfloat16), topk_weights.astype(jnp.bfloat16))"
            new_fp32_route = "(topk_indices.astype(jnp.float32), topk_weights.astype(jnp.float32))"
            if old_bf16_route in fsrc:
                assert fsrc.count(old_bf16_route) == 1, f"unexpected {fmoe_rs_py} routing layout"
                fmoe_rs_py.write_text(fsrc.replace(old_bf16_route, new_fp32_route, 1))
    return DESC_NVFP4


def apply_all() -> list[str]:
    active = [
        _patch_gated_norm(),
        _patch_qwen4_ple(),
        _patch_ngram_pool_int32(),
        _patch_qwen4_mtp_and_tpu_spec(),
        _patch_nvfp4(),
    ]
    # drop stale bytecode so every spawned rank re-reads the patched sources
    patched_files = [
        SERVER_ARGS_PY,
        GATED_NORM_PY,
        QWEN4_EXP_PY,
        QWEN4_EXP_MTP_PY,
        QWEN2_MOE_PY,
        MODELOPT_PY,
        PLE_POOL_PY,
        ROTARY_BASE_PY,
        ROTARY_MROPE_PY,
        HYPERCONN_PY,
        HYBRID_ATTN_PY,
    ]
    for tpu_root in TPU_ROOTS:
        patched_files.append(tpu_root / "compilation/hybrid_linear_direct_compile.py")
        patched_files.append(tpu_root / "experimental/speculative/mtp/mtp_draft_worker.py")
        patched_files.append(tpu_root / "moe/pallas/fused_rs/gmm_fused_rs_nodedup.py")
        patched_files.append(tpu_root / "moe/pallas/fused_rs/fused_moe.py")
    for py in patched_files:
        if py.exists():
            for pyc in py.parent.glob(f"__pycache__/{py.stem}.*.pyc"):
                pyc.unlink(missing_ok=True)
    return active


if __name__ == "__main__":
    for line in apply_all():
        print("q38 patch active:", line)
    sys.exit(0)
