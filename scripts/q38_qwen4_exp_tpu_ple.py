

# ======================================================================================
# q38 TPU bring-up patch for Qwen4-Exp per-layer n-gram embeddings (PLE).
# Appended verbatim to sglang/srt/models/qwen4_exp.py by q38_patches.py (TPU image
# sglang-torchtpu-diffusion:v0.2.0). Every function below replaces the module-level
# function / method of the same name defined above; callers resolve them at call time.
#
# Why: TorchTPU compiles the decoder forward with Dynamo (fullgraph=True) at bucketed
# (padded) shapes and arms fail_on_recompile at serve time. The upstream PLE code
#   1. builds a msgspec.Struct (_PLEBatch) inside the traced forward -> Dynamo
#      "Unsupported object.__new__ user-defined class construction";
#   2. slices by host-side real-token counts (global_num_token_non_padded_cpu,
#      sum/max(extend_seq_lens_cpu)) -> one compiled program per prompt length;
#   3. calls CUDA/Triton fused kernels and torch.cummax;
#   4. hashes n-grams with int64 multiply / xor / remainder and int64 cumsum, which the
#      TPU XLA backend rejects ("Unsupported conversion from vmreg/vreg to U64",
#      windowing_util window_bounds check) -> compiler abort.
# This block keeps the math identical but
#   * _PLEBatch is a typing.NamedTuple (positional construction);
#   * processed_tokens == physical (padded) tokens; padded rows are masked with
#     valid_tokens and their state writes go to the reserved slot 0;
#   * EXTEND uses a flat token-major layout (n-gram windows, short-conv taps and state
#     commits are gathers keyed by req_indices / token_offsets / seq_starts) instead of a
#     (num_reqs, max_extend_len) padded tensor whose width came from a host list;
#   * fresh requests (extend_prefix_lens == 0) start from a zero short-conv state and an
#     EOS n-gram history inside the graph: TpuMambaPool.clear_slots does not reset the
#     PLE side-state pools registered on MambaPool (_slot_siblings);
#   * all integer math is int32: the n-gram hash  (sum_k^xor tok_k * M_k) mod V_h  is
#     evaluated exactly with base-2^12 limbs (every product of the checkpoint's
#     multipliers and token ids is < 2^63 by construction, so the xor is non-negative and
#     the int64 Python-style remainder equals the unsigned limb remainder);
#   * no fused CUDA/Triton kernels, no cross-layer window cache, no cummax / cumsum.
# ======================================================================================
from typing import NamedTuple as _Q38NamedTuple

_Q38_I32 = torch.int32


class _PLEBatch(_Q38NamedTuple):  # noqa: F811  (replaces the msgspec.Struct above)
    mode: ForwardMode
    use_decode_fast_path: bool
    physical_tokens: int
    processed_tokens: int
    lengths: torch.Tensor
    row_width: int
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: Optional[torch.Tensor]
    ngram_eos_token_id: Optional[int]
    flat_extend: bool
    seq_starts: Optional[torch.Tensor]
    has_initial_state: Optional[torch.Tensor]
    ngram_history: Optional[torch.Tensor]
    ngram_tokens: Optional[torch.Tensor]
    ngram_windows: Optional[torch.Tensor]


def _q38_ngram_get(pool, state_indices: torch.Tensor) -> torch.Tensor:
    ngram_pool = getattr(pool, "ngram_pool", None)
    if ngram_pool is not None and getattr(pool, "layer_transfer_counter", None) is None:
        return ngram_pool.context.index_select(0, state_indices).to(_Q38_I32)
    return pool.get_ngram_context(state_indices).to(_Q38_I32)


def _q38_ngram_set(pool, state_indices: torch.Tensor, context: torch.Tensor) -> None:
    ngram_pool = getattr(pool, "ngram_pool", None)
    if ngram_pool is not None:
        if ngram_pool.context is not None:
            ngram_pool.context[state_indices] = context.to(ngram_pool.context.dtype)
        return
    pool.set_ngram_context(state_indices, context)


def _prepare_ple_batch(  # noqa: F811
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    ngram_size: Optional[int],
    ngram_eos_token_id: Optional[int],
) -> Optional[_PLEBatch]:
    if forward_batch.tbo_parent_token_range is not None:
        raise NotImplementedError("Qwen4 PLE is not compatible with two-batch overlap")
    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and spec_algorithm.is_ngram():
        raise NotImplementedError("Qwen4 PLE does not support NGRAM speculation")
    if (
        forward_batch.spec_info is not None
        and getattr(forward_batch.spec_info, "topk", 1) != 1
    ):
        raise NotImplementedError("Qwen4 PLE speculative decoding supports only topk=1")

    # On TPU (especially DP>1 dp-attention), prepare_mlp_sync_batch materializes idle or
    # peer-decode ranks into the peer's active forward_batch.forward_mode (e.g., EXTEND with
    # padded input_ids [T] and req_pool_indices [B]) while recording _original_forward_mode=DECODE.
    # Every SPMD/DP rank must trace the active forward_batch.forward_mode, never _original_forward_mode.
    mode = forward_batch.forward_mode
    if mode.is_idle():
        return None
    if input_ids.dim() > 1:
        input_ids = input_ids.reshape(-1)
    physical_tokens = input_ids.shape[0]
    device = input_ids.device
    tokens = input_ids.to(_Q38_I32)
    positions = torch.arange(physical_tokens, device=device, dtype=_Q38_I32)
    pool = get_req_to_token_pool()

    flat_extend = False
    use_decode_fast_path = False
    seq_starts = None
    has_initial_state = None
    if mode.is_target_verify():
        assert forward_batch.spec_info is not None
        row_width = int(forward_batch.spec_info.draft_token_num)
        if row_width <= 0 or physical_tokens % row_width != 0:
            raise RuntimeError(
                "target verify rows must contain complete draft strides: "
                f"{physical_tokens=} {row_width=}"
            )
        sequence_count = physical_tokens // row_width
        if (
            forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_seq_lens is not None
        ):
            lengths = forward_batch.extend_seq_lens[:sequence_count].to(_Q38_I32)
        else:
            lengths = torch.full(
                (sequence_count,), row_width, dtype=_Q38_I32, device=device
            )
        req_indices = torch.div(positions, row_width, rounding_mode="floor")
        token_offsets = positions - req_indices * row_width
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)
    elif mode.is_decode():
        use_decode_fast_path = True
        sequence_count = physical_tokens
        row_width = 1
        lengths = torch.ones(physical_tokens, dtype=_Q38_I32, device=device)
        req_indices = positions
        token_offsets = torch.zeros_like(positions)
        valid_tokens = torch.ones(physical_tokens, dtype=torch.bool, device=device)
    else:
        if forward_batch.extend_seq_lens is None:
            raise RuntimeError(f"PLE requires sequence lengths in {mode!r}")
        flat_extend = True
        row_width = 0  # unused by the flat layout
        lengths = forward_batch.extend_seq_lens.to(_Q38_I32)
        sequence_count = lengths.shape[0]
        seq_index = torch.arange(sequence_count, device=device, dtype=_Q38_I32)
        # inclusive prefix sum without cumsum: [B, B] lower-triangular mask
        lower = (seq_index.unsqueeze(1) >= seq_index.unsqueeze(0)).to(_Q38_I32)
        seq_ends = (lower * lengths.unsqueeze(0)).sum(dim=1, dtype=_Q38_I32)
        seq_starts = seq_ends - lengths
        # Request owning each flat row = number of requests that end at or before it
        # (robust to zero-length padded requests); padded rows clamp to the last one
        # and are invalid because their offset >= that request's length.
        req_indices = (
            (seq_ends.unsqueeze(0) <= positions.unsqueeze(1))
            .to(_Q38_I32)
            .sum(dim=1, dtype=_Q38_I32)
            .clamp(max=sequence_count - 1)
        )
        token_offsets = positions - seq_starts.index_select(0, req_indices)
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)
        if forward_batch.extend_prefix_lens is not None:
            has_initial_state = forward_batch.extend_prefix_lens[:sequence_count] > 0
        else:
            has_initial_state = torch.zeros(
                sequence_count, dtype=torch.bool, device=device
            )

    state_indices = pool.get_mamba_indices(
        forward_batch.req_pool_indices[:sequence_count]
    ).to(_Q38_I32)
    # Padded or DP-idle dummy sequences use request slot 0 / out_cache_loc 0, which may
    # belong to a real request: route them to the state pools' reserved dummy slot 0.
    out_cache_loc = forward_batch.out_cache_loc
    valid = lengths.ne(0)
    if out_cache_loc is not None and mode.is_decode():
        valid = valid & out_cache_loc[:sequence_count].ne(0)
    elif out_cache_loc is not None and mode.is_target_verify():
        valid = valid & out_cache_loc[:physical_tokens].reshape(
            sequence_count, row_width
        ).ne(0).any(dim=1)
    elif out_cache_loc is not None and flat_extend and physical_tokens > 0:
        first_tok = seq_starts.clamp(min=0, max=physical_tokens - 1)
        valid = valid & out_cache_loc.index_select(0, first_tok).ne(0)
    state_indices = torch.where(valid, state_indices, torch.zeros_like(state_indices))

    ngram_context = None
    ngram_history = None
    ngram_tokens = None
    ngram_windows = None
    if ngram_size is not None:
        assert ngram_eos_token_id is not None
        history_width = ngram_size - 1
        history = _q38_ngram_get(pool, state_indices)
        if history.shape[1] != history_width:
            raise RuntimeError(
                "Qwen4 PLE N-gram cache has the wrong context width: "
                f"{history.shape[1]=} {ngram_size=}"
            )
        if flat_extend:
            history = torch.where(
                has_initial_state.unsqueeze(1),
                history,
                torch.full_like(history, ngram_eos_token_id),
            )
            cols = torch.arange(ngram_size, device=device, dtype=_Q38_I32)
            ext = token_offsets.unsqueeze(1) + cols.unsqueeze(0)  # [T, n]
            hist_vals = history[
                req_indices.unsqueeze(1), ext.clamp(max=history_width - 1)
            ]
            tok_vals = tokens[
                (positions.unsqueeze(1) + cols.unsqueeze(0) - history_width).clamp(
                    min=0, max=physical_tokens - 1
                )
            ]
            ngram_windows = torch.where(ext < history_width, hist_vals, tok_vals)
            ngram_history = history
            ngram_tokens = tokens
        elif use_decode_fast_path:
            ngram_context = torch.cat([history, tokens.unsqueeze(1)], dim=1)
        else:
            padded = torch.where(
                valid_tokens, tokens, torch.full_like(tokens, ngram_eos_token_id)
            ).reshape(sequence_count, row_width)
            ngram_context = torch.cat([history, padded], dim=1)

    return _PLEBatch(
        mode,
        use_decode_fast_path,
        physical_tokens,
        physical_tokens,
        lengths,
        row_width,
        req_indices,
        token_offsets,
        valid_tokens,
        state_indices,
        ngram_context,
        ngram_eos_token_id,
        flat_extend,
        seq_starts,
        has_initial_state,
        ngram_history,
        ngram_tokens,
        ngram_windows,
    )


def _commit_ple_batch(batch: Optional[_PLEBatch], forward_batch: ForwardBatch) -> None:  # noqa: F811
    if batch is None or not batch.processed_tokens:
        return
    pool = get_req_to_token_pool()

    if batch.flat_extend:
        history = batch.ngram_history
        if history is None:
            return
        history_width = history.shape[1]
        total = batch.physical_tokens
        cols = torch.arange(history_width, device=history.device, dtype=_Q38_I32)
        rows = torch.arange(history.shape[0], device=history.device, dtype=_Q38_I32)

        def _context_at(offsets: torch.Tensor) -> torch.Tensor:
            # n-gram history after consuming `offsets` chunk tokens of each request
            ext = offsets.to(_Q38_I32).unsqueeze(1) + cols.unsqueeze(0)  # [B, n-1]
            hist_vals = history[rows.unsqueeze(1), ext.clamp(max=history_width - 1)]
            tok_vals = batch.ngram_tokens[
                (batch.seq_starts.unsqueeze(1) + ext - history_width).clamp(
                    min=0, max=total - 1
                )
            ]
            return torch.where(ext < history_width, hist_vals, tok_vals)

        _q38_ngram_set(pool, batch.state_indices, _context_at(batch.lengths))
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, track_offsets = track
            _q38_ngram_set(pool, track_indices.to(_Q38_I32), _context_at(track_offsets))
        return

    if batch.ngram_context is None:
        return
    context = batch.ngram_context
    context_len = context.shape[1] - batch.row_width
    if batch.mode.is_target_verify():
        steps = [
            context[:, step : step + context_len]
            for step in range(1, batch.row_width + 1)
        ]
        step_contexts = torch.stack(steps, dim=1)  # [B, W, n-1]
        valid_steps = batch.valid_tokens.reshape(batch.lengths.shape[0], batch.row_width)
        dst_indices_raw = select_verify_intermediate_state_indices(
            None,
            forward_batch.req_pool_indices,
            batch.lengths.ne(0),
            pool.size,
        )
        pool.set_ngram_intermediate_context(
            torch.where(
                valid_steps.unsqueeze(-1),
                step_contexts,
                torch.full_like(step_contexts, batch.ngram_eos_token_id),
            ),
            dst_indices_raw,
        )
        return

    # decode: every history advances by exactly one column
    next_context = context[:, batch.row_width :]
    _q38_ngram_set(pool, batch.state_indices, next_context)
    track = _ple_track_targets(forward_batch, batch)
    if track is not None:
        track_indices, _ = track
        _q38_ngram_set(pool, track_indices.to(_Q38_I32), next_context)


# ---- exact int32 evaluation of the int64 n-gram hash -----------------------------------
_Q38_LIMB_BITS = 12
_Q38_LIMB_MASK = (1 << _Q38_LIMB_BITS) - 1


def _q38_mul_const_limbs(values: torch.Tensor, multiplier: int):
    """Base-2^12 limbs (6 x int32) of values * multiplier, values in [0, 2^24),
    multiplier a Python int in [0, 2^48). Every intermediate stays below 2^31."""
    m = [(multiplier >> (_Q38_LIMB_BITS * b)) & _Q38_LIMB_MASK for b in range(4)]
    v0 = values & _Q38_LIMB_MASK
    v1 = values >> _Q38_LIMB_BITS
    partial = [
        v0 * m[0],
        v0 * m[1] + v1 * m[0],
        v0 * m[2] + v1 * m[1],
        v0 * m[3] + v1 * m[2],
        v1 * m[3],
    ]
    limbs = []
    carry = None
    for p in partial:
        s = p if carry is None else p + carry
        limbs.append(s & _Q38_LIMB_MASK)
        carry = s >> _Q38_LIMB_BITS
    limbs.append(carry)
    return limbs


def _q38_mod_limbs(limbs, modulus: torch.Tensor) -> torch.Tensor:
    """(sum_j limbs[j] * 2^(12 j)) mod modulus, nibble-wise Horner in int32.
    modulus: int32 [H] with values < 2^26; returns int32 [N, H]."""
    rem = None
    top = len(limbs) - 1
    for j in range(top, -1, -1):
        nibbles = 4 if j == top else 3  # the top limb carries up to 16 bits
        for t in range(nibbles - 1, -1, -1):
            digit = ((limbs[j] >> (4 * t)) & 0xF).unsqueeze(-1)
            rem = digit % modulus if rem is None else (rem * 16 + digit) % modulus
    return rem


def _q38_multipliers_py(self) -> list:
    """Verbatim Python-int replica of Qwen4ExpNGramEmbedding._build_layer_multipliers."""
    seed = int(getattr(self.config, "seed", 1234))
    max_long = (1 << 63) - 1
    m_max = max_long // max(self.unigram_vocab_size, 1)
    half_bound = max(1, m_max // 2)
    base_seed = seed + self._PRIME_1 * self.ple_layer_index
    values = []
    for idx in range(self.ngram_size):
        x0 = (base_seed + self._SPLITMIX_GAMMA * (idx + 1)) & self._MASK64
        mixed = self._splitmix64(x0)
        values.append(int(2 * (mixed % half_bound) + 1))
    return values


_q38_orig_ngram_init = Qwen4ExpNGramEmbedding.__init__


def _q38_ngram_init(self, *args, **kwargs):
    _q38_orig_ngram_init(self, *args, **kwargs)
    self._q38_multipliers = _q38_multipliers_py(self)
    sizes, offsets, _ = self._build_head_vocab_and_offsets()
    self._q38_head_vocab_sizes = [int(v) for v in sizes]
    self._q38_head_offsets = [int(v) for v in offsets]
    dev = self.ngram_heads_vocab_sizes.device
    buf_dev = "cpu" if dev.type == "meta" else dev
    for ngram in range(2, self.ngram_size + 1):
        start = (ngram - 2) * self.heads_per_ngram
        end = start + self.heads_per_ngram
        self.register_buffer(
            f"_q38_mod_{ngram}",
            torch.tensor(self._q38_head_vocab_sizes[start:end], dtype=_Q38_I32, device=buf_dev),
            persistent=False,
        )
        self.register_buffer(
            f"_q38_off_{ngram}",
            torch.tensor(self._q38_head_offsets[start:end], dtype=_Q38_I32, device=buf_dev),
            persistent=False,
        )
    # Exactness preconditions of the int32 limb evaluation.
    assert max(self._q38_multipliers) < (1 << 48)
    assert self.unigram_vocab_size < (1 << 24)
    assert (self.unigram_vocab_size - 1) * max(self._q38_multipliers) < (1 << 63)
    assert max(self._q38_head_vocab_sizes) < (1 << 26)
    assert sum(self._q38_head_vocab_sizes) < (1 << 31)


Qwen4ExpNGramEmbedding.__init__ = _q38_ngram_init


def _q38_hash_contexts(self, contexts: torch.Tensor, *, decode_sized: bool = False) -> torch.Tensor:
    contexts = contexts.to(_Q38_I32)
    n = self.ngram_size
    eos = self.eos_token_id
    # Token k steps left of the window's last position, after the upstream
    # shift-right-ignoring-EOS (EOS once an EOS lies strictly between them).
    last = contexts[:, n - 1]
    shifted = [last]
    seen_eos = torch.zeros_like(last, dtype=torch.bool)
    for k in range(1, n):
        col = contexts[:, n - 1 - k]
        seen_eos = seen_eos | (col == eos)
        shifted.append(torch.where(seen_eos, torch.full_like(col, eos), col))
    products = [
        _q38_mul_const_limbs(shifted[k], self._q38_multipliers[k]) for k in range(n)
    ]
    blocks = []
    for ngram in range(2, n + 1):
        mix = products[0]
        for pos in range(1, ngram):
            mix = [torch.bitwise_xor(a, b) for a, b in zip(mix, products[pos])]
        modulus = getattr(self, f"_q38_mod_{ngram}", None)
        offsets = getattr(self, f"_q38_off_{ngram}", None)
        if modulus is None or offsets is None:
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            modulus = torch.tensor(self._q38_head_vocab_sizes[start:end], dtype=_Q38_I32, device=contexts.device)
            offsets = torch.tensor(self._q38_head_offsets[start:end], dtype=_Q38_I32, device=contexts.device)
        blocks.append(_q38_mod_limbs(mix, modulus) + offsets)
    return torch.cat(blocks, dim=-1)


def _q38_compute_ngram_ids(self, batch: _PLEBatch) -> torch.Tensor:
    if batch.flat_extend:
        contexts = batch.ngram_windows
    else:
        assert batch.ngram_context is not None
        if batch.use_decode_fast_path:
            contexts = batch.ngram_context
        else:
            width = batch.row_width
            windows = [
                batch.ngram_context[:, j : j + width] for j in range(self.ngram_size)
            ]
            contexts = torch.stack(windows, dim=-1).reshape(-1, self.ngram_size)
    return _q38_hash_contexts(self, contexts)


_q38_orig_load_weights = Qwen4ExpForConditionalGeneration.load_weights


def _q38_load_weights(self, weights, *args, **kwargs):
    result = _q38_orig_load_weights(self, weights, *args, **kwargs)
    # The int32 hash uses Python constants; prove they equal the loaded buffers.
    for module in self.modules():
        if not isinstance(module, Qwen4ExpNGramEmbedding):
            continue
        for buffer_name, expected in (
            ("layer_multipliers", module._q38_multipliers),
            ("ngram_heads_vocab_sizes", module._q38_head_vocab_sizes),
            ("ngram_heads_offsets", module._q38_head_offsets),
        ):
            buffer = getattr(module, buffer_name)
            if buffer.device.type == "meta":
                continue
            got = [int(v) for v in buffer.detach().to("cpu").tolist()]
            if got != expected:
                raise RuntimeError(
                    f"q38 PLE int32 hash constants differ from the loaded {buffer_name}: "
                    f"{got[:4]}... vs {expected[:4]}..."
                )
        target_dev = module.ngram_heads_vocab_sizes.device
        if target_dev.type != "meta":
            for ngram in range(2, module.ngram_size + 1):
                start = (ngram - 2) * module.heads_per_ngram
                end = start + module.heads_per_ngram
                module.register_buffer(
                    f"_q38_mod_{ngram}",
                    torch.tensor(module._q38_head_vocab_sizes[start:end], dtype=_Q38_I32, device=target_dev),
                    persistent=False,
                )
                module.register_buffer(
                    f"_q38_off_{ngram}",
                    torch.tensor(module._q38_head_offsets[start:end], dtype=_Q38_I32, device=target_dev),
                    persistent=False,
                )
    # Pre-cast RotaryEmbedding cos_sin_cache to bfloat16 before compilation so
    # _match_cos_sin_cache_dtype is a pure no-op inside direct_compile traces.
    for module in self.modules():
        csc = getattr(module, "cos_sin_cache", None)
        if isinstance(csc, torch.Tensor) and csc.device.type != "meta" and csc.dtype != torch.bfloat16:
            module.cos_sin_cache = csc.to(device=csc.device, dtype=torch.bfloat16)
    embed_tok = getattr(self.model, "embed_tokens", None)
    if embed_tok is not None and hasattr(embed_tok, "lift_rank_to_input"):
        w = getattr(embed_tok, "weight", None)
        if isinstance(w, torch.Tensor) and w.device.type != "meta":
            embed_tok.lift_rank_to_input()
    return result


Qwen4ExpForConditionalGeneration.load_weights = _q38_load_weights
Qwen4ExpForConditionalGeneration.decode_embeds_in_language_model = True

from sglang.srt.compilation.support_torch_compile import (  # noqa: E402
    _COMPILE_POLICY_ATTR as _Q38_COMPILE_POLICY_ATTR,
)

_q38_orig_cg_init = Qwen4ExpForConditionalGeneration.__init__


def _q38_cg_init(self, *args, **kwargs):
    _q38_orig_cg_init(self, *args, **kwargs)
    if getattr(self, "language_model_only", False):
        # Detach Qwen3VLDecoderTpuCompilePolicy from self.model so TpuGraphRunner
        # installs whole-model DecoderTpuCompilePolicy on Qwen4ExpForConditionalGeneration.forward
        # (compiles LogitsProcessor + lm_head in-graph; enables DP-attention & MTP direct_compile).
        setattr(self.model, _Q38_COMPILE_POLICY_ATTR, None)


Qwen4ExpForConditionalGeneration.__init__ = _q38_cg_init

_q38_orig_cg_forward = Qwen4ExpForConditionalGeneration.forward


@torch.no_grad()
def _q38_cg_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    get_embedding: bool = False,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
):
    if getattr(self, "language_model_only", False) and not get_embedding:
        if input_ids is None:
            input_ids = forward_batch.input_ids
        model_output = Qwen4ExpModel.forward(
            self.model,
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=None,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if isinstance(model_output, PPProxyTensors):
            return model_output
        hc_hidden_states = None
        if isinstance(model_output, tuple):
            hidden_states, hc_hidden_states = model_output
        else:
            hidden_states = model_output
        output = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=hc_hidden_states,
        )
        if (
            hc_hidden_states is not None
            and getattr(output, "hidden_states", None) is not None
            and output.hidden_states.shape[-1] != hc_hidden_states.shape[-1]
        ):
            output.hidden_states = hc_hidden_states
        return output
    return _q38_orig_cg_forward(
        self,
        input_ids=input_ids,
        positions=positions,
        forward_batch=forward_batch,
        get_embedding=get_embedding,
        pp_proxy_tensors=pp_proxy_tensors,
    )


Qwen4ExpForConditionalGeneration.forward = _q38_cg_forward


def _q38_short_conv_set(pool, layer_id: int, indices: torch.Tensor, val: torch.Tensor) -> None:
    sc_pool = getattr(pool, "short_conv_pool", None)
    if (
        sc_pool is not None
        and getattr(sc_pool, "conv_state", None) is not None
        and layer_id in getattr(sc_pool, "layer_map", {})
    ):
        lidx = sc_pool.layer_map[layer_id]
        sc_pool.conv_state[lidx, indices.to(_Q38_I32)] = val.to(dtype=sc_pool.conv_state.dtype)
        return
    conv_state = pool.short_conv_layer_cache(layer_id)
    conv_state[indices.to(_Q38_I32)] = val.to(dtype=conv_state.dtype)


def _q38_short_conv(
    self,
    x: torch.Tensor,
    forward_batch: ForwardBatch,
    batch: _PLEBatch,
) -> torch.Tensor:
    if x.shape[0] == 0:
        return x
    pool = get_req_to_token_pool()
    sc_pool = getattr(pool, "short_conv_pool", None)
    if (
        sc_pool is not None
        and getattr(sc_pool, "conv_state", None) is not None
        and self.layer_id in getattr(sc_pool, "layer_map", {})
    ):
        conv_state = sc_pool.conv_state[sc_pool.layer_map[self.layer_id]]
    else:
        conv_state = pool.short_conv_layer_cache(self.layer_id)  # [slots, C, S]
    state_len = self.short_conv_state_len
    dilation = self.short_conv_dilation
    kernel = self.conv_kernel_size
    channels = self.conv_channels
    weight = self.conv1d.weight[:, 0, :].to(torch.float32)  # [C, K]
    state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)  # [B, C, S]

    if batch.use_decode_fast_path:
        conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)  # [B, C, S + 1]
        acc = conv_input[:, :, 0].to(torch.float32) * weight[:, 0]
        for k in range(1, kernel):
            acc = acc + conv_input[:, :, k * dilation].to(torch.float32) * weight[:, k]
        next_state = conv_input[:, :, 1:]
        _q38_short_conv_set(pool, self.layer_id, batch.state_indices, next_state)
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, _ = track
            _q38_short_conv_set(pool, self.layer_id, track_indices, next_state)
        return F.silu(acc).to(x.dtype)

    if batch.flat_extend:
        total = x.shape[0]
        state = torch.where(
            batch.has_initial_state.view(-1, 1, 1), state, torch.zeros_like(state)
        )
        state_t = state.transpose(1, 2)  # [B, S, C]
        positions = torch.arange(total, device=x.device, dtype=_Q38_I32)
        offsets = batch.token_offsets
        acc = None
        for k in range(kernel):
            shift = state_len - k * dilation  # tap k reads token offset o - shift
            if shift <= 0:
                term = x
            else:
                from_state = state_t[
                    batch.req_indices, (offsets + k * dilation).clamp(max=state_len - 1)
                ]
                from_x = x.index_select(0, (positions - shift).clamp(min=0))
                term = torch.where((offsets < shift).unsqueeze(1), from_state, from_x)
            term = term.to(torch.float32) * weight[:, k]
            acc = term if acc is None else acc + term

        cols = torch.arange(state_len, device=x.device, dtype=_Q38_I32)
        rows = torch.arange(state.shape[0], device=x.device, dtype=_Q38_I32)

        def _state_at(lengths: torch.Tensor) -> torch.Tensor:
            # conv window after consuming `lengths` chunk tokens of each request
            ext = lengths.to(_Q38_I32).unsqueeze(1) + cols.unsqueeze(0)  # [B, S]
            from_state = state_t[rows.unsqueeze(1), ext.clamp(max=state_len - 1)]
            flat = (batch.seq_starts.unsqueeze(1) + ext - state_len).clamp(
                min=0, max=total - 1
            )
            from_x = x[flat]  # [B, S, C]
            window = torch.where((ext < state_len).unsqueeze(-1), from_state, from_x)
            return window.transpose(1, 2).to(dtype=conv_state.dtype)  # [B, C, S]

        _q38_short_conv_set(pool, self.layer_id, batch.state_indices, _state_at(batch.lengths))
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, track_offsets = track
            _q38_short_conv_set(pool, self.layer_id, track_indices, _state_at(track_offsets))
        return F.silu(acc).to(x.dtype)

    # target verify: fixed width W, no committed-state write (accepted steps are
    # selected from the intermediate cache afterwards)
    seq_count = batch.lengths.shape[0]
    width = batch.row_width
    seq = x.reshape(seq_count, width, channels).transpose(1, 2)  # [B, C, W]
    conv_input = torch.cat([state, seq], dim=-1)  # [B, C, S + W]
    acc = None
    for k in range(kernel):
        term = conv_input[:, :, k * dilation : k * dilation + width].to(torch.float32)
        term = term * weight[:, k].view(1, -1, 1)
        acc = term if acc is None else acc + term
    conv_output = acc.transpose(1, 2).reshape(seq_count * width, channels)
    dst_indices_raw = select_verify_intermediate_state_indices(
        None,
        forward_batch.req_pool_indices,
        batch.lengths.ne(0),
        pool.size,
    )
    dst_rows = (
        torch.arange(seq_count, device=x.device, dtype=_Q38_I32)
        if dst_indices_raw is None
        else dst_indices_raw.to(_Q38_I32)
    )
    if (
        sc_pool is not None
        and getattr(sc_pool, "intermediate_conv_state", None) is not None
        and self.layer_id in getattr(sc_pool, "layer_map", {})
    ):
        ic = sc_pool.intermediate_conv_state
        lidx = sc_pool.layer_map[self.layer_id]
        if state_len:
            intermediate_state = torch.stack(
                [conv_input[:, :, t : t + state_len] for t in range(1, width + 1)],
                dim=1,
            )  # [B, W, C, S]
        else:
            intermediate_state = x.new_empty((seq_count, width, channels, 0))
        valid_steps = batch.valid_tokens.reshape(seq_count, width, 1, 1)
        intermediate_state = torch.where(
            valid_steps, intermediate_state, torch.zeros_like(intermediate_state)
        ).to(dtype=ic.dtype)
        if width == ic.shape[2]:
            ic[lidx, dst_rows] = intermediate_state
        else:
            wcols = torch.arange(width, device=x.device, dtype=_Q38_I32)
            ic[lidx, dst_rows.unsqueeze(1), wcols.unsqueeze(0)] = intermediate_state
    else:
        intermediate_cache = pool.short_conv_layer_intermediate_cache(self.layer_id)
        if intermediate_cache is not None:
            if state_len:
                intermediate_state = torch.stack(
                    [conv_input[:, :, t : t + state_len] for t in range(1, width + 1)],
                    dim=1,
                )  # [B, W, C, S]
            else:
                intermediate_state = x.new_empty((seq_count, width, channels, 0))
            valid_steps = batch.valid_tokens.reshape(seq_count, width, 1, 1)
            intermediate_state = torch.where(
                valid_steps, intermediate_state, torch.zeros_like(intermediate_state)
            ).to(dtype=intermediate_cache.dtype)
            if width == intermediate_cache.shape[1]:
                intermediate_cache[dst_rows] = intermediate_state
            else:
                wcols = torch.arange(width, device=x.device, dtype=_Q38_I32)
                intermediate_cache[dst_rows.unsqueeze(1), wcols.unsqueeze(0)] = intermediate_state
    return F.silu(conv_output).to(x.dtype)


def _q38_ple_layer_forward(
    self,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    batch: _PLEBatch,
) -> torch.Tensor:
    hidden_states = hidden_states[: batch.processed_tokens]
    embeddings = self.ple_embedding(batch, forward_batch)
    key, _ = self.key_proj(embeddings)
    value, _ = self.value_proj(embeddings)
    token_count = hidden_states.shape[0]
    hidden_size = self.hidden_size
    hc_count = self.hc_count
    if hidden_states.shape[-1] != hc_count * hidden_size:
        raise RuntimeError(
            "PLE hidden size does not match its hyper-connection layout: "
            f"expected {hc_count * hidden_size}, got {hidden_states.shape[-1]}"
        )
    key = key.reshape(token_count, hc_count, hidden_size)
    query = hidden_states.reshape(token_count, hc_count, hidden_size)
    key_normed = self._apply_ple_norm(self.norm_key, key)
    query_normed = self._apply_ple_norm(self.norm_query, query)
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
    gate = gate / math.sqrt(hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gate = torch.sigmoid(gate)
    gated_value = gate * value.unsqueeze(-2)
    gated_value_normed = self._apply_ple_norm(self.norm_conv, gated_value)
    gated_value = gated_value.flatten(-2)
    gated_value_normed = gated_value_normed.flatten(-2)
    conv_output = self._short_conv(gated_value_normed, forward_batch, batch)
    output = gated_value + conv_output
    if not batch.use_decode_fast_path:
        output = torch.where(
            batch.valid_tokens.unsqueeze(-1), output, torch.zeros_like(output)
        )
    return _pad_token_rows(output, batch.physical_tokens)


Qwen4ExpNGramEmbedding._hash_contexts = _q38_hash_contexts
Qwen4ExpNGramEmbedding.compute_ngram_ids = _q38_compute_ngram_ids
Qwen4ExpPLELayer._short_conv = _q38_short_conv
Qwen4ExpPLELayer.forward = _q38_ple_layer_forward
# ================================ end q38 TPU patch ===================================
