"""CPU equivalence test: q38 TPU PLE patch vs upstream Qwen4-Exp PLE code.

Runs inside the SGLang-TorchTPU image (CPU only). Loads the ORIGINAL qwen4_exp.py as
module q4_orig and ORIGINAL+PATCH as q4_new, then drives both with the same fake pool /
forward batch for EXTEND (with padding, chunk continuation, EOS tokens), DECODE (with
padded rows) and TARGET_VERIFY, comparing n-gram ids, short-conv outputs, full PLE layer
math, and committed side states.
"""
import importlib.util
import os
import sys
import types

os.environ["SGLANG_ENABLE_QWEN4_PLE_FUSION"] = "0"
import torch
import torch.nn.functional as F

ORIG = "/usr/local/lib/python3.12/site-packages/sglang/srt/models/qwen4_exp.py"
PATCH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/q38/q38_qwen4_exp_tpu_ple.py"
src = open(ORIG).read()
marker = "# q38 TPU bring-up patch for Qwen4-Exp per-layer n-gram embeddings (PLE)."
if marker in src:  # pod file already patched: recover upstream
    head = src[: src.index(marker)]
    src = head[: head.rstrip("\n").rfind("\n# ====")] + "\n"
assert "class _PLEBatch(msgspec.Struct, frozen=True):" in src and marker not in src
patch = open(PATCH).read()


def load(name, code):
    mod = types.ModuleType(name)
    mod.__file__ = ORIG
    sys.modules[name] = mod
    exec(compile(code, ORIG, "exec"), mod.__dict__)
    return mod


q_old = load("q4_orig", src)
q_new = load("q4_new", src + patch)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

torch.random.default_generator.manual_seed(0)  # CPU only (do not touch the TPU)
EOS = 248044
V = 248320
C = 12
DIL = 3
K = 4
S = (K - 1) * DIL
NG = 3
H = NG - 1
HPN = 8
SLOTS = 16


class FakePool:
    def __init__(self):
        self.req_to_mamba = torch.randperm(SLOTS)[: SLOTS] + 1  # slots 1..SLOTS
        self.req_to_mamba = torch.cat([torch.tensor([0]), self.req_to_mamba])  # req 0 -> 0
        self.conv = torch.randn(SLOTS + 1, C, S)
        self.ctx = torch.randint(0, V, (SLOTS + 1, H))
        self.inter = torch.zeros(SLOTS + 1, 4, C, S)
        self.inter_ctx = torch.zeros(SLOTS + 1, 4, H, dtype=torch.long)
        self.ple_window_cache = None
        self.size = SLOTS

    def clone(self):
        p = FakePool.__new__(FakePool)
        for k, v in vars(self).items():
            setattr(p, k, v.clone() if isinstance(v, torch.Tensor) else v)
        return p

    def get_mamba_indices(self, req):
        return self.req_to_mamba[req.long()]

    def get_ngram_context(self, idx):
        return self.ctx.index_select(0, idx.long())

    def set_ngram_context(self, idx, ctx):
        self.ctx[idx.long()] = ctx.to(self.ctx.dtype)

    def set_ngram_intermediate_context(self, ctx, idx=None):
        if idx is None:
            self.inter_ctx[: ctx.shape[0], : ctx.shape[1]].copy_(ctx)
        else:
            self.inter_ctx[idx.long(), : ctx.shape[1]] = ctx

    def short_conv_layer_cache(self, layer_id):
        return self.conv

    def short_conv_layer_intermediate_cache(self, layer_id):
        return self.inter


_HASH_CONST = {}


def fake_ngram(mod):
    obj = mod.Qwen4ExpNGramEmbedding.__new__(mod.Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(obj)
    obj.config = types.SimpleNamespace(seed=1234)
    obj.ngram_size = NG
    obj.heads_per_ngram = HPN
    obj.ngram_heads = (NG - 1) * HPN
    obj.eos_token_id = EOS
    obj.enable_ple_fusion = False
    obj.unigram_vocab_size = V
    obj.ple_layer_index = 0
    obj.ngram_vocab_size_base = 20_000_000
    if not _HASH_CONST:
        mult_t = obj._build_layer_multipliers(NG)
        sizes, offsets, _ = obj._build_head_vocab_and_offsets()
        mult_py = q_new._q38_multipliers_py(obj)
        assert mult_t.tolist() == mult_py, (mult_t.tolist(), mult_py)
        _HASH_CONST.update(mult=mult_py, sizes=sizes, offsets=offsets)
        print("multipliers", mult_py, "head vocab", sizes[:3], "...")
    obj.layer_multipliers = torch.tensor(_HASH_CONST["mult"], dtype=torch.long)
    obj.ngram_heads_vocab_sizes = torch.tensor(_HASH_CONST["sizes"], dtype=torch.long)
    obj.ngram_heads_offsets = torch.tensor(_HASH_CONST["offsets"], dtype=torch.long)
    obj._q38_multipliers = list(_HASH_CONST["mult"])
    obj._q38_head_vocab_sizes = [int(v) for v in _HASH_CONST["sizes"]]
    obj._q38_head_offsets = [int(v) for v in _HASH_CONST["offsets"]]
    return obj


conv_w = torch.randn(C, 1, K)


def fake_ple_layer(mod):
    obj = mod.Qwen4ExpPLELayer.__new__(mod.Qwen4ExpPLELayer)
    torch.nn.Module.__init__(obj)
    obj.layer_id = 1
    obj.short_conv_state_len = S
    obj.short_conv_dilation = DIL
    obj.conv_kernel_size = K
    obj.conv_channels = C
    obj.conv1d = torch.nn.Conv1d(C, C, K, groups=C, bias=False, dilation=DIL)
    with torch.no_grad():
        obj.conv1d.weight.copy_(conv_w)
    return obj


def fb_ns(**kw):
    base = dict(
        tbo_parent_token_range=None, spec_algorithm=None, spec_info=None,
        _original_forward_mode=None, global_num_token_non_padded_cpu=None,
        extend_seq_lens_cpu=None, extend_seq_lens=None, extend_prefix_lens=None,
        out_cache_loc=None, mamba_track_indices=None, mamba_track_mask=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def run(mod, pool, fb, input_ids, x):
    mod.get_req_to_token_pool = lambda: pool
    batch = mod._prepare_ple_batch(input_ids, fb, ngram_size=NG, ngram_eos_token_id=EOS)
    ids = fake_ngram(mod).compute_ngram_ids(batch)
    conv = fake_ple_layer(mod)._short_conv(x[: batch.processed_tokens].clone(), fb, batch)
    mod._commit_ple_batch(batch, fb)
    return batch, ids, conv


def check(name, a, b, atol=0.0):
    if not a.dtype.is_floating_point and not b.dtype.is_floating_point:
        ok = a.shape == b.shape and torch.equal(a.to(torch.int64), b.to(torch.int64))
        nbad = int((a.to(torch.int64) != b.to(torch.int64)).sum()) if a.shape == b.shape else -1
        print(f"{'OK ' if ok else 'BAD'} {name}: exact int compare, mismatches={nbad}")
    else:
        ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=1e-4 if atol else 0)
        print(f"{'OK ' if ok else 'BAD'} {name}: max|d|={(a.float()-b.float()).abs().max().item():.3g}")
    if not ok:
        print("  old", a)
        print("  new", b)
    return ok


all_ok = True
# ---------------- EXTEND: req lens [5,1,7] (+1 padded req), T=16, req1 continues a chunk
lens = [5, 1, 7]
prefix = [0, 3, 0]
real = sum(lens)
T, B = 16, 4
tokens = torch.randint(0, V, (T,))
tokens[2] = EOS
tokens[9] = EOS
tokens[real:] = 0
req_pool = torch.tensor([3, 5, 9, 0])
out_cache = torch.cat([torch.arange(1, real + 1), torch.zeros(T - real, dtype=torch.long)])
x = torch.randn(T, C)
pool0 = FakePool()
# fresh requests (prefix 0): upstream resets their slots before the forward; the patch
# must not depend on that -> give the patched run garbage in those slots.
pool_old = pool0.clone()
pool_new = pool0.clone()
for r in range(3):
    if prefix[r] == 0:
        slot = pool0.req_to_mamba[req_pool[r]].item()
        pool_old.conv[slot] = 0
        pool_old.ctx[slot] = EOS
        pool_new.conv[slot] = torch.randn(C, S) * 100
        pool_new.ctx[slot] = 3
fb_old = fb_ns(forward_mode=ForwardMode.EXTEND, global_num_token_non_padded_cpu=real,
               extend_seq_lens_cpu=list(lens),
               extend_seq_lens=torch.tensor(lens), extend_prefix_lens=torch.tensor(prefix),
               req_pool_indices=req_pool[:3], out_cache_loc=out_cache[:real])
fb_new = fb_ns(forward_mode=ForwardMode.EXTEND, global_num_token_non_padded_cpu=real,
               extend_seq_lens_cpu=list(lens) + [0],
               extend_seq_lens=torch.tensor(lens + [0]), extend_prefix_lens=torch.tensor(prefix + [0]),
               req_pool_indices=req_pool, out_cache_loc=out_cache)
b_old, ids_old, conv_old = run(q_old, pool_old, fb_old, tokens[:real], x)
b_new, ids_new, conv_new = run(q_new, pool_new, fb_new, tokens, x)
valid = b_new.valid_tokens
print("extend valid rows:", int(valid.sum()), "of", T)
all_ok &= check("extend ngram ids", ids_old, ids_new[:real][valid[:real]])
all_ok &= check("extend short-conv", conv_old, conv_new[:real][valid[:real]], atol=1e-4)
for r in range(3):
    slot = pool0.req_to_mamba[req_pool[r]].item()
    all_ok &= check(f"extend commit ctx req{r}", pool_old.ctx[slot], pool_new.ctx[slot])
    all_ok &= check(f"extend commit conv req{r}", pool_old.conv[slot], pool_new.conv[slot], atol=1e-5)

# ---------------- DECODE: 6 real rows + 2 padded (out_cache_loc == 0)
Bd, real_d = 8, 6
tokens = torch.randint(0, V, (Bd,))
tokens[1] = EOS
req_pool = torch.tensor([3, 5, 9, 2, 4, 6, 0, 0])
out_cache = torch.cat([torch.arange(10, 10 + real_d), torch.zeros(Bd - real_d, dtype=torch.long)])
x = torch.randn(Bd, C)
pool0 = FakePool()
pool_old, pool_new = pool0.clone(), pool0.clone()
fb_old = fb_ns(forward_mode=ForwardMode.DECODE, global_num_token_non_padded_cpu=real_d,
               req_pool_indices=req_pool[:real_d], out_cache_loc=out_cache[:real_d])
fb_new = fb_ns(forward_mode=ForwardMode.DECODE, global_num_token_non_padded_cpu=real_d,
               req_pool_indices=req_pool, out_cache_loc=out_cache)
b_old, ids_old, conv_old = run(q_old, pool_old, fb_old, tokens[:real_d], x)
b_new, ids_new, conv_new = run(q_new, pool_new, fb_new, tokens, x)
all_ok &= check("decode ngram ids", ids_old, ids_new[:real_d])
all_ok &= check("decode short-conv", conv_old, conv_new[:real_d], atol=1e-4)
for r in range(real_d):
    slot = pool0.req_to_mamba[req_pool[r]].item()
    all_ok &= check(f"decode commit ctx req{r}", pool_old.ctx[slot], pool_new.ctx[slot])
    all_ok &= check(f"decode commit conv req{r}", pool_old.conv[slot], pool_new.conv[slot], atol=1e-5)

# ---------------- TARGET_VERIFY: 3 reqs x W=4
W, Bv = 4, 3
tokens = torch.randint(0, V, (Bv * W,))
tokens[5] = EOS
req_pool = torch.tensor([3, 5, 9])
out_cache = torch.arange(20, 20 + Bv * W)
x = torch.randn(Bv * W, C)
spec = types.SimpleNamespace(draft_token_num=W, topk=1)
pool0 = FakePool()
pool_old, pool_new = pool0.clone(), pool0.clone()
kw = dict(forward_mode=ForwardMode.TARGET_VERIFY, spec_info=spec, global_num_token_non_padded_cpu=Bv * W,
          extend_seq_lens=torch.full((Bv,), W), req_pool_indices=req_pool, out_cache_loc=out_cache)
b_old, ids_old, conv_old = run(q_old, pool_old, fb_ns(**kw), tokens, x)
b_new, ids_new, conv_new = run(q_new, pool_new, fb_ns(**kw), tokens, x)
all_ok &= check("verify ngram ids", ids_old, ids_new)
all_ok &= check("verify short-conv", conv_old, conv_new, atol=1e-4)
all_ok &= check("verify inter conv", pool_old.inter, pool_new.inter, atol=1e-5)
all_ok &= check("verify inter ctx", pool_old.inter_ctx, pool_new.inter_ctx)

# ---------------- bulk hash: 200k random windows, extreme ids, dense EOS
N = 200_000
ctx = torch.randint(0, V, (N, NG))
ctx[: N // 10] = V - 1
ctx[N // 10 : N // 5] = 0
eos_mask = torch.rand(N, NG) < 0.2
ctx = torch.where(eos_mask, torch.full_like(ctx, EOS), ctx)
q_old.get_req_to_token_pool = lambda: FakePool()
h_old = fake_ngram(q_old)._hash_contexts(ctx.clone())
h_new = q_new._q38_hash_contexts(fake_ngram(q_new), ctx.to(torch.int32))
all_ok &= check(f"bulk hash ({N} windows)", h_old, h_new)
print("hash dtype", h_new.dtype, "range", int(h_new.min()), int(h_new.max()))
print("ALL_OK" if all_ok else "SOME_FAILED")
