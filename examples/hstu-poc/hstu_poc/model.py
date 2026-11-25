"""HSTU inference model for the POC: the math of examples/hstu/modules/paged_hstu_infer_layer.py, trimmed,
random weights, BF16, no KV manager. Study doc §14.

Layer:  x -> LN -> silu(x W_uvqk + b) -> split u|v|q|k -> attention -> u * LN(attn) -> W_proj + b + x
Attention: FBGEMM's C++ Hopper kernel (hstu_attn_varlen_func), silu(alpha q.k) / scaling_seqlen, no softmax,
causal with targets: history tokens attend causally, each target (candidate) attends to all history + itself.

KV layout used everywhere (local GPU buffers, the remote store, host buffers): per layer a K block and a V block,
each [tokens, heads, head_dim] contiguous BF16. One user's history in one layer = 2 contiguous byte ranges.
"""
import dataclasses
import math

import torch
import torch.nn.functional as F

from hstu import hstu_attn_varlen_func


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    layers: int
    hidden: int
    heads: int
    head_dim: int
    max_len: int = 32768          # longest history supported; also the attention scaling_seqlen
    emb_dim: int = 128
    n_items: int = 1 << 20
    n_actions: int = 256
    pos_buckets: int = 1024
    mlp: tuple = (512, 8)
    dtype: torch.dtype = torch.bfloat16

    @property
    def hd(self):
        return self.heads * self.head_dim

    @property
    def kv_tok_bytes(self):         # K + V bytes per token per layer
        return 2 * self.hd * torch.finfo(self.dtype).bits // 8


CONFIGS = {
    "S": Config("S", layers=3, hidden=512, heads=4, head_dim=128),
    "M": Config("M", layers=8, hidden=1024, heads=4, head_dim=256),
    "D": Config("D", layers=16, hidden=1024, heads=8, head_dim=128),   # deep: parked for now
    "L": Config("L", layers=8, hidden=2048, heads=8, head_dim=256),    # large: M at 2x width (4x dense, 2x KV)
}


class Batch:
    """Jagged batch metadata for B users. hist[u] = cached/history tokens L_u, cand[u] = C_u, new[u] = S_u
    (W workload: new history tokens computed now). All offsets are int32 CUDA tensors."""

    def __init__(self, hist, cand, new=None, device="cuda"):
        self.B = len(hist)
        self.hist = list(map(int, hist))
        self.cand = list(map(int, cand))
        self.new = list(map(int, new)) if new is not None else [0] * self.B
        dev = torch.device(device)
        i32 = dict(dtype=torch.int32, device=dev)
        self.q_len = [s + c for s, c in zip(self.new, self.cand)]              # tokens computed now
        self.k_len = [l + q for l, q in zip(self.hist, self.q_len)]            # keys per user
        self.cu_q = torch.tensor([0] + list(_cumsum(self.q_len)), **i32)
        self.cu_k = torch.tensor([0] + list(_cumsum(self.k_len)), **i32)
        self.cu_full = self.cu_k                                               # recompute: q = k = all tokens
        self.num_targets = torch.tensor(self.cand, **i32)
        self.max_q, self.max_k = max(self.q_len), max(self.k_len)
        self.total_q, self.total_k = sum(self.q_len), sum(self.k_len)
        self.k_off = [int(x) for x in self.cu_k[:-1].tolist()]
        # rows of the K/V buffers that receive the tokens computed now (after each user's history)
        dst = [torch.arange(o + l, o + k, dtype=torch.int64) for o, l, k in zip(self.k_off, self.hist, self.k_len)]
        self.q_dst = torch.cat(dst).to(dev)
        # positions of candidates inside the "computed now" tokens and inside the full sequence
        self.cand_in_q = torch.cat([torch.arange(o + s, o + s + c, dtype=torch.int64)
                                    for o, s, c in zip(self.cu_q[:-1].tolist(), self.new, self.cand)]).to(dev)
        self.cand_in_full = torch.cat([torch.arange(o + l + s, o + k, dtype=torch.int64)
                                       for o, l, s, k in zip(self.k_off, self.hist, self.new, self.k_len)]).to(dev)
        self.pos_q = torch.cat([torch.arange(l, k, dtype=torch.int64) for l, k in zip(self.hist, self.k_len)]).to(dev)
        self.pos_full = torch.cat([torch.arange(0, k, dtype=torch.int64) for k in self.k_len]).to(dev)


def _cumsum(xs):
    s = 0
    for x in xs:
        s += x
        yield s


def attn_flops(batch: Batch, cfg: Config, recompute: bool) -> float:
    """Attention FLOPs (q.k and p.v, 2 FLOPs per MAC) for one layer."""
    f = 0.0
    for l, s, c in zip(batch.hist, batch.new, batch.cand):
        if recompute:
            n = l + s
            pairs = n * (n + 1) / 2 + c * (n + 1)
        else:
            pairs = s * l + s * (s + 1) / 2 + c * (l + s + 1)
        f += 4 * cfg.hd * pairs
    return f


def linear_flops(tokens: int, cfg: Config) -> float:
    return 2 * tokens * cfg.hidden * 4 * cfg.hd + 2 * tokens * cfg.hd * cfg.hidden


class HSTU(torch.nn.Module):
    def __init__(self, cfg: Config, device="cuda", seed=0):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator(device="cpu").manual_seed(seed)
        dt, dev = cfg.dtype, torch.device(device)

        def rnd(*shape, scale):
            return (torch.randn(*shape, generator=g) * scale).to(dt).to(dev)

        h, hd = cfg.hidden, cfg.hd
        self.item_emb = rnd(cfg.n_items, cfg.emb_dim, scale=0.05)
        self.action_emb = rnd(cfg.n_actions, cfg.emb_dim, scale=0.05)
        self.w_in = rnd(cfg.emb_dim, h, scale=1 / math.sqrt(cfg.emb_dim))
        self.pos_emb = rnd(cfg.pos_buckets, h, scale=0.02)
        self.layers = []
        for _ in range(cfg.layers):
            self.layers.append(dict(
                ln_w=torch.ones(h, dtype=dt, device=dev), ln_b=torch.zeros(h, dtype=dt, device=dev),
                w_uvqk=rnd(h, 4 * hd, scale=1 / math.sqrt(h)), b_uvqk=rnd(4 * hd, scale=0.02),
                lno_w=torch.ones(hd, dtype=dt, device=dev), lno_b=torch.zeros(hd, dtype=dt, device=dev),
                w_proj=rnd(hd, h, scale=1 / math.sqrt(hd)), b_proj=rnd(h, scale=0.02),
            ))
        self.out_ln_w, self.out_ln_b = torch.ones(h, dtype=dt, device=dev), torch.zeros(h, dtype=dt, device=dev)
        m1, m2 = cfg.mlp
        self.mlp_w1, self.mlp_b1 = rnd(h, m1, scale=1 / math.sqrt(h)), rnd(m1, scale=0.02)
        self.mlp_w2, self.mlp_b2 = rnd(m1, m2, scale=1 / math.sqrt(m1)), rnd(m2, scale=0.02)
        self.alpha = 1.0 / math.sqrt(cfg.head_dim)

    # ------------------------------------------------------------------ pieces
    def bucket(self, pos):
        # log-ish bucketing like HSTU's relative position buckets, clamped to the table
        return torch.clamp((torch.log2(pos.float() + 1) * 64).long(), max=self.cfg.pos_buckets - 1)

    def embed(self, item_ids, action_ids, pos):
        """item_ids/action_ids: [tokens] int64 (action_ids may be None for candidates)."""
        e = self.item_emb[item_ids]
        if action_ids is not None:
            e = e + self.action_emb[action_ids]
        x = e @ self.w_in + self.pos_emb[self.bucket(pos)]
        return F.layer_norm(x, (self.cfg.hidden,))

    def uvqk(self, i, x):
        p = self.layers[i]
        xn = F.layer_norm(x, (self.cfg.hidden,), p["ln_w"], p["ln_b"])
        m = torch.addmm(p["b_uvqk"], xn, p["w_uvqk"])
        F.silu(m, inplace=True)
        hd, H, D = self.cfg.hd, self.cfg.heads, self.cfg.head_dim
        u, v, q, k = torch.split(m, [hd, hd, hd, hd], dim=-1)
        return u, v.view(-1, H, D), q.view(-1, H, D), k.view(-1, H, D)

    def finish(self, i, attn, u, x):
        p = self.layers[i]
        y = u * F.layer_norm(attn.view(-1, self.cfg.hd), (self.cfg.hd,), p["lno_w"], p["lno_b"])
        out = torch.addmm(x, y, p["w_proj"])
        out.add_(p["b_proj"])
        return out

    def attention(self, q, k, v, cu_q, cu_k, max_q, max_k, num_targets):
        return hstu_attn_varlen_func(q, k, v, cu_q, cu_k, None, None, max_q, max_k, self.cfg.max_len, None,
                                     num_targets, 1, (-1, 0), self.alpha)

    def head(self, x_cand):
        y = F.layer_norm(x_cand, (self.cfg.hidden,), self.out_ln_w, self.out_ln_b)
        y = F.relu(torch.addmm(self.mlp_b1, y, self.mlp_w1))
        return torch.sigmoid(torch.addmm(self.mlp_b2, y, self.mlp_w2))

    # ------------------------------------------------------------------ full recompute (no cache)
    def forward_full(self, x, b: Batch, kv_out=None):
        """x: [total_k, hidden] embedded tokens of every user (history, new, candidates). Optionally writes
        every layer's K/V for all tokens into kv_out(layer) -> (K [total_k,H,D], V [total_k,H,D])."""
        for i in range(self.cfg.layers):
            u, v, q, k = self.uvqk(i, x)
            if kv_out is not None:
                K, V = kv_out(i)
                K.copy_(k)
                V.copy_(v)
            a = self.attention(q, k, v, b.cu_full, b.cu_full, b.max_k, b.max_k, b.num_targets)
            x = self.finish(i, a, u, x)
        return self.head(x[b.cand_in_full])

    # ------------------------------------------------------------------ cached history (R, W)
    def forward_cached(self, xq, b: Batch, kv, before_attn=None, after_kv=None, attn_fn=None):
        """xq: [total_q, hidden] embedded tokens computed now (W's new tokens + candidates).
        kv(layer) -> (K, V) buffers [>= total_k, H, D] whose rows [k_off[u], k_off[u] + L_u) hold user u's
        cached history; this function writes the new tokens' K/V after it, then runs one attention call
        (queries shorter than keys, causal + targets).
        before_attn(layer): hook right before layer l's attention (per-layer wait for fetched KV).
        after_kv(layer, k, v): hook once layer l's new K/V exist (W: pack + put)."""
        x = xq
        for i in range(self.cfg.layers):
            u, v, q, k = self.uvqk(i, x)
            K, V = kv(i)
            K.index_copy_(0, b.q_dst, k)
            V.index_copy_(0, b.q_dst, v)
            if after_kv is not None:
                after_kv(i, k, v)
            if attn_fn is not None:       # P4 variants: the source owns the wait and the attention launch
                a = attn_fn(i, q, K, V, b, self)
            else:
                if before_attn is not None:
                    before_attn(i)
                a = self.attention(q, K, V, b.cu_q, b.cu_k, b.max_q, b.max_k, b.num_targets)
            x = self.finish(i, a, u, x)
        return self.head(x[b.cand_in_q])


def reference_attention(q, k, v, b: Batch, alpha, scaling, recompute: bool):
    """Pure-PyTorch reference of the cached / full HSTU attention mask (repo's _get_delta_valid_attn_mask)."""
    outs = []
    cq = b.cu_full if recompute else b.cu_q
    for u in range(b.B):
        q0, q1 = int(cq[u]), int(cq[u + 1])
        k0, k1 = int(b.cu_k[u]), int(b.cu_k[u + 1])
        qu, ku, vu = q[q0:q1].float(), k[k0:k1].float(), v[k0:k1].float()
        n, nq, T = k1 - k0, q1 - q0, b.cand[u]
        rows = torch.arange(n - nq, n, device=q.device).view(-1, 1)
        cols = torch.arange(0, n, device=q.device).view(1, -1)
        rc, cc = rows.clamp(max=n - T), cols.clamp(max=n - T)
        mask = (rows == cols) | (rc - cc > 0)
        s = F.silu(torch.einsum("xhd,yhd->hxy", qu, ku) * alpha) / scaling
        s = s * mask.unsqueeze(0)
        outs.append(torch.einsum("hxy,yhd->xhd", s, vu))
    return torch.cat(outs).to(q.dtype)
