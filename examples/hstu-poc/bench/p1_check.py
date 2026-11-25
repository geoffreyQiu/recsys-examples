"""P1 correctness: FBGEMM's C++ Hopper HSTU kernel with a cached history (queries shorter than keys, + targets),
against a pure-PyTorch reference; then the model end to end: recompute vs the cached path fed with the K/V
that recompute produced (must give the same candidate scores)."""
import argparse

import torch

from hstu_poc.model import CONFIGS, HSTU, Batch, reference_attention


def rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-6)).item()


def kernel_checks(cfg, dev):
    torch.manual_seed(0)
    H, D = cfg.heads, cfg.head_dim
    alpha = 1.0 / D ** 0.5
    model = HSTU(cfg, device=dev)
    cases = {
        "R: cached history, candidates only": Batch([700, 1500, 64], [64, 256, 17], device=dev),
        "W: prefix + new tokens + candidates": Batch([300, 1000, 0], [32, 0, 50], new=[100, 513, 200], device=dev),
        "W: fresh sequences (P=0)": Batch([0, 0], [16, 0], new=[900, 333], device=dev),
    }
    out = []
    for name, b in cases.items():
        k = torch.randn(b.total_k, H, D, device=dev, dtype=cfg.dtype) * 0.5
        v = torch.randn(b.total_k, H, D, device=dev, dtype=cfg.dtype) * 0.5
        q = torch.randn(b.total_q, H, D, device=dev, dtype=cfg.dtype) * 0.5
        got = model.attention(q, k, v, b.cu_q, b.cu_k, b.max_q, b.max_k, b.num_targets)
        ref = reference_attention(q, k, v, b, alpha, cfg.max_len, recompute=False)
        out.append((f"kernel {name}", rel_err(got, ref)))
        # the same users as a full (q = k) call: queries for every token
        qf = torch.randn(b.total_k, H, D, device=dev, dtype=cfg.dtype) * 0.5
        gotf = model.attention(qf, k, v, b.cu_full, b.cu_full, b.max_k, b.max_k, b.num_targets)
        reff = reference_attention(qf, k, v, b, alpha, cfg.max_len, recompute=True)
        out.append((f"kernel {name} (full q=k)", rel_err(gotf, reff)))
    return out


def model_check(cfg, dev):
    torch.manual_seed(1)
    model = HSTU(cfg, device=dev)
    hist, cand = [1000, 2500, 300, 4000], [64, 256, 1, 128]
    full = Batch([0] * 4, cand, new=hist, device=dev)          # recompute: every history token is "new"
    cached = Batch(hist, cand, device=dev)                       # R: history cached
    T = full.total_k
    items = torch.randint(0, cfg.n_items, (T,), device=dev)
    acts = torch.randint(0, cfg.n_actions, (T,), device=dev)
    x = model.embed(items, acts, full.pos_full)
    is_cand = torch.zeros(T, dtype=torch.bool, device=dev)
    is_cand[full.cand_in_full] = True
    x = torch.where(is_cand.unsqueeze(1), model.embed(items, None, full.pos_full), x)
    Ks = [torch.empty(T, cfg.heads, cfg.head_dim, device=dev, dtype=cfg.dtype) for _ in range(cfg.layers)]
    Vs = [torch.empty_like(Ks[0]) for _ in range(cfg.layers)]
    y_full = model.forward_full(x, full, kv_out=lambda i: (Ks[i], Vs[i]))
    # cached path: K/V buffers hold history rows from recompute; candidate rows get overwritten by the model
    Kc = [k.clone() for k in Ks]
    Vc = [v.clone() for v in Vs]
    for i in range(cfg.layers):
        Kc[i][cached.q_dst] = 0
        Vc[i][cached.q_dst] = 0
    y_cached = model.forward_cached(x[full.cand_in_full], cached, lambda i: (Kc[i], Vc[i]))
    return [("model: cached path vs recompute (candidate scores)", rel_err(y_cached, y_full)),
            ("model: candidate K/V rows rebuilt", max(rel_err(Kc[i][cached.q_dst], Ks[i][cached.q_dst])
                                                     for i in range(cfg.layers)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="S,M,D")
    a = ap.parse_args()
    dev = "cuda"
    ok = True
    for c in a.configs.split(","):
        cfg = CONFIGS[c]
        for name, err in kernel_checks(cfg, dev) + model_check(cfg, dev):
            good = err < 2e-2
            ok &= good
            print(f"[{cfg.name}] {'PASS' if good else 'FAIL'} rel.err {err:.2e}  {name}")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
