"""Where a request's cached history K/V comes from. Every source exposes the same hooks, called by the model
inside one CUDA graph per cell:
    setup(batch)      allocate / plan for this batch shape (outside the graph)
    kv(layer)         (K, V) views [>= total_k, heads, head_dim] of the local GPU buffers
    start()           at request start: begin moving every layer's history (post gets, issue copies, ...)
    before_attn(l)    right before layer l's attention: wait until layer l's history has arrived
    end()             after the last layer (join side streams, ...)
Local GPU buffer layout: [layer][K|V][total_k][heads][head_dim]; user u's history occupies rows
[k_off[u], k_off[u] + L_u) of each block, so one user's history in one layer is two contiguous ranges.
"""
import torch

from .model import Batch, Config


class LocalKV:
    """Local GPU-memory hit: the history is already in GPU memory (moves nothing)."""
    name = "local"

    def __init__(self, cfg: Config, device="cuda", local_buf=None):
        self.cfg, self.dev = cfg, torch.device(device)
        self.local_buf = local_buf      # carve buffers from here when given (a transport's registered window)

    def alloc(self, nbytes):
        if self.local_buf is not None:
            if nbytes > self.local_buf.numel():
                raise torch.cuda.OutOfMemoryError(f"local window {self.local_buf.numel() / 1e9:.1f} GB < {nbytes / 1e9:.1f} GB")
            return self.local_buf[:nbytes]
        return torch.empty(nbytes, dtype=torch.uint8, device=self.dev)

    def setup(self, b: Batch):
        cfg = self.cfg
        self.b = b
        self.blk = b.total_k * cfg.hd * 2                     # bytes of one K (or V) block
        self.buf = self.alloc(cfg.layers * 2 * self.blk)
        self.views = []
        for l in range(cfg.layers):
            k = self.buf[(2 * l) * self.blk:(2 * l + 1) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            v = self.buf[(2 * l + 1) * self.blk:(2 * l + 2) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            self.views.append((k, v))
        self.views[0][0].normal_(0, 0.5)
        for l in range(cfg.layers):        # any finite values will do for timing
            for t in self.views[l]:
                t.copy_(self.views[0][0])

    def kv(self, l):
        return self.views[l]

    def start(self):
        pass

    def before_attn(self, l):
        pass

    def end(self):
        pass

    def bytes_moved(self):
        return 0

    def release(self):
        self.views, self.buf = None, None


class H2DKV(LocalKV):
    """Local host memory -> GPU with copy engines: per user and layer, 2 cudaMemcpyAsync (K, V) on a copy stream
    issued at request start, one event per layer; the compute stream waits on layer l's event."""
    name = "h2d"

    def setup(self, b: Batch):
        super().setup(b)
        cfg = self.cfg
        tok = cfg.hd * 2
        # host store: per user a slot [layer][K|V][L_u] tokens (users at distinct offsets, like distinct users)
        self.slot = [cfg.layers * 2 * L * tok for L in b.hist]
        self.host = torch.empty(sum(self.slot), dtype=torch.uint8, pin_memory=True)
        self.host.view(cfg.dtype)[: min(self.host.numel() // 2, 1 << 24)].normal_(0, 0.5)
        self.copy_stream = torch.cuda.Stream(device=self.dev)
        self.events = [torch.cuda.Event() for _ in range(cfg.layers)]
        self.plan = []                     # per layer: list of (dst view, src view)
        base = 0
        offs = []
        for s in self.slot:
            offs.append(base)
            base += s
        for l in range(cfg.layers):
            ops = []
            for u, L in enumerate(b.hist):
                if L == 0:
                    continue
                for kv in range(2):
                    src = self.host[offs[u] + (2 * l + kv) * L * tok: offs[u] + (2 * l + kv + 1) * L * tok]
                    dst = self.buf[(2 * l + kv) * self.blk + b.k_off[u] * tok:(2 * l + kv) * self.blk + (b.k_off[u] + L) * tok]
                    ops.append((dst, src))
            self.plan.append(ops)

    def start(self):
        main = torch.cuda.current_stream()
        self.copy_stream.wait_stream(main)
        with torch.cuda.stream(self.copy_stream):
            for l, ops in enumerate(self.plan):
                for dst, src in ops:
                    dst.copy_(src, non_blocking=True)
                self.events[l].record(self.copy_stream)

    def before_attn(self, l):
        torch.cuda.current_stream().wait_event(self.events[l])

    def end(self):
        torch.cuda.current_stream().wait_stream(self.copy_stream)

    def bytes_moved(self):
        return sum(self.b.hist) * self.cfg.layers * self.cfg.kv_tok_bytes

    def release(self):
        super().release()
        self.host, self.plan = None, None


class RemoteKV(LocalKV):
    """History in a remote store's memory (one slot per user: [layer][K|V][slot_len][heads][head_dim]).
    Subclasses move it with a transport; this class builds the transfer plan. Users are distinct per request
    and rotate over `variants` user sets between requests (production-like scattered reads)."""
    variants = 8

    def __init__(self, cfg, local_buf, pool_bytes, device="cuda", seed=7):
        super().__init__(cfg, device, local_buf)
        self.pool_bytes = pool_bytes
        self.rng = __import__("numpy").random.default_rng(seed)

    def setup(self, b: Batch):
        cfg = self.cfg
        self.b = b
        self.blk = b.total_k * cfg.hd * 2
        self.buf = self.alloc(cfg.layers * 2 * self.blk)
        self.views = []
        for l in range(cfg.layers):
            k = self.buf[(2 * l) * self.blk:(2 * l + 1) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            v = self.buf[(2 * l + 1) * self.blk:(2 * l + 2) * self.blk].view(cfg.dtype).view(b.total_k, cfg.heads, cfg.head_dim)
            self.views.append((k, v))
        tok = cfg.hd * 2
        slot_len = max(b.hist)
        slot = cfg.layers * 2 * slot_len * tok
        n_users = self.pool_bytes // max(slot, 1)
        if n_users < b.B:
            raise torch.cuda.OutOfMemoryError(f"store pool holds {n_users} users of {slot / 2**20:.0f} MiB < B={b.B}")
        self.tables, self.checks = [], []
        for _ in range(self.variants):
            users = self.rng.choice(n_users, size=b.B, replace=False)
            segs, rows, lp = [], [], [0]
            for l in range(cfg.layers):
                for u, L in enumerate(b.hist):
                    if L == 0:
                        continue
                    for kv in range(2):
                        remote = int(users[u]) * slot + (2 * l + kv) * slot_len * tok
                        local = (2 * l + kv) * self.blk + b.k_off[u] * tok
                        segs.append([remote, local, L * tok, int(users[u])])
                        rows.append([local, L * tok, 0, remote // 8])     # store pattern is by word offset
                lp.append(len(segs))
            self.tables.append(torch.tensor(segs, dtype=torch.int64))
            self.checks.append(torch.tensor(rows, dtype=torch.int64))
        self.layer_ptr_cpu = torch.tensor(lp, dtype=torch.int32)
        self.cur = 0

    def bytes_moved(self):
        return sum(self.b.hist) * self.cfg.layers * self.cfg.kv_tok_bytes

    def check(self, ext, seed):
        """Data check of the last request's history rows against the store pattern (mismatched words)."""
        rows = self.checks[self.cur].to(self.dev)
        bad = torch.zeros(1, dtype=torch.int64, device=self.dev)
        ext.check_pattern_gpu(self.local_buf, rows, seed, bad)
        return int(bad.item())

    def release(self):
        self.views, self.buf, self.tables, self.checks = None, None, None, None


class GinKVSource(RemoteKV):
    """GIN (GDAKI or Proxy, by NCCL_GIN_TYPE), 5'a: a post kernel at request start issues every get and stores
    per-layer tickets; a 1-thread wait kernel before each layer's attention."""
    name = "gin"

    def __init__(self, cfg, gin, ext, peer, pool_bytes, n_ctx=2, device="cuda"):
        super().__init__(cfg, gin.local(), pool_bytes, device)
        self.g, self.E, self.peer, self.n_ctx = gin, ext, peer, n_ctx

    def setup(self, b):
        super().setup(b)
        self.segs = self.tables[0].to(self.dev)
        self.dev_tables = [t.to(self.dev) for t in self.tables]
        self.lp = self.layer_ptr_cpu.to(self.dev)
        self.tickets = torch.zeros(self.cfg.layers * self.n_ctx, 2, dtype=torch.int64, device=self.dev)

    def pre_replay(self, i):
        self.cur = i % self.variants
        self.segs.copy_(self.dev_tables[self.cur], non_blocking=True)

    def start(self):
        self.g.post_gets(self.segs, self.lp, self.tickets, self.peer, ctx0=0, n_ctx=self.n_ctx)

    def before_attn(self, l):
        self.g.wait_layer(self.tickets, l, 0, self.n_ctx)


class VerbsKVSource(RemoteKV):
    """CPU verbs, 5' for CPU verbs: the CPU posts every read before the request's graph launches; its worker
    thread polls the CQ and stores seq into flags[l] per completed layer; the graph's first kernel increments
    the device-side seq and each layer's attention waits for flags[l] >= seq (1-thread spin kernel)."""
    name = "verbs"

    def __init__(self, cfg, verbs, ext, pool_bytes, device="cuda"):
        super().__init__(cfg, verbs.local(), pool_bytes, device)
        self.v, self.E = verbs, ext
        self.seq_dev = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.seq = 0
        self.flags = verbs.flags_ptr()

    def setup(self, b):
        super().setup(b)
        self.v.wait_idle()
        self.seq = int(self.seq_dev.item())

    def pre_replay(self, i):
        self.cur = i % self.variants
        self.seq += 1
        self.v.post_reads(self.tables[self.cur], self.layer_ptr_cpu, self.seq)

    def start(self):
        self.E.seq_inc(self.seq_dev)

    def before_attn(self, l):
        self.E.wait_flags_seq(self.flags, l, self.seq_dev)

    def release(self):
        self.v.wait_idle()
        super().release()


class RawGdakiSource(RemoteKV):
    """Raw GDAKI (DOCA GPUNetIO, no NCCL), 5'a like GIN: a post kernel issues every get (spread over the QPs) and
    keeps one ticket per (layer, QP); a 1-thread wait kernel per layer. The store is a GPU-less verbs pool."""
    name = "raw"

    def __init__(self, cfg, raw, pool_bytes, device="cuda"):
        super().__init__(cfg, raw.local(), pool_bytes, device)
        self.r = raw

    def setup(self, b):
        super().setup(b)
        self.dev_tables = [t.to(self.dev) for t in self.tables]
        self.segs = self.dev_tables[0].clone()
        self.lp = self.layer_ptr_cpu.to(self.dev)
        self.tickets = torch.zeros(self.cfg.layers * self.r.nqp(), dtype=torch.int64, device=self.dev)

    def pre_replay(self, i):
        self.cur = i % self.variants
        self.segs.copy_(self.dev_tables[self.cur], non_blocking=True)

    def start(self):
        self.r.post_gets(self.segs, self.lp, self.tickets)

    def before_attn(self, l):
        self.r.wait_layer(self.tickets, l)
