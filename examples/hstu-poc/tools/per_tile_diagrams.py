#!/usr/bin/env python3
"""Sequence diagrams of the per-tile fetch, for docs/per-tile-walkthrough.md (schematic, not measured):
  per-tile-verbs.png  CPU verbs v2: split posting (HPOC_SPLIT_POST=1), relay v2 (HPOC_RELAY=2)
  per-tile-raw.png    raw GDAKI v3: post kernel in its own graph (HPOC_RAW_POST=2, HPOC_RAW_PRELAUNCH=1)
One column per thing that runs code, time flowing down (not to scale), numbered boxes in step order with each step's
call or kernel, key lines and file:line in examples/hstu-poc. The line numbers are looked up in the source when the
figure is drawn (patterns as in doc_anchors.py: "pattern" or "after>>pattern"), and the drawing is checked for text
that leaves its box or overlaps a box, a label or an arrow. Palette and look of fetch_diagrams.py.
    tools/per_tile_diagrams.py docs/figs     write both PNGs (needs matplotlib: PyTorch container)
    tools/per_tile_diagrams.py --refs        print every reference with its source line (no matplotlib needed)"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"grid.py": "bench/grid.py", "kvsource.py": "hstu_poc/kvsource.py", "model.py": "hstu_poc/model.py",
         "verbs_kv.cpp": "csrc/verbs_kv.cpp", "kvutil.cu": "csrc/kvutil.cu", "raw_gdaki.cu": "csrc/raw_gdaki.cu",
         "bindings.cpp": "csrc/bindings.cpp",
         "mainloop_fwd_sm90_tma_gmma_ws.hpp": "attn/hstu_hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp"}
MAINLOOP = "mainloop_fwd_sm90_tma_gmma_ws.hpp"

DPI = 140
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
FS_T, FS_C, FS_R, FS_L, FS_N = 8.0, 7.0, 6.4, 6.6, 6.9   # box title, code, file:line, arrow label, note (pt)
LH_T, LH_C, LH_R, LH_N = 0.18, 0.128, 0.108, 0.122       # row pitches (in)
PADX, PADY, GAP_R, BADGE = 0.1, 0.07, 0.04, 0.165
MARGIN, GAP, NIC_W, BAR_W = 0.28, 0.3, 0.95, 0.09
HEAD_T, HEAD_H = 0.92, 0.46        # lane headers: top and height (t = inches from the figure's top)
SEQ0 = HEAD_T + HEAD_H + 0.22      # top of the sequence area


# ---------------------------------------------------------------- source references
def poc_root():
    """examples/hstu-poc: this file's repository (POC copy of tools/) or the task folder's checkout."""
    for d in (os.path.join(HERE, ".."), os.path.join(HERE, "..", "repos/recsys-examples/examples/hstu-poc")):
        if os.path.isfile(os.path.join(d, "hstu_poc", "kvsource.py")):
            return os.path.abspath(d)
    raise SystemExit("examples/hstu-poc not found next to tools/")


_SRC = {}


def src_lines(file):
    if file not in _SRC:
        _SRC[file] = open(os.path.join(poc_root(), FILES[file])).read().splitlines()
    return _SRC[file]


def line_of(file, spec):
    """1-based line of spec in file: the first line containing "pattern", or the first one at or after the first
    line containing "after" for "after>>pattern"."""
    ls = src_lines(file)
    after, pat = spec.split(">>", 1) if ">>" in spec else (None, spec)
    i0 = 0
    if after is not None:
        i0 = next((i for i, s in enumerate(ls) if after in s), None)
        if i0 is None:
            raise SystemExit(f"{file}: not found: {after!r}")
    for i in range(i0, len(ls)):
        if pat in ls[i]:
            return i + 1
    raise SystemExit(f"{file}: not found: {spec!r}")


def fill(s, marks):
    """Replace @name by the line number of marks[name] = (file, spec)."""
    return re.sub(r"@(\w+)", lambda m: str(line_of(*marks[m.group(1)])), s)


def ref_text(refs):
    """[(file, [spec...])] -> ["file:l1, l2", ...] in the order given."""
    return [f"{f}:" + ", ".join(str(line_of(f, s)) for s in specs) for f, specs in refs]


def _s(c):
    return c if isinstance(c, str) else c[0]


# ---------------------------------------------------------------- content model
class Box:
    """One step: lane (column), title (the call or kernel), code lines (str, or (str, "hl" | "mute")), file:line refs.
    at: placement constraints on its top (see Seq.t); it also stays below the previous step and its lane's last box."""

    def __init__(self, n, lane, title, code, refs, at=()):
        self.n, self.lane, self.title, self.code, self.refs, self.at = n, lane, title, list(code), refs, list(at)
        self.kind, self.top, self.x0, self.x1, self.ref_lines = None, 0.0, 0.0, 0.0, []

    def off(self, key):
        for i, c in enumerate(self.code):
            if key in _s(c):
                return PADY + LH_T + (i + 0.5) * LH_C
        raise KeyError(f"box {self.n}: no code line contains {key!r}")

    def line_t(self, key):
        return self.top + self.off(key)

    @property
    def h(self):
        return PADY + LH_T + len(self.code) * LH_C + GAP_R + len(self.ref_lines) * LH_R + PADY

    @property
    def bot(self):
        return self.top + self.h

    @property
    def rect_t(self):
        return self.x0, self.top, self.x1, self.bot


class Seq:
    def __init__(self, title, sub, lanes, groups, marks):
        self.title, self.sub, self.lanes, self.groups, self.marks = title, sub, lanes, groups, marks
        self.boxes, self.arrows, self.bars, self.notes, self.named = {}, [], [], [], {}
        self.bottom = None

    def box(self, *a, **k):
        b = Box(*a, **k)
        b.kind = self.lanes[b.lane][2]
        self.boxes[b.n] = b

    def arrow(self, name, a, b, label, dashed=False, lab=(0.5, 0.0, 0.0, "center", "center")):
        """a, b: ("box", n, key) = the edge of box n facing the other end, at its code line containing key;
        ("life", lane) = that lane's lifeline at the other end's height, or ("life", lane, tspec).
        lab: (fraction along the arrow, dx, dt (down), ha, va) of the label."""
        self.arrows.append(dict(name=name, a=a, b=b, label=label, dashed=dashed, lab=lab))

    def bar(self, lane, t0, t1, spin=False):
        self.bars.append((lane, t0, t1, spin))

    def t(self, spec, box=None):
        """Heights (inches from the figure's top): ("abs", t) from the sequence area's top, ("top"|"bot", n, d),
        ("line", n, key, d), ("arr", name, "a"|"b", d), ("align", key, spec, d) = the top that puts this box's
        line containing key at spec + d."""
        k = spec[0]
        if k == "abs":
            return SEQ0 + spec[1]
        if k == "top":
            return self.boxes[spec[1]].top + spec[2]
        if k == "bot":
            return self.boxes[spec[1]].bot + spec[2]
        if k == "line":
            return self.boxes[spec[1]].line_t(spec[2]) + spec[3]
        if k == "arr":
            return self.named[spec[1]]["t" + spec[2]] + spec[3]
        if k == "align":
            return self.t(spec[2]) + spec[3] - box.off(spec[1])
        raise ValueError(spec)

    def refs(self):
        out = [(f, s) for b in self.boxes.values() for f, ss in b.refs for s in ss]
        return out + list(self.marks.values())


# ---------------------------------------------------------------- the two figures
def bottom_box(mode, steps_box):
    """The per-tile wait inside hstu_fwd (both figures); mode: the one this figure uses (3 or 5)."""
    h3, h5 = ("hl", "mute") if mode == 3 else ("mute", "hl")
    return dict(
        title=f"Per-tile wait inside hstu_fwd (this figure: mode {mode}): one elected lane of each CTA's producer "
              "warp, before each K tile's TMA load",
        refs="attn/hstu_hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp: tile_wait :@tw, mode 5 :@m5, mode 3 :@m3, "
             "seq :@sq; called at :@c1 (first key block) and :@c2 (the others)",
        cols=[["tile_wait(nb):                                   // :@tw",
               "  start = nb * kBlockN       // first row of key block nb",
               "  if start >= L_hist: return    // the candidates' rows",
               "  need = ceil((L_hist - start) / kv_tile_rows)",
               "  if kv_got >= need: return     // an earlier wait saw it",
               "  do",
               ("    mode 3 (:@m3): got = ld.acquire.gpu.s32 tsteps[l]", h3),
               ("    mode 5 (:@m5): p = ld.acquire.gpu.u64 prog[l]", h5),
               ("                 got = p>>32 == seq ? (p & 0xffffffff) : 0", h5),
               "  while got < need",
               "  fence.proxy.async.global     // TMA reads after the acquire",
               "  kv_got = got"],
              ["producer warp, key blocks last to first:",
               "  nb = n_block_max - 1",
               "  tile_wait(nb); TMA load K[nb]           // :@c1",
               "  for nb = n_block_max-2 down to n_block_min:",
               "    tile_wait(nb); TMA load K[nb]         // :@c2",
               "    TMA load V[nb+1]   // V follows: covered by",
               "                       // the wait for K[nb+1]"]],
        note="Chunk k of a user's K or V holds rows [L_hist − (k+1)·r, L_hist − k·r), r = kv_tile_rows (the mode's "
             "bits 8+): the history arrives from its end, the order in which these loads read it. The progress word "
             "counts steps (step k = chunk k of every user), so key block nb waits for steps 0 … need − 1. "
             "L_hist: the user's history rows (kv_hist_len); kBlockN: key rows per block; seq = *kv_seq, read once "
             f"(:@sq). The candidates' own K/V rows were written by index_copy_ before the launch (box {steps_box}).",
        marks=dict(tw=(MAINLOOP, "auto tile_wait = [&](int nb) {"), m5=(MAINLOOP, "if (kv_tile_mode == 5) {"),
                   m3=(MAINLOOP, "if (kv_tile_mode == 3) {"), sq=(MAINLOOP, "const uint32_t kv_want ="),
                   c1=(MAINLOOP, "tile_wait(n_block);"), c2=(MAINLOOP, "tile_wait(n_block_next);")))


EMBED = ("grid.py", ["x = model.embed(items, None, b.pos_q)"])
GRAPH_REPLAY = ("grid.py", ["pre(i + 3)>>g.replay()"])
NOTE_REFS = ("Not to scale. file:line in examples/hstu-poc: kvsource.py and model.py in hstu_poc/, grid.py in bench/, "
             ".cpp and .cu files in csrc/, the attention kernel in attn/hstu_hopper/.")


def verbs():
    lanes = [("Request thread", "CPU · bench/grid.py loop", "cpu"), ("NIC", "RDMA READs", "nic"),
             ("Worker thread", "CPU, spinning · VerbsKV", "cpu"), ("GPU side stream", "relay kernel", "gpu"),
             ("GPU main stream", "embedding, layers, head", "gpu")]
    groups = [(3, 4, [("request CUDA graph: captured once, ", SANS), ("g.replay()", MONO), (" per request", SANS)])]
    marks = dict(inl=("verbs_kv.cpp", "void VerbsKV::post_reads_tiled(>>if (p_->inline_post"),
                 rj=("verbs_kv.cpp", "if (!j.posted) post_job(p, j);"),
                 r1=("kvutil.cu", "__global__ void relay_tiles_kernel("))
    S = Seq("CPU verbs v2: how one request's history K/V is fetched with per-tile waits",
            "Split posting (HPOC_SPLIT_POST=1) and relay v2 (HPOC_RELAY=2); 4 QPs, a spinning worker and inline "
            "posting* (--tuned)", lanes, groups, marks)
    S.box(1, 0, "pre_replay(i) → _post_now()", [
        "cur = i % 8; seq += 1     # user set, tag",
        "v.post_reads_tiled(…, first_layers=1)",
        "→ VerbsKV::post_reads_tiled  # inline*",
        "  plan_tiled(j)  # steps, chunks per step",
        "  post_tiled_layers(layer 0):",
        "    one signaled RDMA READ per chunk,",
        "    history's end first, in step order;",
        "    wr_id = layer<<32 | step, QP rr++ % 4",
        "    ibv_post_send: one chain per QP",
        "  submit(job)     # to the worker"],
        [("kvsource.py", ["class VerbsKVSource>>def pre_replay", "class VerbsKVSource>>self.v.post_reads_tiled("]),
         ("verbs_kv.cpp", ["void VerbsKV::post_reads_tiled(", "static void plan_tiled(",
                           "static void post_tiled_layers(", "w.wr_id = ((uint64_t)l << 32)",
                           "ibv_post_send(p->qps[q], wrs[q].data()",
                           "void VerbsKV::post_reads_tiled(>>submit(p_, std::move(j));"])],
        at=[("abs", 0.0)])
    S.box(2, 0, "g.replay()", ["launch the request graph: boxes 5 to 9"], [GRAPH_REPLAY])
    S.box(3, 2, "run_job → poll_tiled", [
        "until every layer is complete:",
        "  n = ibv_poll_cq(cq, 64, wc)",
        "  got[l][step] += 1 per CQE (wr_id)",
        "  d = leading steps of l fully landed",
        "  prog_host[l] = seq<<32 | d  # release",
        "  d = 0x7fffffff: layer l complete"],
        [("verbs_kv.cpp", ["static void worker_main(", "static void run_job(", "static void poll_tiled(",
                           "static void poll_tiled(>>ibv_poll_cq(p->cq, 64, wc)", "got[wc[k].wr_id >> 32]",
                           "&p->prog_host[l], ((uint64_t)j.seq << 32)"])],
        at=[("bot", 2, 0.16)])
    S.box(4, 0, "post_replay(i) → post_rest()", [
        "post_tiled_layers(layers 1..n-1)",
        "  ibv_post_send: a chain per layer, QP"],
        [("kvsource.py", ["class VerbsKVSource>>def post_replay", "self.v.post_rest()"]),
         ("verbs_kv.cpp", ["void VerbsKV::post_rest()", "void VerbsKV::post_rest()>>post_tiled_layers("])],
        at=[("top", 3, 0.2)])
    S.box(5, 4, "start(): the graph's first nodes", [
        "E.seq_inc(seq_dev)    # device tag = seq",
        "tsteps.zero_()        # progress per layer",
        "p4_fork(relay_tiles)  # side branch"],
        [("kvsource.py", ["class VerbsKVSource>>def start(self):", "class VerbsKVSource>>self.tsteps.zero_()",
                          "class VerbsKVSource>>self.p4_fork(lambda: self.E.relay_tiles(", "def p4_fork("]),
         ("kvutil.cu", ["__global__ void seq_inc_kernel"])],
        at=[("top", 4, 0.2)])
    S.box(6, 3, "relay_tiles2_kernel<<<1, 1>>>", [
        "for each layer l:",
        "  until last == 0x7fffffff:",
        "    p = prog_host[l]     # one PCIe read",
        "    if p>>32 == seq and steps(p) > last:",
        "      last = steps(p)",
        "      __threadfence_system()",
        "      st.release.gpu tsteps[l] = last"],
        [("kvutil.cu", ["__global__ void relay_tiles2_kernel(",
                        "__global__ void relay_tiles2_kernel(>>st.release.gpu.global.s32",
                        "relay_tiles2_kernel<<<1, 1, 0, s>>>"])],
        at=[("line", 5, "p4_fork", 0.2), ("align", "p = prog_host", ("line", 3, "prog_host[l] =", 0.0), 0.3)])
    S.box(7, 4, "embed → per layer l: forward_cached", [
        "x = embed(candidates' tokens)",
        "u, v, q, k = uvqk(l, x)  # LN, GEMM, SiLU",
        "K/V.index_copy_(q_dst, k/v)  # candidates",
        "hstu_fwd(mode 3, kv_ready=&tsteps[l])"],
        [EMBED, ("model.py", ["def forward_cached(", "def forward_cached(>>self.uvqk(i, x)",
                              "K.index_copy_(0, b.q_dst, k)", "a = attn_fn(i, q, K, V, b, self)"]),
         ("kvsource.py", ["def _p4_attn(", "return hstu_fwd(*args, kv_ready=self.tsteps.data_ptr()"])])
    S.box(8, 4, "hstu_fwd: per-tile wait (box below)", [
        "producer warp of each CTA, per K tile:",
        "spin until tsteps[l] covers its chunks",
        "then the K tile's TMA load (V follows)"],
        [(MAINLOOP, ["if (kv_tile_mode == 3) {", "tile_wait(n_block);", "tile_wait(n_block_next);"])],
        at=[("align", "TMA load", ("bot", 6, 0.0), 0.16)])
    S.box(9, 4, "finish → next layer → head → end()", [
        "x = finish(l, a, u, x)  # u·LN(a), GEMM, +x",
        "next layer: boxes 7 to 9 again",
        "head(x[cand_in_q]); p4_join()  # join"],
        [("model.py", ["def forward_cached(>>x = self.finish(i, a, u, x)", "return self.head(x[b.cand_in_q])"]),
         ("kvsource.py", ["class VerbsKVSource>>def end(self):", "def p4_join("])])
    S.arrow("db0", ("box", 1, "ibv_post_send"), ("life", 1), "doorbells", lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("sub", ("box", 1, "submit(job)"), ("life", 2), "submit(job)", lab=(0.75, 0.0, 0.045, "center", "top"))
    S.arrow("launch", ("box", 2, "launch"), ("life", 4), "cudaGraphLaunch", lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("db1", ("box", 4, "ibv_post_send"), ("life", 1), "doorbells",
            lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("cqe", ("life", 1), ("box", 3, "ibv_poll_cq"), "CQE per chunk", lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("prog", ("box", 3, "prog_host[l] ="), ("box", 6, "p = prog_host"), "prog_host[l]\npinned host memory",
            dashed=True, lab=(1.0, -0.35, -0.3, "right", "center"))
    S.arrow("fork", ("box", 5, "p4_fork"), ("life", 3), "fork (graph branch)", lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("tst", ("box", 6, "st.release.gpu"), ("box", 8, "spin until"), "tsteps[l]\nGPU memory", dashed=True,
            lab=(1.0, -0.25, -0.31, "right", "center"))
    S.arrow("bytes", ("life", 1), ("box", 8, "TMA load"), "chunk bytes → local K/V buffer", dashed=True,
            lab=(0.5, 0.0, 0.0, "center", "center"))
    S.arrow("join", ("life", 3, ("line", 9, "p4_join", 0.0)), ("box", 9, "p4_join"), "join (graph edge)",
            lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.bar(0, ("top", 1, 0.0), ("bot", 4, 0.0))
    S.bar(1, ("arr", "db0", "b", 0.0), ("arr", "bytes", "a", 0.12))
    S.bar(2, ("arr", "sub", "b", 0.0), ("arr", "bytes", "a", 0.24), spin=True)
    S.bar(3, ("arr", "fork", "b", 0.0), ("arr", "join", "a", 0.0), spin=True)
    S.bar(4, ("arr", "launch", "b", 0.0), ("bot", 9, 0.0))
    S.bottom = bottom_box(3, 7)
    S.notes = ["* Inline posting: the request thread posts only while the worker is idle (verbs_kv.cpp:@inl); "
               "otherwise the worker posts the whole job in run_job (:@rj).",
               "v1 posts every layer before the launch (first_layers = 0) and uses relay v1 (relay_tiles_kernel, "
               "kvutil.cu:@r1), which also reads the worker's per-layer flags.",
               NOTE_REFS]
    return S


def raw():
    lanes = [("Request thread", "CPU · bench/grid.py loop", "cpu"), ("NIC", "RDMA READs", "nic"),
             ("QP warps 0–3", "warp w owns QP w", "gpu"), ("Publisher warp 4", "lane 0", "gpu"),
             ("GPU main stream", "the request CUDA graph", "gpu")]
    groups = [(2, 3, [("raw_post_tiled2_kernel<<<1, 32×5>>>", MONO), (": its own 2-node graph, on its own stream",
                                                                       SANS)])]
    marks = dict(gs=("kvsource.py", "def _gdaki_start("), gf=("kvsource.py", "def _gdaki_start(>>self.p4_fork(side)"))
    S = Seq("Raw GDAKI v3: how one request's history K/V is fetched with per-tile waits",
            "The v2 post kernel (HPOC_RAW_POST=2) launched in its own graph before the request graph "
            "(HPOC_RAW_PRELAUNCH=1); sequence-tagged progress words", lanes, groups, marks)
    S.box(1, 0, "pre_replay(i)", [
        "_select(i): var_np[0] = i % 8",
        "  # the user set, in pinned memory",
        "pseq += 1",
        "post_now():",
        "  E.graph_launch(pexec, pstream_ptr)",
        "  # C++ cudaGraphLaunch of the post graph:",
        "  # seq_inc(pseq_dev), then the post kernel"],
        [("kvsource.py", ["class RawGdakiSource>>def pre_replay", "def _select(", "self.var_np[0] = self.cur",
                          "self.pseq += 1", "class RawGdakiSource>>def post_now", "self.E.graph_launch(self.pexec",
                          "def _post_kernels("]),
         ("bindings.cpp", ['m.def("graph_launch"'])],
        at=[("abs", 0.0)])
    S.box(2, 0, "g.replay()", ["launch the request graph: boxes 4, 6, 7"], [GRAPH_REPLAY])
    S.box(3, 2, "QP warps: DOCA EXCLUSIVE mode", [
        "thread 0 first: s_off = *var * stride,",
        "                s_seq = *pseq_dev",
        "per layer l (warp w owns QP w):",
        "  lane 0: reserve_wq_slots(qp, cnt)",
        "  32 lanes: wqe_prepare_read per chunk,",
        "    in step order; tickets tk[l][k][w]",
        "  lane 0: mark_wqes_ready, submit",
        "  lane 0: atomicAdd(&s_posted[l], 1)"],
        [("raw_gdaki.cu", ["__global__ void raw_post_tiled2_kernel(", "__global__ void raw_post_tiled2_kernel(>>s_off =",
                           "s_seq = seq_ptr ?", "void raw_tiled_request(", "reserve_wq_slots<RAW_EXCL",
                           "void raw_tiled_request(>>doca_gpu_dev_verbs_wqe_prepare_read(",
                           "void raw_tiled_request(>>tk[(long long)k * nqp] = my + 1", "mark_wqes_ready<RAW_EXCL>",
                           "doca_gpu_dev_verbs_submit<RAW_EXCL", "atomicAdd(&s_posted[l], 1)",
                           "raw_post_tiled2_kernel<<<1, 32 * (nq + 1)"])],
        at=[("bot", 2, 0.16)])
    S.box(4, 4, "start() → embed → per layer l", [
        "E.seq_inc(seq_dev)   # = pseq: the tag",
        "x = embed(candidates' tokens)",
        "u, v, q, k = uvqk(l, x); K/V.index_copy_",
        "hstu_fwd(mode 5, kv_ready=&prog[l])"],
        [("kvsource.py", ["class RawGdakiSource>>def start(self):", "class RawGdakiSource>>self.E.seq_inc(self.seq_dev)",
                          "kv_ready=self.prog.data_ptr() + 8 * l"]), EMBED,
         ("model.py", ["def forward_cached(>>self.uvqk(i, x)", "K.index_copy_(0, b.q_dst, k)",
                       "a = attn_fn(i, q, K, V, b, self)"])],
        at=[("top", 3, 0.2)])
    S.box(5, 3, "publisher: warp 4, lane 0", [
        "per layer l:",
        "  spin until s_posted[l] == 4",
        "  per step k, each QP q with a ticket:",
        "    doca_gpu_dev_verbs_wait(qp, tk[l][k][q])",
        "      # polls that QP's CQ",
        "    st.release.gpu prog[l] = seq<<32 | k+1",
        "  prog[l] = seq<<32 | 0x7fffffff"],
        [("raw_gdaki.cu", ["while (*(volatile int*)&s_posted[l] < nqp)", "doca_gpu_dev_verbs_wait<RAW_EXCL>",
                           "raw_publish(tsteps, prog, seq, l, k + 1)", "void raw_publish(>>st.release.gpu.global.u64",
                           "raw_publish(tsteps, prog, seq, l, 0x7fffffff)"])],
        at=[("align", "doca_gpu_dev_verbs_wait", ("bot", 3, 0.0), 0.42)])
    S.box(6, 4, "hstu_fwd: per-tile wait (box below)", [
        "producer warp of each CTA, per K tile:",
        "spin until prog[l], tagged seq, covers it",
        "then the K tile's TMA load (V follows)"],
        [(MAINLOOP, ["if (kv_tile_mode == 5) {", "tile_wait(n_block);", "tile_wait(n_block_next);"])],
        at=[("align", "TMA load", ("bot", 5, 0.0), 0.46)])
    S.box(7, 4, "finish → next layer → head", [
        "x = finish(l, a, u, x)  # u·LN(a), GEMM, +x",
        "next layer: boxes 4, 6, 7 again",
        "head(x[cand_in_q])",
        "end(): no join, the post kernel is",
        "  outside this graph"],
        [("model.py", ["def forward_cached(>>x = self.finish(i, a, u, x)", "return self.head(x[b.cand_in_q])"]),
         ("kvsource.py", ["class RawGdakiSource>>def end(self):"])])
    S.arrow("plaunch", ("box", 1, "graph_launch"), ("life", 2), "cudaGraphLaunch (pstream)",
            lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("launch", ("box", 2, "launch"), ("life", 4), "cudaGraphLaunch", lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("db", ("box", 3, "submit"), ("life", 1), "doorbell", lab=(0.5, 0.0, -0.045, "center", "bottom"))
    S.arrow("posted", ("box", 3, "atomicAdd"), ("box", 5, "s_posted[l] == 4"), "s_posted[l]\nshared memory",
            dashed=True, lab=(1.0, -0.36, 0.05, "right", "center"))
    S.arrow("cqe", ("life", 1), ("box", 5, "doca_gpu_dev_verbs_wait"), "CQE per chunk (CQ in GPU memory)",
            lab=(0.4, 0.0, -0.045, "center", "bottom"))
    S.arrow("prog", ("box", 5, "st.release.gpu"), ("box", 6, "spin until"), "prog[l]\nGPU memory", dashed=True,
            lab=(1.0, -0.35, -0.06, "right", "center"))
    S.arrow("bytes", ("life", 1), ("box", 6, "TMA load"), "chunk bytes → local K/V buffer", dashed=True,
            lab=(0.5, 0.0, 0.0, "center", "center"))
    S.bar(0, ("top", 1, 0.0), ("bot", 2, 0.0))
    S.bar(1, ("arr", "db", "b", 0.0), ("arr", "bytes", "a", 0.12))
    S.bar(2, ("arr", "plaunch", "b", 0.0), ("bot", 3, 0.0))
    S.bar(3, ("arr", "plaunch", "b", 0.0), ("arr", "bytes", "a", 0.24), spin=True)
    S.bar(4, ("arr", "launch", "b", 0.0), ("bot", 7, 0.0))
    S.bottom = bottom_box(5, 4)
    S.notes = ["v2 (no HPOC_RAW_PRELAUNCH) runs the same kernel as a side branch inside the request graph (_gdaki_start, kvsource.py:@gs, "
               "fork at :@gf), publishes int32 tsteps[l] (attention mode 3) and is joined at the end (p4_join).",
               NOTE_REFS]
    return S


# ---------------------------------------------------------------- drawing
def _mods():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import lines, patches
    sys.path.insert(0, HERE)
    import fetch_diagrams as fd   # one palette for the schematic figures
    return plt, patches, lines, fd


def _wrap(text, width, size, tw, family=SANS):
    """Greedy word wrap to width inches (measured)."""
    out, cur = [], ""
    for w in text.split(" "):
        cand = w if not cur else cur + " " + w
        if tw(cand, size, family) <= width or not cur:
            cur = cand
        else:
            out.append(cur)
            cur = w
    return out + ([cur] if cur else [])


def _wrap_refs(groups, width, tw):
    """Join "file:l1, l2" groups with " · ", breaking between groups (or inside a too-long group's list)."""
    out, cur = [], ""
    for g in groups:
        cand = g if not cur else cur + "  ·  " + g
        if tw(cand, FS_R) <= width:
            cur = cand
            continue
        if cur:
            out.append(cur)
        cur = ""
        if tw(g, FS_R) <= width:
            cur = g
            continue
        head, nums = g.split(":", 1)
        parts = nums.split(", ")
        cur = head + ":" + parts[0]
        for p in parts[1:]:
            if tw(cur + ", " + p, FS_R) <= width:
                cur += ", " + p
            else:
                out.append(cur + ",")
                cur = "    " + p
    return out + ([cur] if cur else [])


def render(S, out):
    plt, P, Lm, fd = _mods()
    BOXC, INK, INK2, MUTED, SURFACE = fd.BOX, fd.INK, fd.INK2, fd.MUTED, fd.SURFACE
    meas = plt.figure(dpi=DPI)
    rnd = meas.canvas.get_renderer()
    cache = {}

    def tw(s, size, family=SANS, weight="normal"):
        k = (s, size, family, weight)
        if k not in cache:
            t = meas.text(0, 0, s, fontsize=size, family=family, fontweight=weight)
            cache[k] = t.get_window_extent(rnd).width / DPI
            t.remove()
        return cache[k]

    CW = tw("0" * 40, FS_C, MONO) / 40
    # ---- columns: as wide as their widest box (or header)
    widths = []
    for i, (name, sub, kind) in enumerate(S.lanes):
        need = max(tw(name, 8.2, SANS, "bold"), tw(sub, 6.8)) + 0.24
        bs = [b for b in S.boxes.values() if b.lane == i]
        for b in bs:
            need = max(need, 2 * PADX + BADGE + 0.07 + tw(b.title, FS_T, SANS, "bold"),
                       2 * PADX + max(len(_s(c)) for c in b.code) * CW)
        widths.append(need if bs else max(need, NIC_W))
    lanes, x = [], MARGIN
    for w in widths:
        lanes.append((x, x + w))
        x += w + GAP
    W = x - GAP + MARGIN
    cx = [(a + b) / 2 for a, b in lanes]
    # ---- boxes: refs wrapped to the column, then placed top-down in step order
    for b in S.boxes.values():
        b.x0, b.x1 = lanes[b.lane]
        b.ref_lines = _wrap_refs(ref_text(b.refs), b.x1 - b.x0 - 2 * PADX, tw)
    prev, lane_bot = None, {}
    for n in sorted(S.boxes):
        b = S.boxes[n]
        cand = [SEQ0] + [S.t(c, b) for c in b.at]
        if prev is not None:
            cand.append(prev.top + 0.1)
        if b.lane in lane_bot:
            cand.append(lane_bot[b.lane] + 0.16)
        b.top = max(cand)
        lane_bot[b.lane] = b.bot
        prev = b
    # ---- arrows and activation bars
    def approx_x(e):
        return (S.boxes[e[1]].x0 + S.boxes[e[1]].x1) / 2 if e[0] == "box" else cx[e[1]]

    def end(e, other_x):
        if e[0] == "box":
            b = S.boxes[e[1]]
            return (b.x1 if other_x > (b.x0 + b.x1) / 2 else b.x0), b.line_t(e[2])
        xl = cx[e[1]] + (BAR_W / 2 if other_x > cx[e[1]] else -BAR_W / 2)
        return xl, (S.t(e[2]) if len(e) > 2 else None)

    for a in S.arrows:
        xa, ta = end(a["a"], approx_x(a["b"]))
        xb, tb = end(a["b"], approx_x(a["a"]))
        a.update(xa=xa, ta=ta if ta is not None else tb, xb=xb, tb=tb if tb is not None else ta)
        S.named[a["name"]] = a
    bars = [(lane, S.t(t0), S.t(t1), spin) for lane, t0, t1, spin in S.bars]
    seq_end = max([b.bot for b in S.boxes.values()] + [max(a["ta"], a["tb"]) for a in S.arrows] +
                  [t1 for _, _, t1, _ in bars]) + 0.22
    # ---- bottom box, legend, notes
    bb = S.bottom
    marks = dict(S.marks, **bb["marks"])
    cols = [[(fill(_s(c), marks), None if isinstance(c, str) else c[1]) for c in col] for col in bb["cols"]]
    col_w = [max(len(s) for s, _ in col) * CW for col in cols]
    bx0, bx1 = MARGIN, W - MARGIN
    note_x = bx0 + PADX + sum(col_w) + 0.4 * len(col_w)
    note_lines = _wrap(fill(bb["note"], marks), bx1 - PADX - note_x, FS_N, tw)
    b_top = seq_end + 0.1
    b_body = b_top + PADY + LH_T + LH_R + 0.07
    b_bot = b_body + max(max(len(c) for c in cols) * LH_C, len(note_lines) * LH_N) + PADY
    t_leg = b_bot + 0.16
    notes = [ln for n in S.notes for ln in _wrap(fill(n, S.marks), W - 2 * MARGIN, FS_N, tw)]
    t_notes = t_leg + 0.44
    H = t_notes + len(notes) * 0.15 + 0.16
    plt.close(meas)

    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    def Y(t):
        return H - t

    rects, owned, labels, arrows_xy = {}, [], [], []
    ax.text(MARGIN, Y(0.26), S.title, ha="left", va="center", fontsize=10.5, fontweight="bold", color=INK)
    t_sub = ax.text(MARGIN, Y(0.5), S.sub, ha="left", va="center", fontsize=7.6, color=INK2)
    free = [t_sub]
    for a, b, segs in S.groups:                  # bracket over grouped columns
        x0, x1 = lanes[a][0], lanes[b][1]
        tb = HEAD_T - 0.09
        ax.plot([x0, x0, x1, x1], [Y(HEAD_T - 0.03), Y(tb), Y(tb), Y(HEAD_T - 0.03)], color=INK2, lw=0.8)
        total = sum(tw(s, 7.4, f) for s, f in segs)
        xs = (x0 + x1) / 2 - total / 2
        for s, f in segs:
            free.append(ax.text(xs, Y(tb - 0.09), s, ha="left", va="center_baseline", fontsize=7.4, family=f,
                                color=INK2))
            xs += tw(s, 7.4, f)
    for i, (name, sub, kind) in enumerate(S.lanes):   # headers and lifelines
        fc, ec, tc, bc = BOXC[kind]
        x0, x1 = lanes[i]
        ax.add_patch(P.FancyBboxPatch((x0, Y(HEAD_T + HEAD_H)), x1 - x0, HEAD_H, boxstyle="round,pad=0,rounding_size=0.06",
                                      facecolor=fc, edgecolor=ec, lw=0.8, zorder=3))
        rects[f"header {i}"] = (x0, Y(HEAD_T + HEAD_H), x1, Y(HEAD_T))
        owned.append((f"header {i}", ax.text(cx[i], Y(HEAD_T + 0.165), name, ha="center", va="center", fontsize=8.2,
                                             fontweight="bold", color=tc, zorder=4)))
        owned.append((f"header {i}", ax.text(cx[i], Y(HEAD_T + 0.335), sub, ha="center", va="center", fontsize=6.8,
                                             color=bc, zorder=4)))
        ax.plot([cx[i], cx[i]], [Y(HEAD_T + HEAD_H), Y(seq_end)], color=MUTED, lw=0.7, ls=(0, (3, 3)), alpha=0.8,
                zorder=1)
    bar_rects = []
    for lane, t0, t1, spin in bars:              # activation bars
        fc, ec = BOXC[S.lanes[lane][2]][:2]
        ax.add_patch(P.Rectangle((cx[lane] - BAR_W / 2, Y(t1)), BAR_W, t1 - t0, facecolor=fc, edgecolor=ec, lw=0.7,
                                 hatch="/////" if spin else None, zorder=2))
        bar_rects.append((f"bar {lane}", (cx[lane] - BAR_W / 2, Y(t1), cx[lane] + BAR_W / 2, Y(t0))))

    def code_line(x, y, c, style, tc, bc, owner):
        s = c
        k = s.find("  # ")
        if k < 0:
            k = s.find("  // ")
        code, com = (s, "") if k < 0 else (s[:k], s[k:])
        if code.strip():
            owned.append((owner, ax.text(x, y, code, ha="left", va="center_baseline", fontsize=FS_C, family=MONO,
                                         color=MUTED if style == "mute" else tc,
                                         fontweight="bold" if style == "hl" else "normal", zorder=5)))
        if com:
            owned.append((owner, ax.text(x + len(code) * CW, y, com, ha="left", va="center_baseline", fontsize=FS_C,
                                         family=MONO, fontstyle="italic", color=MUTED if style == "mute" else bc,
                                         zorder=5)))

    for b in S.boxes.values():                   # step boxes
        fc, ec, tc, bc = BOXC[b.kind]
        name = f"box {b.n}"
        ax.add_patch(P.FancyBboxPatch((b.x0, Y(b.bot)), b.x1 - b.x0, b.h, boxstyle="round,pad=0,rounding_size=0.06",
                                      facecolor=fc, edgecolor=ec, lw=0.8, zorder=3))
        rects[name] = (b.x0, Y(b.bot), b.x1, Y(b.top))
        ty = Y(b.top + PADY + LH_T / 2)
        ax.add_patch(P.Circle((b.x0 + PADX + BADGE / 2, ty), BADGE / 2, facecolor=tc, edgecolor="none", zorder=4))
        ax.text(b.x0 + PADX + BADGE / 2, ty, str(b.n), ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", zorder=5)
        owned.append((name, ax.text(b.x0 + PADX + BADGE + 0.07, ty, b.title, ha="left", va="center", fontsize=FS_T,
                                    fontweight="bold", color=tc, zorder=5)))
        for i, c in enumerate(b.code):
            code_line(b.x0 + PADX, Y(b.top + PADY + LH_T + (i + 0.5) * LH_C), _s(c),
                      None if isinstance(c, str) else c[1], tc, bc, name)
        for j, rl in enumerate(b.ref_lines):
            owned.append((name, ax.text(b.x0 + PADX, Y(b.top + PADY + LH_T + len(b.code) * LH_C + GAP_R + (j + 0.5) * LH_R),
                                        rl, ha="left", va="center", fontsize=FS_R, color=INK2, zorder=5)))
    for a in S.arrows:                           # arrows and their labels
        p0, p1 = (a["xa"], Y(a["ta"])), (a["xb"], Y(a["tb"]))
        kw = dict(color=INK2, lw=0.9, shrinkA=0, shrinkB=0, zorder=6)
        if a["dashed"]:                          # dashed shaft, solid head
            d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            q = (p1[0] - (p1[0] - p0[0]) / d * 0.09, p1[1] - (p1[1] - p0[1]) / d * 0.09)
            ax.add_patch(P.FancyArrowPatch(p0, q, arrowstyle="-", linestyle=(0, (3.2, 2.2)), **kw))
            ax.add_patch(P.FancyArrowPatch(q, p1, arrowstyle="-|>", mutation_scale=9, **kw))
        else:
            ax.add_patch(P.FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, **kw))
        f, dx, dt, ha, va = a["lab"]
        lx, ly = p0[0] + f * (p1[0] - p0[0]) + dx, p0[1] + f * (p1[1] - p0[1]) - dt
        if a["dashed"]:
            fc, ec, tc, _ = BOXC["mem"]
            box_kw = dict(boxstyle="round,pad=0.25,rounding_size=0.4", facecolor=fc, edgecolor=ec, lw=0.6)
        else:
            tc, box_kw = INK2, dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none")
        labels.append((a["name"], ax.text(lx, ly, a["label"], ha=ha, va=va, fontsize=FS_L, color=tc, zorder=8,
                                          multialignment="center",
                                          bbox=box_kw)))
        src = f"box {a['a'][1]}" if a["a"][0] == "box" else None
        dst = f"box {a['b'][1]}" if a["b"][0] == "box" else None
        arrows_xy.append((a["name"], p0, p1, src, dst))
    # ---- the per-tile wait (bottom box)
    fc, ec, tc, bc = BOXC["gpu"]
    ax.add_patch(P.FancyBboxPatch((bx0, Y(b_bot)), bx1 - bx0, b_bot - b_top, boxstyle="round,pad=0,rounding_size=0.07",
                                  facecolor=fc, edgecolor=ec, lw=0.8, zorder=3))
    rects["bottom"] = (bx0, Y(b_bot), bx1, Y(b_top))
    owned.append(("bottom", ax.text(bx0 + PADX, Y(b_top + PADY + LH_T / 2), bb["title"], ha="left", va="center",
                                    fontsize=FS_T, fontweight="bold", color=tc, zorder=5)))
    owned.append(("bottom", ax.text(bx0 + PADX, Y(b_top + PADY + LH_T + LH_R / 2), fill(bb["refs"], marks), ha="left",
                                    va="center", fontsize=FS_R, color=INK2, zorder=5)))
    xc = bx0 + PADX
    for col, w in zip(cols, col_w):
        for i, (s, style) in enumerate(col):
            code_line(xc, Y(b_body + (i + 0.5) * LH_C), s, style, tc, bc, "bottom")
        xc += w + 0.4
    for i, s in enumerate(note_lines):
        owned.append(("bottom", ax.text(note_x, Y(b_body + (i + 0.5) * LH_N), s, ha="left", va="center",
                                        fontsize=FS_N, color=bc, zorder=5)))
    # ---- legend and notes
    handles = [P.Patch(facecolor=BOXC[k][0], edgecolor=BOXC[k][1], label=lab) for k, lab in
               (("cpu", "CPU thread"), ("gpu", "GPU kernel or stream"), ("nic", "NIC"),
                ("mem", "memory word (dashed arrow)"))]
    handles += [Lm.Line2D([], [], color=INK2, lw=0.9, label="call, launch, doorbell or completion"),
                Lm.Line2D([], [], color=INK2, lw=0.9, ls=(0, (3.2, 2.2)),
                          label="a value in memory that the next column reads"),
                P.Patch(facecolor=BOXC["gpu"][0], edgecolor=BOXC["gpu"][1], label="activation: running"),
                P.Patch(facecolor=BOXC["gpu"][0], edgecolor=BOXC["gpu"][1], hatch="/////",
                        label="activation: spinning or polling")]
    leg = fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=7.2, labelcolor=INK,
                     bbox_to_anchor=(MARGIN / W - 0.004, 1 - t_leg / H), handlelength=2.2, columnspacing=1.8,
                     borderaxespad=0)
    for i, s in enumerate(notes):
        free.append(ax.text(MARGIN, Y(t_notes + (i + 0.5) * 0.15), s, ha="left", va="center", fontsize=FS_N,
                            color=INK2))
    n_bad = check(fig, rects, owned, labels, arrows_xy, bar_rects, free, leg, W, H)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}  ({W:.2f} x {H:.2f} in, {n_bad} layout problems)")
    return n_bad


def check(fig, rects, owned, labels, arrows_xy, bar_rects, free, leg, W, H):
    """Print text outside its box, overlapping boxes, labels on boxes / bars / other labels / other arrows, arrows
    through boxes other than their ends, and anything outside the figure. Returns the number of problems."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def ext(artist):
        p = artist.get_bbox_patch() if hasattr(artist, "get_bbox_patch") else None
        e = (p if p is not None else artist).get_window_extent(r)
        return e.x0 / DPI, e.y0 / DPI, e.x1 / DPI, e.y1 / DPI

    def inside(a, b, tol=0.012):
        return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol

    def overlap(a, b, tol=0.0):
        return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol

    bad = []
    for owner, t in owned:
        if not inside(ext(t), rects[owner]):
            bad.append(f"text leaves {owner}: {t.get_text()!r}")
    names = list(rects)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if overlap(rects[names[i]], rects[names[j]], 0.005):
                bad.append(f"{names[i]} overlaps {names[j]}")
    lab_ext = [(n, ext(t)) for n, t in labels]
    for n, e in lab_ext:
        for rn, rc in list(rects.items()) + bar_rects:
            if overlap(e, rc, 0.005):
                bad.append(f"label of {n} overlaps {rn}")
        for n2, e2 in lab_ext:
            if n2 > n and overlap(e, e2, 0.005):
                bad.append(f"labels of {n} and {n2} overlap")
    for n, p0, p1, src, dst in arrows_xy:
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        hit = set()
        for k in range(1, int(d / 0.01)):
            f = k * 0.01 / d
            if f * d < 0.03 or (1 - f) * d < 0.03:
                continue
            x, y = p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1])
            for rn, (a, b, c, e) in rects.items():
                if rn not in (src, dst) and a + 0.015 < x < c - 0.015 and b + 0.015 < y < e - 0.015:
                    hit.add(rn)
            for n2, (a, b, c, e) in lab_ext:
                if n2 != n and a < x < c and b < y < e:
                    hit.add(f"label of {n2}")
        bad += [f"arrow {n} crosses {h}" for h in sorted(hit)]
    for t in [t for _, t in owned] + [t for _, t in labels] + free + [leg]:
        e = ext(t)
        if e[0] < 0.02 or e[1] < 0.02 or e[2] > W - 0.02 or e[3] > H - 0.02:
            bad.append(f"outside the figure: {getattr(t, 'get_text', lambda: 'legend')()!r}")
    le = ext(leg)
    for rn, rc in rects.items():
        if overlap(le, rc):
            bad.append(f"legend overlaps {rn}")
    for t in free:
        for rn, rc in rects.items():
            if overlap(ext(t), rc):
                bad.append(f"{t.get_text()!r} overlaps {rn}")
        if overlap(ext(t), le):
            bad.append(f"legend overlaps {t.get_text()!r}")
    for b in bad:
        print("  LAYOUT:", b)
    return len(bad)


def print_refs():
    for S in (verbs(), raw()):
        print("==", S.title)
        for f, spec in S.refs() + list(S.bottom["marks"].values()):
            n = line_of(f, spec)
            print(f"  {f}:{n:<5} {src_lines(f)[n - 1].strip()[:96]}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--refs"]:
        print_refs()
        sys.exit(0)
    d = sys.argv[1] if len(sys.argv) > 1 else "docs/figs"
    bad = render(verbs(), f"{d}/per-tile-verbs.png") + render(raw(), f"{d}/per-tile-raw.png")
    sys.exit(1 if bad else 0)
