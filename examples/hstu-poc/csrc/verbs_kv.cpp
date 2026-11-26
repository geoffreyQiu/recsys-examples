// CPU verbs transport, see verbs_kv.h.
#include "verbs_kv.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace hpoc {

#define VCHECK(x, msg) \
  do {                 \
    if (!(x)) throw std::runtime_error(std::string("verbs: ") + msg + ": " + strerror(errno)); \
  } while (0)
#define CUDACHK(cmd)                                                                                       \
  do {                                                                                                     \
    cudaError_t r_ = (cmd);                                                                                \
    if (r_ != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + #cmd + ": " + cudaGetErrorString(r_)); \
  } while (0)

struct Job {
  std::vector<int64_t> segs;   // 4 per segment
  std::vector<int> lp;
  int n_layers = 0;
  uint32_t seq = 0;
  bool write = false;
  bool gated = false;          // W path: layer l waits for wflags[l] >= seq, fence + done flag at the end
  int64_t fence_remote = 0, fence_local = 0;
  std::vector<int> expect;     // per layer: signaled (layer, QP) posts to wait for
  bool posted = false;         // inline mode: the caller thread already posted the reads
  // tiled reads (per-tile waits): every segment split into chunks of tile_bytes posted from its END backwards (the
  // attention consumes key blocks last-to-first); step k of a layer = chunk k of every segment of the layer
  int64_t tile_bytes = 0;
  std::vector<std::vector<int>> step_cnt;
  std::vector<int> n_steps;
};

struct VerbsImpl {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  std::vector<ibv_qp*> qps;
  ibv_mr* mr = nullptr;
  void* buf = nullptr;
  size_t bytes = 0;
  bool gpu = false;
  int cuda_dev = -1;
  int port = 1;
  ibv_port_attr pattr{};
  ibv_gid gid{};
  uint64_t raddr = 0;
  uint32_t rkey = 0;
  uint32_t* flags_host = nullptr;
  uint32_t* flags_dev = nullptr;
  uint64_t* prog_host = nullptr;     // tiled reads: per layer (seq << 32) | steps landed (contiguous from step 0)
  uint64_t* prog_dev = nullptr;
  uint32_t* wflags_host = nullptr;   // GPU -> CPU: layer l's new K/V are ready to be written
  uint32_t* wflags_dev = nullptr;
  uint32_t* done_host = nullptr;     // CPU -> GPU: request seq's writes are durable
  uint32_t* done_dev = nullptr;
  int max_layers = 0, flush_mode = 0, cpu_core = -1, gdr_order = -1;
  int sq_depth = 4096;
  std::thread worker;
  std::mutex mu;
  std::condition_variable cv, cv_idle;
  std::deque<Job> jobs;
  bool stop = false;
  int64_t submitted = 0;
  std::atomic<int64_t> done{0};
  std::atomic<int64_t> cpu_ns{0};
  std::string err;
  // latency options: spin = the worker busy-polls for new jobs (no futex wake-up per request, one core always busy);
  // inline_post = the submitting thread posts the reads itself when nothing is in flight (the worker only polls the CQ)
  std::atomic<int64_t> pending{0};
  std::atomic<bool> spin{false}, stop_a{false};
  bool inline_post = false;
  // timeline trace (optional): CLOCK_MONOTONIC ns of the last request's posting (start, end), the worker taking the
  // job, the first completion, the end of the job, and events (layer, value published, time)
  std::atomic<bool> trace{false};
  std::vector<int64_t> tev;
  int64_t t_post0 = 0, t_post1 = 0, t_pick = 0, t_cqe0 = 0, t_end = 0, t_post2 = 0;
  // split posting: the layers of a tiled request still to be posted by post_rest() (request thread, after the graph
  // launch) while the worker already polls the completions of the first part
  struct Pending {
    bool active = false;
    std::vector<int64_t> segs;
    std::vector<int> lp;
    int64_t tb = 0;
    int l0 = 0, n_layers = 0, rr = 0;
  } pend;
};

static int nic_numa(const std::string& dev) {
  std::ifstream f("/sys/class/infiniband/" + dev + "/device/numa_node");
  int n = -1;
  if (f) f >> n;
  return n < 0 ? 0 : n;
}

static void* numa_host_alloc(size_t bytes, int node) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  VCHECK(p != MAP_FAILED, "mmap");
  madvise(p, bytes, MADV_HUGEPAGE);
  unsigned long mask = 1ul << node;
  long r = syscall(SYS_mbind, p, bytes, 2 /*MPOL_BIND*/, &mask, 64, 0);
  VCHECK(r == 0, "mbind");
  return p;
}

static int64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
static void trace_begin(VerbsImpl* p) {
  p->tev.clear();
  p->t_post2 = 0;
  p->t_cqe0 = p->t_end = 0;   // t_pick: set when the worker takes the job (before or after posting)
  p->t_post0 = mono_ns();
}
static inline void trace_ev(VerbsImpl* p, int l, int64_t v) {
  p->tev.push_back(l);
  p->tev.push_back(v);
  p->tev.push_back(mono_ns());
}

static int64_t thread_ns() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

// plan of a tiled request: steps and chunks per step of every layer (the worker needs them before it polls)
static void plan_tiled(Job& j) {
  const int64_t tb = j.tile_bytes;
  j.step_cnt.assign(j.n_layers, {});
  j.n_steps.assign(j.n_layers, 0);
  j.expect.assign(j.n_layers, 0);
  for (int l = 0; l < j.n_layers; ++l) {
    int64_t maxb = 0;
    for (int i = j.lp[l]; i < j.lp[l + 1]; ++i) maxb = std::max(maxb, j.segs[4 * (size_t)i + 2]);
    const int ns = (int)((maxb + tb - 1) / tb);
    j.n_steps[l] = ns;
    j.step_cnt[l].assign(ns, 0);
    for (int k = 0; k < ns; ++k)
      for (int i = j.lp[l]; i < j.lp[l + 1]; ++i)
        if (j.segs[4 * (size_t)i + 2] > k * tb) j.step_cnt[l][k]++;
  }
}

// post layers [l0, l1) of a tiled request, chunk reads in step order, round-robin over the QPs (rr carries on across
// calls), one chained ibv_post_send per layer and QP
static void post_tiled_layers(VerbsImpl* p, const std::vector<int64_t>& segs, const std::vector<int>& lp, int64_t tb,
                              int l0, int l1, int& rr) {
  const int nqp = (int)p->qps.size();
  std::vector<std::vector<ibv_sge>> sges(nqp);
  std::vector<std::vector<ibv_send_wr>> wrs(nqp);
  for (int l = l0; l < l1; ++l) {
    const int s0 = lp[l], s1 = lp[l + 1];
    int64_t maxb = 0;
    size_t total = 0;
    for (int i = s0; i < s1; ++i) {
      maxb = std::max(maxb, segs[4 * (size_t)i + 2]);
      total += (size_t)((segs[4 * (size_t)i + 2] + tb - 1) / tb);
    }
    const int ns = (int)((maxb + tb - 1) / tb);
    for (int q = 0; q < nqp; ++q) {
      sges[q].clear();
      wrs[q].clear();
      sges[q].reserve(total);
      wrs[q].reserve(total);
    }
    for (int k = 0; k < ns; ++k) {
      for (int i = s0; i < s1; ++i) {
        const int64_t* s = &segs[4 * (size_t)i];
        const int64_t bytes = s[2];
        if (bytes <= k * tb) continue;
        const int64_t len = std::min(tb, bytes - k * tb);
        const int64_t off = bytes - k * tb - len;
        const int q = rr++ % nqp;
        ibv_sge g{};
        g.addr = (uint64_t)p->buf + (uint64_t)(s[1] + off);
        g.length = (uint32_t)len;
        g.lkey = p->mr->lkey;
        sges[q].push_back(g);
        ibv_send_wr w;
        memset(&w, 0, sizeof(w));
        w.wr_id = ((uint64_t)l << 32) | (uint64_t)k;
        w.num_sge = 1;
        w.opcode = IBV_WR_RDMA_READ;
        w.send_flags = IBV_SEND_SIGNALED;
        w.wr.rdma.remote_addr = p->raddr + (uint64_t)(s[0] + off);
        w.wr.rdma.rkey = p->rkey;
        wrs[q].push_back(w);
      }
    }
    for (int q = 0; q < nqp; ++q) {
      if (wrs[q].empty()) continue;
      for (size_t m = 0; m < wrs[q].size(); ++m) {
        wrs[q][m].sg_list = &sges[q][m];
        wrs[q][m].next = (m + 1 < wrs[q].size()) ? &wrs[q][m + 1] : nullptr;
      }
      ibv_send_wr* bad = nullptr;
      int r = ibv_post_send(p->qps[q], wrs[q].data(), &bad);
      if (r) throw std::runtime_error("ibv_post_send (tiled) failed: " + std::string(strerror(r)));
    }
  }
}

static void post_tiled(VerbsImpl* p, Job& j) {
  if (p->trace) trace_begin(p);
  plan_tiled(j);
  int rr = 0;
  post_tiled_layers(p, j.segs, j.lp, j.tile_bytes, 0, j.n_layers, rr);
  j.posted = true;
  if (p->trace) p->t_post1 = mono_ns();
}

static void poll_tiled(VerbsImpl* p, Job& j) {
  std::vector<std::vector<int>> got(j.n_layers);
  std::vector<int> done(j.n_layers, 0);
  for (int l = 0; l < j.n_layers; ++l) got[l].assign(j.n_steps[l], 0);
  int next = 0;
  ibv_wc wc[64];
  while (true) {
    for (int l = next; l < j.n_layers; ++l) {
      int d = done[l];
      while (d < j.n_steps[l] && got[l][d] >= j.step_cnt[l][d]) ++d;
      if (d != done[l]) {
        done[l] = d;
        const uint64_t v = d >= j.n_steps[l] ? 0x7fffffffull : (uint64_t)d;   // complete: one read tells the relay
        __atomic_store_n(&p->prog_host[l], ((uint64_t)j.seq << 32) | v, __ATOMIC_RELEASE);
        if (p->trace) trace_ev(p, l, d);
      }
    }
    while (next < j.n_layers && done[next] >= j.n_steps[next]) {
      __atomic_store_n(&p->flags_host[next], j.seq, __ATOMIC_RELEASE);
      ++next;
    }
    if (next >= j.n_layers) break;
    int n = ibv_poll_cq(p->cq, 64, wc);
    if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
    if (n > 0 && p->trace && p->t_cqe0 == 0) p->t_cqe0 = mono_ns();
    for (int k = 0; k < n; ++k) {
      if (wc[k].status != IBV_WC_SUCCESS)
        throw std::runtime_error(std::string("work completion error: ") + ibv_wc_status_str(wc[k].status));
      got[wc[k].wr_id >> 32][wc[k].wr_id & 0xffffffffull]++;
    }
  }
  if (p->trace) p->t_end = mono_ns();
}

static void post_job(VerbsImpl* p, Job& j) {
  if (j.tile_bytes > 0) return post_tiled(p, j);
  if (p->trace && !j.write) trace_begin(p);
  const int nqp = (int)p->qps.size();
  std::vector<int>& expect = j.expect;
  expect.assign(j.n_layers, 0);
  std::vector<ibv_sge> sges;
  std::vector<ibv_send_wr> wrs;
  for (int l = 0; l < j.n_layers; ++l) {
    int s0 = j.lp[l], s1 = j.lp[l + 1];
    for (int q = 0; q < nqp; ++q) {
      sges.clear();
      wrs.clear();
      for (int i = s0 + q; i < s1; i += nqp) {
        const int64_t* s = &j.segs[4 * (size_t)i];
        ibv_sge g{};
        g.addr = (uint64_t)p->buf + (uint64_t)s[1];
        g.length = (uint32_t)s[2];
        g.lkey = p->mr->lkey;
        sges.push_back(g);
      }
      if (sges.empty()) continue;
      wrs.resize(sges.size());
      for (size_t k = 0; k < sges.size(); ++k) {
        int i = s0 + q + (int)k * nqp;
        const int64_t* s = &j.segs[4 * (size_t)i];
        ibv_send_wr& w = wrs[k];
        memset(&w, 0, sizeof(w));
        w.wr_id = ((uint64_t)l << 8) | (uint64_t)q;
        w.sg_list = &sges[k];
        w.num_sge = 1;
        w.opcode = j.write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
        w.send_flags = (k + 1 == sges.size()) ? IBV_SEND_SIGNALED : 0;
        w.wr.rdma.remote_addr = p->raddr + (uint64_t)s[0];
        w.wr.rdma.rkey = p->rkey;
        w.next = (k + 1 < sges.size()) ? &wrs[k + 1] : nullptr;
      }
      ibv_send_wr* bad = nullptr;
      int r = ibv_post_send(p->qps[q], wrs.data(), &bad);
      if (r) throw std::runtime_error("ibv_post_send failed: " + std::string(strerror(r)));
      expect[l]++;
    }
  }
  j.posted = true;
  if (p->trace && !j.write) p->t_post1 = mono_ns();
}

static void run_job(VerbsImpl* p, Job& j) {
  if (!j.posted) post_job(p, j);
  if (j.tile_bytes > 0) return poll_tiled(p, j);
  const std::vector<int>& expect = j.expect;
  std::vector<int> got(j.n_layers, 0);
  int next = 0;
  ibv_wc wc[64];
  while (next < j.n_layers) {
    while (next < j.n_layers && got[next] >= expect[next]) {
      if (!j.write && p->flush_mode == 1) {
        cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                   CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
      }
      if (!j.write) __atomic_store_n(&p->flags_host[next], j.seq, __ATOMIC_RELEASE);
      if (p->trace && !j.write) trace_ev(p, next, 0x7fffffff);
      ++next;
    }
    if (next >= j.n_layers) break;
    int n = ibv_poll_cq(p->cq, 64, wc);
    if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
    if (n > 0 && p->trace && !j.write && p->t_cqe0 == 0) p->t_cqe0 = mono_ns();
    for (int k = 0; k < n; ++k) {
      if (wc[k].status != IBV_WC_SUCCESS)
        throw std::runtime_error(std::string("work completion error: ") + ibv_wc_status_str(wc[k].status));
      got[wc[k].wr_id >> 8]++;
    }
  }
  if (p->trace && !j.write) p->t_end = mono_ns();
}

// W path: post layer l's RDMA WRITEs once the GPU says its K/V rows exist, then an 8-byte RDMA READ per QP as a
// durability fence (served after the earlier writes on that QP), then done = seq.
static void run_write_job(VerbsImpl* p, const Job& j) {
  const int nqp = (int)p->qps.size();
  int expect = 0, got = 0;
  std::vector<ibv_sge> sges;
  std::vector<ibv_send_wr> wrs;
  ibv_wc wc[64];
  auto drain = [&]() {
    int n = ibv_poll_cq(p->cq, 64, wc);
    if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
    for (int k = 0; k < n; ++k) {
      if (wc[k].status != IBV_WC_SUCCESS)
        throw std::runtime_error(std::string("work completion error: ") + ibv_wc_status_str(wc[k].status));
      got++;
    }
  };
  for (int l = 0; l < j.n_layers; ++l) {
    while (__atomic_load_n(&p->wflags_host[l], __ATOMIC_ACQUIRE) < j.seq) drain();
    int s0 = j.lp[l], s1 = j.lp[l + 1];
    for (int q = 0; q < nqp; ++q) {
      sges.clear();
      wrs.clear();
      for (int i = s0 + q; i < s1; i += nqp) {
        const int64_t* sg = &j.segs[4 * (size_t)i];
        ibv_sge g{};
        g.addr = (uint64_t)p->buf + (uint64_t)sg[1];
        g.length = (uint32_t)sg[2];
        g.lkey = p->mr->lkey;
        sges.push_back(g);
      }
      if (sges.empty()) continue;
      wrs.resize(sges.size());
      for (size_t k = 0; k < sges.size(); ++k) {
        const int64_t* sg = &j.segs[4 * (size_t)(s0 + q + (int)k * nqp)];
        ibv_send_wr& w = wrs[k];
        memset(&w, 0, sizeof(w));
        w.wr_id = ((uint64_t)l << 8) | (uint64_t)q;
        w.sg_list = &sges[k];
        w.num_sge = 1;
        w.opcode = IBV_WR_RDMA_WRITE;
        w.send_flags = (k + 1 == sges.size()) ? IBV_SEND_SIGNALED : 0;
        w.wr.rdma.remote_addr = p->raddr + (uint64_t)sg[0];
        w.wr.rdma.rkey = p->rkey;
        w.next = (k + 1 < sges.size()) ? &wrs[k + 1] : nullptr;
      }
      ibv_send_wr* bad = nullptr;
      int r = ibv_post_send(p->qps[q], wrs.data(), &bad);
      if (r) throw std::runtime_error("ibv_post_send (write) failed: " + std::string(strerror(r)));
      expect++;
    }
  }
  for (int q = 0; q < nqp; ++q) {   // fence: 8-byte read after the writes on each QP
    ibv_sge g{};
    g.addr = (uint64_t)p->buf + (uint64_t)j.fence_local + 8 * q;
    g.length = 8;
    g.lkey = p->mr->lkey;
    ibv_send_wr w{};
    w.wr_id = 0xFFFF00ull | (uint64_t)q;
    w.sg_list = &g;
    w.num_sge = 1;
    w.opcode = IBV_WR_RDMA_READ;
    w.send_flags = IBV_SEND_SIGNALED;
    w.wr.rdma.remote_addr = p->raddr + (uint64_t)j.fence_remote;
    w.wr.rdma.rkey = p->rkey;
    ibv_send_wr* bad = nullptr;
    int r = ibv_post_send(p->qps[q], &w, &bad);
    if (r) throw std::runtime_error("ibv_post_send (fence) failed: " + std::string(strerror(r)));
    expect++;
  }
  while (got < expect) drain();
  __atomic_store_n(&p->done_host[0], j.seq, __ATOMIC_RELEASE);
}

static void worker_main(VerbsImpl* p) {
  if (p->cpu_core >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(p->cpu_core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
  if (p->gpu) cudaSetDevice(p->cuda_dev);
  while (true) {
    Job j;
    if (p->spin.load(std::memory_order_relaxed)) {
      while (p->pending.load(std::memory_order_acquire) == 0) {
        if (p->stop_a.load(std::memory_order_relaxed) || !p->spin.load(std::memory_order_relaxed)) break;
        __builtin_ia32_pause();
      }
    }
    {
      std::unique_lock<std::mutex> lk(p->mu);
      p->cv.wait(lk, [p] { return p->stop || !p->jobs.empty(); });
      if (p->stop && p->jobs.empty()) return;
      j = std::move(p->jobs.front());
      p->jobs.pop_front();
      p->pending--;
    }
    if (p->trace && !j.write && !j.gated) p->t_pick = mono_ns();
    int64_t t0 = thread_ns();
    try {
      if (j.gated) run_write_job(p, j);
      else run_job(p, j);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lk(p->mu);
      p->err = e.what();
    }
    p->cpu_ns += thread_ns() - t0;
    p->done++;
    p->cv_idle.notify_all();
  }
}

VerbsKV::VerbsKV(const std::string& dev, int64_t bytes, bool gpu_buffer, int nqp, int max_layers, int flush_mode,
                 int cpu_core)
    : p_(new VerbsImpl) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  VCHECK(list, "ibv_get_device_list");
  for (int i = 0; i < n; ++i)
    if (dev == ibv_get_device_name(list[i])) p_->ctx = ibv_open_device(list[i]);
  ibv_free_device_list(list);
  if (!p_->ctx) throw std::runtime_error("verbs: device not found: " + dev);
  VCHECK(ibv_query_port(p_->ctx, p_->port, &p_->pattr) == 0, "ibv_query_port");
  ibv_query_gid(p_->ctx, p_->port, 0, &p_->gid);
  p_->pd = ibv_alloc_pd(p_->ctx);
  VCHECK(p_->pd, "ibv_alloc_pd");
  p_->cq = ibv_create_cq(p_->ctx, p_->sq_depth * nqp + 16, nullptr, nullptr, 0);
  VCHECK(p_->cq, "ibv_create_cq");
  for (int q = 0; q < nqp; ++q) {
    ibv_qp_init_attr ia{};
    ia.send_cq = p_->cq;
    ia.recv_cq = p_->cq;
    ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = p_->sq_depth;
    ia.cap.max_recv_wr = 1;
    ia.cap.max_send_sge = 1;
    ia.cap.max_recv_sge = 1;
    ibv_qp* qp = ibv_create_qp(p_->pd, &ia);
    VCHECK(qp, "ibv_create_qp");
    p_->qps.push_back(qp);
  }
  p_->bytes = (size_t)bytes;
  p_->gpu = gpu_buffer;
  if (gpu_buffer) {
    CUDACHK(cudaGetDevice(&p_->cuda_dev));
    CUDACHK(cudaMalloc(&p_->buf, p_->bytes));
    CUdevice cd;
    cuDeviceGet(&cd, p_->cuda_dev);
    cuDeviceGetAttribute(&p_->gdr_order, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, cd);
  } else {
    p_->buf = numa_host_alloc(p_->bytes, nic_numa(dev));
    memset(p_->buf, 0, p_->bytes);   // first touch on the bound node
  }
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  p_->mr = ibv_reg_mr(p_->pd, p_->buf, p_->bytes, access);
  VCHECK(p_->mr, "ibv_reg_mr");
  p_->max_layers = max_layers;
  p_->flush_mode = flush_mode;
  p_->cpu_core = cpu_core;
  if (gpu_buffer) {
    CUDACHK(cudaHostAlloc((void**)&p_->flags_host, max_layers * sizeof(uint32_t), cudaHostAllocMapped));
    memset(p_->flags_host, 0, max_layers * sizeof(uint32_t));
    CUDACHK(cudaHostGetDevicePointer((void**)&p_->flags_dev, p_->flags_host, 0));
    CUDACHK(cudaHostAlloc((void**)&p_->prog_host, max_layers * sizeof(uint64_t), cudaHostAllocMapped));
    memset(p_->prog_host, 0, max_layers * sizeof(uint64_t));
    CUDACHK(cudaHostGetDevicePointer((void**)&p_->prog_dev, p_->prog_host, 0));
    CUDACHK(cudaHostAlloc((void**)&p_->wflags_host, (max_layers + 1) * sizeof(uint32_t), cudaHostAllocMapped));
    memset(p_->wflags_host, 0, (max_layers + 1) * sizeof(uint32_t));
    CUDACHK(cudaHostGetDevicePointer((void**)&p_->wflags_dev, p_->wflags_host, 0));
    p_->done_host = p_->wflags_host + max_layers;
    p_->done_dev = p_->wflags_dev + max_layers;
  }
  p_->worker = std::thread(worker_main, p_);
}

std::string VerbsKV::local_info() const {
  std::ostringstream o;
  o << p_->pattr.lid << " " << (uint64_t)p_->buf << " " << p_->mr->rkey << " " << p_->qps.size();
  for (auto* q : p_->qps) o << " " << q->qp_num;
  for (int i = 0; i < 16; ++i) o << " " << (int)p_->gid.raw[i];
  return o.str();
}

void VerbsKV::connect(const std::string& remote) {
  std::istringstream in(remote);
  uint32_t lid, nq;
  uint64_t addr;
  in >> lid >> addr >> p_->rkey >> nq;
  p_->raddr = addr;
  if (nq != p_->qps.size()) throw std::runtime_error("verbs: QP count mismatch");
  std::vector<uint32_t> rq(nq);
  for (auto& x : rq) in >> x;
  ibv_gid rgid{};
  for (int i = 0; i < 16; ++i) {
    int v;
    in >> v;
    rgid.raw[i] = (uint8_t)v;
  }
  const bool roce = p_->pattr.link_layer == IBV_LINK_LAYER_ETHERNET;
  for (size_t q = 0; q < nq; ++q) {
    ibv_qp* qp = p_->qps[q];
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = p_->port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    VCHECK(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0, "INIT");
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = rq[q];
    a.rq_psn = 0;
    a.max_dest_rd_atomic = 16;
    a.min_rnr_timer = 12;
    a.ah_attr.dlid = (uint16_t)lid;
    a.ah_attr.sl = 0;
    a.ah_attr.port_num = p_->port;
    if (roce) {
      a.ah_attr.is_global = 1;
      a.ah_attr.grh.dgid = rgid;
      a.ah_attr.grh.sgid_index = 0;
      a.ah_attr.grh.hop_limit = 255;
    }
    VCHECK(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                     IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0, "RTR");
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = 0;
    a.max_rd_atomic = 16;
    VCHECK(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                     IBV_QP_MAX_QP_RD_ATOMIC) == 0, "RTS");
  }
}

void* VerbsKV::buffer() const { return p_->buf; }
int64_t VerbsKV::bytes() const { return (int64_t)p_->bytes; }
uint64_t VerbsKV::flags_device_ptr() const { return (uint64_t)p_->flags_dev; }
uint64_t VerbsKV::progress_device_ptr() const { return (uint64_t)p_->prog_dev; }

int VerbsKV::gdr_writes_ordering() const { return p_->gdr_order; }

static void submit(VerbsImpl* p, Job&& j) {
  {
    std::lock_guard<std::mutex> lk(p->mu);
    if (!p->err.empty()) throw std::runtime_error("verbs worker failed earlier: " + p->err);
    p->jobs.push_back(std::move(j));
    p->submitted++;
    p->pending++;
  }
  p->cv.notify_one();
}

void VerbsKV::set_spin(bool on) {
  p_->spin = on;
  p_->cv.notify_all();
}
void VerbsKV::set_inline_post(bool on) { p_->inline_post = on; }
void VerbsKV::set_trace(bool on) { p_->trace = on; }
std::vector<int64_t> VerbsKV::trace_read() const {
  std::vector<int64_t> r = {p_->t_post0, p_->t_post1, p_->t_pick, p_->t_cqe0, p_->t_end, p_->t_post2,
                           (int64_t)(p_->tev.size() / 3)};
  r.insert(r.end(), p_->tev.begin(), p_->tev.end());
  return r;
}

void VerbsKV::post_reads(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq) {
  if (n_layers > p_->max_layers) throw std::runtime_error("verbs: too many layers");
  Job j;
  j.n_layers = n_layers;
  j.lp.assign(layer_ptr, layer_ptr + n_layers + 1);
  j.segs.assign(segs, segs + 4 * (size_t)layer_ptr[n_layers]);
  j.seq = seq;
  // inline posting only when the worker is idle: its CQ then holds no completions of an earlier request, so the
  // per-layer completion counts cannot mix two requests
  if (p_->inline_post && p_->done.load() == p_->submitted) post_job(p_, j);
  submit(p_, std::move(j));
}

void VerbsKV::post_reads_tiled(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq,
                               int64_t tile_bytes, int first_layers) {
  if (n_layers > p_->max_layers) throw std::runtime_error("verbs: too many layers");
  if (p_->pend.active) throw std::runtime_error("verbs: post_rest() of the previous split request missing");
  Job j;
  j.n_layers = n_layers;
  j.lp.assign(layer_ptr, layer_ptr + n_layers + 1);
  j.segs.assign(segs, segs + 4 * (size_t)layer_ptr[n_layers]);
  j.seq = seq;
  j.tile_bytes = tile_bytes;
  if (p_->inline_post && p_->done.load() == p_->submitted) {
    if (first_layers > 0 && first_layers < n_layers) {
      // split: post the first layers now, the rest in post_rest() after the caller launched the graph
      if (p_->trace) trace_begin(p_);
      plan_tiled(j);
      int rr = 0;
      post_tiled_layers(p_, j.segs, j.lp, tile_bytes, 0, first_layers, rr);
      j.posted = true;
      VerbsImpl::Pending& q = p_->pend;
      q.segs = std::move(j.segs);
      q.lp = j.lp;
      q.tb = tile_bytes;
      q.l0 = first_layers;
      q.n_layers = n_layers;
      q.rr = rr;
      q.active = true;
      if (p_->trace) p_->t_post1 = mono_ns();
    } else {
      post_job(p_, j);
    }
  }
  submit(p_, std::move(j));
}

void VerbsKV::post_rest() {
  VerbsImpl::Pending& q = p_->pend;
  if (!q.active) return;
  post_tiled_layers(p_, q.segs, q.lp, q.tb, q.l0, q.n_layers, q.rr);
  q.active = false;
  if (p_->trace) p_->t_post2 = mono_ns();
}

void VerbsKV::post_writes(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq, int64_t fence_remote,
                          int64_t fence_local) {
  if (n_layers > p_->max_layers) throw std::runtime_error("verbs: too many layers");
  Job j;
  j.n_layers = n_layers;
  j.lp.assign(layer_ptr, layer_ptr + n_layers + 1);
  j.segs.assign(segs, segs + 4 * (size_t)layer_ptr[n_layers]);
  j.seq = seq;
  j.write = true;
  j.gated = true;
  j.fence_remote = fence_remote;
  j.fence_local = fence_local;
  submit(p_, std::move(j));
}
uint64_t VerbsKV::wflags_device_ptr() const { return (uint64_t)p_->wflags_dev; }
uint64_t VerbsKV::done_device_ptr() const { return (uint64_t)p_->done_dev; }

void VerbsKV::write_sync(const int64_t* segs, int n) {
  Job j;
  j.n_layers = 1;
  j.lp = {0, n};
  j.segs.assign(segs, segs + 4 * (size_t)n);
  j.write = true;
  submit(p_, std::move(j));
  wait_idle();
}

void VerbsKV::wait_idle() {
  std::unique_lock<std::mutex> lk(p_->mu);
  p_->cv_idle.wait(lk, [this] { return p_->done.load() >= p_->submitted || !p_->err.empty(); });
  if (!p_->err.empty()) throw std::runtime_error("verbs worker: " + p_->err);
}

double VerbsKV::worker_cpu_s() const { return p_->cpu_ns.load() / 1e9; }
int64_t VerbsKV::requests_done() const { return p_->done.load(); }

void VerbsKV::destroy() {
  if (!p_->ctx) return;
  {
    std::lock_guard<std::mutex> lk(p_->mu);
    p_->stop = true;
  }
  p_->stop_a = true;
  p_->cv.notify_all();
  if (p_->worker.joinable()) p_->worker.join();
  for (auto* q : p_->qps) ibv_destroy_qp(q);
  p_->qps.clear();
  if (p_->mr) ibv_dereg_mr(p_->mr);
  if (p_->cq) ibv_destroy_cq(p_->cq);
  if (p_->pd) ibv_dealloc_pd(p_->pd);
  ibv_close_device(p_->ctx);
  p_->ctx = nullptr;
  if (p_->buf) {
    if (p_->gpu) cudaFree(p_->buf);
    else munmap(p_->buf, p_->bytes);
  }
  if (p_->flags_host) cudaFreeHost(p_->flags_host);
  if (p_->prog_host) cudaFreeHost(p_->prog_host);
  if (p_->wflags_host) cudaFreeHost(p_->wflags_host);
  p_->wflags_host = nullptr;
  p_->buf = nullptr;
  p_->flags_host = nullptr;
}

VerbsKV::~VerbsKV() {
  try {
    destroy();
  } catch (...) {
  }
  delete p_;
}

}  // namespace hpoc
