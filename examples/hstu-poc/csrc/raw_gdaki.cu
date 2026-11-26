// Raw GDAKI transport, see raw_gdaki.h. Host setup mirrors NCCL's gin_host_gdaki.cc (QP creation through the
// high-level DOCA GPUNetIO API, RTR/RTS through doca_verbs); device code calls doca_gpu_dev_verbs_get/wait.
#include "raw_gdaki.h"

#include <infiniband/verbs.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "doca_gpunetio.h"
#include "doca_gpunetio_device.h"
#include "doca_gpunetio_high_level.h"
#include "doca_verbs.h"
#include "kvutil.h"

namespace hpoc {

#define DOCA_OK(x)                                                                                    \
  do {                                                                                                \
    doca_error_t e_ = (x);                                                                            \
    if (e_ != DOCA_SUCCESS) throw std::runtime_error(std::string("DOCA ") + #x + ": " + std::to_string((int)e_)); \
  } while (0)
#define RCHECK(x, msg) \
  do {                 \
    if (!(x)) throw std::runtime_error(std::string("raw gdaki: ") + msg + ": " + strerror(errno)); \
  } while (0)
#define CUDA_OK2(cmd)                                                                                      \
  do {                                                                                                     \
    cudaError_t r_ = (cmd);                                                                                \
    if (r_ != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + #cmd + ": " + cudaGetErrorString(r_)); \
  } while (0)

struct RawImpl {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_mr* mr = nullptr;
  ibv_port_attr pattr{};
  ibv_gid gid{};
  void* buf = nullptr;
  size_t bytes = 0;
  doca_gpu_t* gdev = nullptr;
  doca_dev_t* ndev = nullptr;
  doca_verbs_ah_attr_t* ah = nullptr;
  std::vector<doca_gpu_verbs_qp_hl*> qps;
  doca_gpu_dev_verbs_qp** d_qps = nullptr;
  uint64_t raddr = 0;
  uint32_t rkey = 0;
};

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// timeline trace event (see kvutil.h for the layout)
__device__ __forceinline__ void rtrace_event(unsigned long long* tr, int& n, int l, long long v) {
  if (tr && n < TRACE_CAP) {
    unsigned long long* e = tr + TRACE_EV + 3 * n;
    e[0] = (unsigned long long)l;
    e[1] = (unsigned long long)v;
    e[2] = gtime();
    ++n;
  }
}

// warp w posts on QP w the segments j of each layer with j % nqp == w; lane 0 stores the layer's last ticket + 1
__global__ void raw_post_gets_kernel(doca_gpu_dev_verbs_qp** qps, int nqp, const Seg* segs, const int* lp,
                                     int n_layers, uint64_t rbase, uint32_t rkey_be, uint64_t lbase, uint32_t lkey_be,
                                     unsigned long long* tickets, uint64_t* t_post, const volatile int* var,
                                     long long var_stride) {
  if (t_post && threadIdx.x == 0) t_post[0] = gtime();
  // zero-copy request descriptor: the CPU writes the request's table index into pinned host memory, the kernel reads
  // it once (no per-request copy of the segment table)
  __shared__ long long s_off;
  if (threadIdx.x == 0) s_off = var ? (long long)(*var) * var_stride : 0;
  __syncthreads();
  segs += s_off;
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  doca_gpu_dev_verbs_qp* qp = qps[w];
  for (int l = 0; l < n_layers; ++l) {
    unsigned long long mx = 0;
    for (int i = lp[l] + w + nqp * lane; i < lp[l + 1]; i += nqp * 32) {
      Seg sg = segs[i];
      doca_gpu_dev_verbs_addr ra, la, da{};
      ra.addr = rbase + (uint64_t)sg.remote;
      ra.key = rkey_be;
      la.addr = lbase + (uint64_t)sg.local;
      la.key = lkey_be;
      doca_gpu_dev_verbs_ticket_t t;
      doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, ra, la, (size_t)sg.bytes, da, &t);
      mx = max(mx, (unsigned long long)t + 1ull);
    }
    for (int o = 16; o; o >>= 1) mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) tickets[l * nqp + w] = mx;
    __syncwarp();
  }
  if (t_post) {
    __syncthreads();
    if (threadIdx.x == 0) t_post[1] = gtime();
  }
}

__global__ void raw_wait_kernel(doca_gpu_dev_verbs_qp** qps, int nqp, const unsigned long long* tickets, int layer,
                                uint64_t* t_done) {
  asm volatile("griddepcontrol.launch_dependents;");   // P4 PDL
  for (int q = 0; q < nqp; ++q) {
    unsigned long long t = tickets[layer * nqp + q];
    if (t) doca_gpu_dev_verbs_wait<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qps[q], (doca_gpu_dev_verbs_ticket_t)(t - 1));
  }
  if (t_done) t_done[layer] = gtime();
}

__global__ void raw_poll_kernel(doca_gpu_dev_verbs_qp** qps, int nqp, const unsigned long long* tickets, int n_layers,
                                const uint32_t* seq, uint32_t* ready, unsigned long long* tr) {
  if (tr) tr[1] = gtime();
  int ne = 0;
  const uint32_t s = *seq;
  for (int l = 0; l < n_layers; ++l) {
    for (int q = 0; q < nqp; ++q) {
      unsigned long long t = tickets[l * nqp + q];
      if (t) doca_gpu_dev_verbs_wait<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qps[q], (doca_gpu_dev_verbs_ticket_t)(t - 1));
    }
    __threadfence();
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(ready + l), "r"(s) : "memory");
    rtrace_event(tr, ne, l, 0x7fffffff);
  }
  if (tr) tr[0] = ne;
}

// Per-tile waits: every segment split into chunks of tb bytes posted from its END backwards (the attention consumes
// key blocks last to first); step k of layer l = chunk k of every segment of the layer, the chunk of segment j going
// to QP (j + k) % nqp; tickets[(l * max_steps + k) * nqp + w] = last WQE of step k on QP w + 1 (0 = none).
// Batched posting, warp-parallel: per (layer, QP) the QP's warp counts the layer's chunk reads on that QP, lane 0
// reserves their slots at once, the lanes write the WQEs in parallel (one per lane per round, in step order), and lane
// 0 marks them ready and rings one doorbell (one reservation, one ready mark and one doorbell per layer and QP).
// Chunk (segment j, step k) goes to QP (j + k) % nqp; a step's ticket = its last read's WQE index + 1 on each QP.
// The same kernel then polls the completions and publishes tsteps[l] = steps landed (0x7fffffff once the layer is
// complete): the publisher is resident from the request's start, before any attention CTA spins on its progress (a
// separate poller launched after the posts can find every SM taken by spinning attention CTAs and never run).
__global__ void raw_post_tiled_kernel(doca_gpu_dev_verbs_qp** qps, int nqp, const Seg* segs, const int* lp,
                                      int n_layers, int max_steps, long long tb, uint64_t rbase, uint32_t rkey_be,
                                      uint64_t lbase, uint32_t lkey_be, unsigned long long* tickets,
                                      const volatile int* var, long long var_stride, int* tsteps,
                                      unsigned long long* tr) {
  if (tr && threadIdx.x == 0) tr[1] = gtime();
  __shared__ long long s_off;
  if (threadIdx.x == 0) s_off = var ? (long long)(*var) * var_stride : 0;
  __syncthreads();
  segs += s_off;
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  doca_gpu_dev_verbs_qp* qp = qps[w];
  const enum doca_gpu_dev_verbs_wqe_ctrl_flags flags =
      doca_gpu_dev_verbs_get_ctrl_flags<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
  for (int l = 0; l < n_layers; ++l) {
    const int s0 = lp[l], n = lp[l + 1] - s0;
    const int per = (n + nqp - 1) / nqp;   // candidate chunks of this QP per step
    const int total = max_steps * per;     // candidate t = k * per + i: segment j = (w - k) mod nqp + nqp * i, step k
    unsigned long long* tk = tickets + (long long)l * max_steps * nqp + w;
    for (int k = lane; k < max_steps; k += 32) tk[(long long)k * nqp] = 0;
    int cnt = 0;
    for (int t = lane; t < total; t += 32) {
      const int k = t / per, j = ((w - k) % nqp + nqp) % nqp + nqp * (t % per);
      if (j < n && segs[s0 + j].bytes > (long long)k * tb) ++cnt;
    }
    for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
    if (cnt == 0) continue;
    unsigned long long base = 0;
    if (lane == 0)
      base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                                 DOCA_GPUNETIO_VERBS_QP_SQ>(qp, cnt,
                                                                            DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
    base = __shfl_sync(0xffffffffu, base, 0);
    __syncwarp();   // the zeroed tickets before the lanes' atomicMax below
    unsigned long long idx = base;
    for (int t0 = 0; t0 < total; t0 += 32) {
      const int t = t0 + lane;
      int k = 0;
      bool has = false;
      Seg sg{};
      if (t < total) {
        k = t / per;
        const int j = ((w - k) % nqp + nqp) % nqp + nqp * (t % per);
        if (j < n) {
          sg = segs[s0 + j];
          has = sg.bytes > (long long)k * tb;
        }
      }
      const unsigned m = __ballot_sync(0xffffffffu, has);
      if (has) {
        const unsigned long long my = idx + __popc(m & ((1u << lane) - 1u));
        const long long done = (long long)k * tb;
        const long long len = min(tb, (long long)sg.bytes - done);
        const long long off = (long long)sg.bytes - done - len;
        struct doca_gpu_dev_verbs_wqe* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, my);
        doca_gpu_dev_verbs_wqe_prepare_read(qp, wqe, my, flags, rbase + (uint64_t)(sg.remote + off), rkey_be,
                                            lbase + (uint64_t)(sg.local + off), lkey_be, (size_t)len);
        atomicMax(&tk[(long long)k * nqp], my + 1);
      }
      idx += __popc(m);
    }
    __syncwarp();   // every lane's WQE is written; mark_wqes_ready's release fence covers them
    if (lane == 0) {
      doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, idx - 1);
      doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD>(
          qp, idx);
      if (tr && l < 16 && w < 8) tr[TRACE_DB + 8 * l + w] = gtime();
    }
  }
  __syncthreads();   // every QP's tickets are written
  if (threadIdx.x != 0 || tsteps == nullptr) return;
  int ne = 0;
  for (int l = 0; l < n_layers; ++l) {
    for (int k = 0; k < max_steps; ++k) {
      bool any = false;
      for (int q = 0; q < nqp; ++q) {
        const unsigned long long t = tickets[((long long)l * max_steps + k) * nqp + q];
        if (t) {
          any = true;
          doca_gpu_dev_verbs_wait<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qps[q], (doca_gpu_dev_verbs_ticket_t)(t - 1));
        }
      }
      if (!any) break;
      __threadfence();
      asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(tsteps + l), "r"(k + 1) : "memory");
      rtrace_event(tr, ne, l, k + 1);
    }
    __threadfence();
    asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(tsteps + l), "r"(0x7fffffff) : "memory");
    rtrace_event(tr, ne, l, 0x7fffffff);
  }
  if (tr) tr[0] = ne;
}

// Per-tile waits, v2. Same chunk layout and tickets as raw_post_tiled_kernel, three changes:
// - each queue pair is owned by one warp for the whole request, so DOCA's EXCLUSIVE sharing mode applies (plain loads
//   and stores instead of the lock, CAS and atomics of GPU mode in reserve / mark-ready / submit), and the slot
//   availability check is skipped (the previous request's reads completed and were polled before this one was posted;
//   a request posts at most 2048 reads on 4 queues of 4096 slots);
// - a separate publisher warp (warp nqp, lane 0) polls the completions and publishes tsteps[l] as soon as every QP warp
//   has posted layer l (a per-layer counter in shared memory), instead of after the whole request is posted;
// - the step tickets are the per-step maximum over the warp (one store per step and QP, no global atomics).
constexpr auto RAW_EXCL = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
// progress of a tiled request: int32 steps (graph-launched post kernel; reset by the graph) or a sequence-tagged
// 64-bit word seq << 32 | steps (post kernel launched outside the request graph; the graph resets nothing)
__device__ __forceinline__ void raw_publish(int* tsteps, unsigned long long* prog, unsigned long long seq, int l, int v) {
  if (prog)
    asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(prog + l), "l"((seq << 32) | (unsigned)v) : "memory");
  else
    asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(tsteps + l), "r"(v) : "memory");
}

// One tiled request, v2 (see raw_post_tiled2_kernel): warps 0..nqp-1 post, warp nqp publishes. s_posted: shared,
// zeroed by the caller before a __syncthreads.
__device__ __forceinline__ void raw_tiled_request(doca_gpu_dev_verbs_qp** qps, int nqp, const Seg* segs, const int* lp,
                                                  int n_layers, int max_steps, long long tb, uint64_t rbase,
                                                  uint32_t rkey_be, uint64_t lbase, uint32_t lkey_be,
                                                  unsigned long long* tickets, int* s_posted, int* tsteps,
                                                  unsigned long long* prog, unsigned long long seq,
                                                  unsigned long long* tr) {
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (w < nqp) {
    doca_gpu_dev_verbs_qp* qp = qps[w];
    const enum doca_gpu_dev_verbs_wqe_ctrl_flags flags =
        doca_gpu_dev_verbs_get_ctrl_flags<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
    const uint32_t opt = DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK;
    for (int l = 0; l < n_layers; ++l) {
      const int s0 = lp[l], n = lp[l + 1] - s0;
      const int per = (n + nqp - 1) / nqp;
      const int total = max_steps * per;
      unsigned long long* tk = tickets + (long long)l * max_steps * nqp + w;
      for (int k = lane; k < max_steps; k += 32) tk[(long long)k * nqp] = 0;   // steps without a chunk on this QP
      int cnt = 0;
      for (int t = lane; t < total; t += 32) {
        const int k = t / per, j = ((w - k) % nqp + nqp) % nqp + nqp * (t % per);
        if (j < n && segs[s0 + j].bytes > (long long)k * tb) ++cnt;
      }
      for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, o);
      __syncwarp();                                     // zeroed tickets before the step tickets below
      if (cnt > 0) {
        unsigned long long base = 0;
        if (lane == 0)
          base = doca_gpu_dev_verbs_reserve_wq_slots<RAW_EXCL, DOCA_GPUNETIO_VERBS_QP_SQ>(qp, cnt, opt);
        base = __shfl_sync(0xffffffffu, base, 0);
        unsigned long long idx = base;
        for (int t0 = 0; t0 < total; t0 += 32) {
          const int t = t0 + lane;
          int k = 0;
          bool has = false;
          Seg sg{};
          if (t < total) {
            k = t / per;
            const int j = ((w - k) % nqp + nqp) % nqp + nqp * (t % per);
            if (j < n) {
              sg = segs[s0 + j];
              has = sg.bytes > (long long)k * tb;
            }
          }
          const unsigned m = __ballot_sync(0xffffffffu, has);
          const unsigned long long my = idx + __popc(m & ((1u << lane) - 1u));
          if (has) {
            const long long done = (long long)k * tb;
            const long long len = min(tb, (long long)sg.bytes - done);
            const long long off = (long long)sg.bytes - done - len;
            struct doca_gpu_dev_verbs_wqe* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, my);
            doca_gpu_dev_verbs_wqe_prepare_read(qp, wqe, my, flags, rbase + (uint64_t)(sg.remote + off), rkey_be,
                                                lbase + (uint64_t)(sg.local + off), lkey_be, (size_t)len);
          }
          // the step's ticket = its last WQE on this QP + 1: lanes hold chunks in step order, so a valid lane is the
          // last of its step in this round when the next valid lane has another step (a step spanning two rounds is
          // overwritten by the later round)
          const unsigned higher = (lane == 31) ? 0u : (m & (0xffffffffu << (lane + 1)));
          const int knext = __shfl_sync(0xffffffffu, k, higher ? (__ffs(higher) - 1) : lane);
          if (has && (higher == 0 || knext != k)) tk[(long long)k * nqp] = my + 1;
          idx += __popc(m);
        }
        __syncwarp();
        if (lane == 0) {
          doca_gpu_dev_verbs_mark_wqes_ready<RAW_EXCL>(qp, base, idx - 1);
          doca_gpu_dev_verbs_submit<RAW_EXCL, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD>(qp, idx);
        }
      }
      __syncwarp();
      if (lane == 0) {
        if (tr && l < 16 && w < 8) tr[TRACE_DB + 8 * l + w] = gtime();
        __threadfence_block();                           // tickets (global) before the shared-memory flag
        atomicAdd(&s_posted[l], 1);
      }
    }
    return;
  }
  if (w != nqp || lane != 0 || (tsteps == nullptr && prog == nullptr)) return;
  int ne = 0;
  for (int l = 0; l < n_layers; ++l) {
    while (*(volatile int*)&s_posted[l] < nqp) {
    }
    __threadfence_block();
    for (int k = 0; k < max_steps; ++k) {
      bool any = false;
      for (int q = 0; q < nqp; ++q) {
        const unsigned long long t = tickets[((long long)l * max_steps + k) * nqp + q];
        if (t) {
          any = true;
          doca_gpu_dev_verbs_wait<RAW_EXCL>(qps[q], (doca_gpu_dev_verbs_ticket_t)(t - 1));
        }
      }
      if (!any) break;
      __threadfence();
      raw_publish(tsteps, prog, seq, l, k + 1);
      rtrace_event(tr, ne, l, k + 1);
    }
    __threadfence();
    raw_publish(tsteps, prog, seq, l, 0x7fffffff);
    rtrace_event(tr, ne, l, 0x7fffffff);
  }
  if (tr) tr[0] = ne;
}

__global__ void raw_post_tiled2_kernel(doca_gpu_dev_verbs_qp** qps, int nqp, const Seg* segs, const int* lp,
                                       int n_layers, int max_steps, long long tb, uint64_t rbase, uint32_t rkey_be,
                                       uint64_t lbase, uint32_t lkey_be, unsigned long long* tickets,
                                       const volatile int* var, long long var_stride, int* tsteps,
                                       unsigned long long* tr, unsigned long long* prog, unsigned long long seq,
                                       const uint32_t* seq_ptr) {
  __shared__ long long s_off;
  __shared__ unsigned long long s_seq;
  __shared__ int s_posted[64];
  if (tr && threadIdx.x == 0) tr[1] = gtime();
  if (threadIdx.x < 64) s_posted[threadIdx.x] = 0;
  if (threadIdx.x == 0) {
    s_off = var ? (long long)(*var) * var_stride : 0;
    s_seq = seq_ptr ? (unsigned long long)*seq_ptr : seq;   // seq_ptr: the request counter of the post graph
  }
  __syncthreads();
  raw_tiled_request(qps, nqp, segs + s_off, lp, n_layers, max_steps, tb, rbase, rkey_be, lbase, lkey_be, tickets,
                    s_posted, tsteps, prog, s_seq, tr);
}

RawGdaki::RawGdaki(const std::string& dev, int64_t bytes, int nqp, int sq_depth) : p_(new RawImpl) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  RCHECK(list, "ibv_get_device_list");
  for (int i = 0; i < n; ++i)
    if (dev == ibv_get_device_name(list[i])) p_->ctx = ibv_open_device(list[i]);
  ibv_free_device_list(list);
  if (!p_->ctx) throw std::runtime_error("raw gdaki: device not found: " + dev);
  RCHECK(ibv_query_port(p_->ctx, 1, &p_->pattr) == 0, "ibv_query_port");
  ibv_query_gid(p_->ctx, 1, 0, &p_->gid);
  p_->pd = ibv_alloc_pd(p_->ctx);
  RCHECK(p_->pd, "ibv_alloc_pd");
  int cdev = 0;
  CUDA_OK2(cudaGetDevice(&cdev));
  p_->bytes = (size_t)bytes;
  CUDA_OK2(cudaMalloc(&p_->buf, p_->bytes));
  p_->mr = ibv_reg_mr(p_->pd, p_->buf, p_->bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  RCHECK(p_->mr, "ibv_reg_mr (GPU memory, nvidia_peermem)");
  char bus[32] = {0};
  CUDA_OK2(cudaDeviceGetPCIBusId(bus, sizeof(bus), cdev));
  DOCA_OK(doca_gpu_create(bus, &p_->gdev));
  DOCA_OK(doca_verbs_dev_open(p_->pd, &p_->ndev));
  DOCA_OK(doca_verbs_ah_attr_create(p_->ndev, &p_->ah));
  std::vector<doca_gpu_dev_verbs_qp*> dq;
  for (int q = 0; q < nqp; ++q) {
    doca_gpu_verbs_qp_init_attr_hl a;
    memset(&a, 0, sizeof(a));
    a.gpu_dev = p_->gdev;
    a.net_dev = p_->ndev;
    a.ibpd = p_->pd;
    a.sq_nwqe = (uint16_t)sq_depth;
    a.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
    a.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
    a.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
    a.cq_collapsed = false;
    a.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
    a.ordering_semantic = DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA;
    doca_gpu_verbs_qp_hl* qp = nullptr;
    DOCA_OK(doca_gpu_verbs_create_qp_hl(&a, &qp));
    p_->qps.push_back(qp);
    doca_gpu_dev_verbs_qp* d = nullptr;
    DOCA_OK(doca_gpu_verbs_get_qp_dev(qp->qp_gverbs, &d));
    dq.push_back(d);
  }
  CUDA_OK2(cudaMalloc((void**)&p_->d_qps, nqp * sizeof(doca_gpu_dev_verbs_qp*)));
  CUDA_OK2(cudaMemcpy(p_->d_qps, dq.data(), nqp * sizeof(doca_gpu_dev_verbs_qp*), cudaMemcpyHostToDevice));
}

std::string RawGdaki::local_info() const {
  std::ostringstream o;
  o << p_->pattr.lid << " " << (uint64_t)p_->buf << " " << p_->mr->rkey << " " << p_->qps.size();
  for (auto* q : p_->qps) {
    uint32_t qpn = 0;
    DOCA_OK(doca_verbs_qp_get_qpn(q->qp, &qpn));
    o << " " << qpn;
  }
  for (int i = 0; i < 16; ++i) o << " " << (int)p_->gid.raw[i];
  return o.str();
}

void RawGdaki::connect(const std::string& remote) {
  std::istringstream in(remote);
  uint32_t lid, nq;
  uint64_t addr;
  in >> lid >> addr >> p_->rkey >> nq;
  p_->raddr = addr;
  if (nq != p_->qps.size()) throw std::runtime_error("raw gdaki: QP count mismatch");
  std::vector<uint32_t> rq(nq);
  for (auto& x : rq) in >> x;
  doca_verbs_gid rgid;
  for (int i = 0; i < 16; ++i) {
    int v;
    in >> v;
    rgid.raw[i] = (uint8_t)v;
  }
  DOCA_OK(doca_verbs_ah_attr_set_addr_type(p_->ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
  DOCA_OK(doca_verbs_ah_attr_set_hop_limit(p_->ah, 0));
  DOCA_OK(doca_verbs_ah_attr_set_gid(p_->ah, rgid));
  DOCA_OK(doca_verbs_ah_attr_set_dlid(p_->ah, lid));
  for (size_t q = 0; q < nq; ++q) {
    doca_verbs_qp_t* qp = p_->qps[q]->qp;
    doca_verbs_qp_attr_t* at = nullptr;
    DOCA_OK(doca_verbs_qp_attr_create(&at));
    DOCA_OK(doca_verbs_qp_attr_set_path_mtu(at, DOCA_VERBS_MTU_SIZE_4K_BYTES));
    DOCA_OK(doca_verbs_qp_attr_set_rq_psn(at, 0));
    DOCA_OK(doca_verbs_qp_attr_set_sq_psn(at, 0));
    DOCA_OK(doca_verbs_qp_attr_set_port_num(at, 1));
    DOCA_OK(doca_verbs_qp_attr_set_ack_timeout(at, 20));
    DOCA_OK(doca_verbs_qp_attr_set_retry_cnt(at, 7));
    DOCA_OK(doca_verbs_qp_attr_set_rnr_retry(at, 7));
    DOCA_OK(doca_verbs_qp_attr_set_min_rnr_timer(at, 12));
    DOCA_OK(doca_verbs_qp_attr_set_next_state(at, DOCA_VERBS_QP_STATE_INIT));
    DOCA_OK(doca_verbs_qp_attr_set_allow_remote_write(at, 1));
    DOCA_OK(doca_verbs_qp_attr_set_allow_remote_read(at, 1));
    DOCA_OK(doca_verbs_qp_attr_set_atomic_mode(at, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
    DOCA_OK(doca_verbs_qp_attr_set_ah_attr(at, p_->ah));
    DOCA_OK(doca_verbs_qp_attr_set_dest_qp_num(at, rq[q]));
    DOCA_OK(doca_verbs_qp_attr_set_pkey_index(at, 0));
    DOCA_OK(doca_verbs_qp_modify(qp, at, DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                           DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
                                           DOCA_VERBS_QP_ATTR_PORT_NUM));
    DOCA_OK(doca_verbs_qp_attr_set_max_dest_rd_atomic(at, 16));
    DOCA_OK(doca_verbs_qp_attr_set_next_state(at, DOCA_VERBS_QP_STATE_RTR));
    DOCA_OK(doca_verbs_qp_modify(qp, at, DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                           DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                           DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                           DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC | DOCA_VERBS_QP_ATTR_ATOMIC_MODE));
    DOCA_OK(doca_verbs_qp_attr_set_max_rd_atomic(at, 16));
    DOCA_OK(doca_verbs_qp_attr_set_next_state(at, DOCA_VERBS_QP_STATE_RTS));
    DOCA_OK(doca_verbs_qp_modify(qp, at, DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                           DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                           DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));
    DOCA_OK(doca_verbs_qp_attr_destroy(at));
  }
}

void* RawGdaki::buffer() const { return p_->buf; }
int64_t RawGdaki::bytes() const { return (int64_t)p_->bytes; }
int RawGdaki::nqp() const { return (int)p_->qps.size(); }

void RawGdaki::post_gets(const Seg* segs, const int* layer_ptr, int n_layers, uint64_t* tickets, uint64_t* t_post,
                         cudaStream_t s, const int* var, int64_t var_stride) {
  const int nq = (int)p_->qps.size();
  raw_post_gets_kernel<<<1, 32 * nq, 0, s>>>(p_->d_qps, nq, segs, layer_ptr, n_layers, p_->raddr,
                                             htobe32(p_->rkey), (uint64_t)p_->buf, htobe32(p_->mr->lkey),
                                             (unsigned long long*)tickets, t_post, var, (long long)var_stride);
  CUDA_OK2(cudaGetLastError());
}

void RawGdaki::wait_layer(const uint64_t* tickets, int layer, uint64_t* t_done, cudaStream_t s) {
  raw_wait_kernel<<<1, 1, 0, s>>>(p_->d_qps, (int)p_->qps.size(), (const unsigned long long*)tickets, layer, t_done);
  CUDA_OK2(cudaGetLastError());
}

void RawGdaki::poll(const uint64_t* tickets, int n_layers, const uint32_t* seq, uint32_t* ready, cudaStream_t s,
                    unsigned long long* trace) {
  raw_poll_kernel<<<1, 1, 0, s>>>(p_->d_qps, (int)p_->qps.size(), (const unsigned long long*)tickets, n_layers, seq, ready,
                                  trace);
  CUDA_OK2(cudaGetLastError());
}

void RawGdaki::post_gets_tiled(const Seg* segs, const int* layer_ptr, int n_layers, uint64_t* tickets, int max_steps,
                               int64_t tile_bytes, int* tsteps, cudaStream_t s, const int* var, int64_t var_stride,
                               unsigned long long* trace, int version, uint64_t* prog, uint64_t seq,
                               const uint32_t* seq_ptr) {
  const int nq = (int)p_->qps.size();
  if (version == 2) {   // + one publisher warp; prog: sequence-tagged progress instead of tsteps
    raw_post_tiled2_kernel<<<1, 32 * (nq + 1), 0, s>>>(p_->d_qps, nq, segs, layer_ptr, n_layers, max_steps,
                                                       (long long)tile_bytes, p_->raddr, htobe32(p_->rkey),
                                                       (uint64_t)p_->buf, htobe32(p_->mr->lkey),
                                                       (unsigned long long*)tickets, var, (long long)var_stride, tsteps,
                                                       trace, (unsigned long long*)prog, (unsigned long long)seq,
                                                       seq_ptr);
    CUDA_OK2(cudaGetLastError());
    return;
  }
  raw_post_tiled_kernel<<<1, 32 * nq, 0, s>>>(p_->d_qps, nq, segs, layer_ptr, n_layers, max_steps,
                                              (long long)tile_bytes, p_->raddr, htobe32(p_->rkey), (uint64_t)p_->buf,
                                              htobe32(p_->mr->lkey), (unsigned long long*)tickets, var,
                                              (long long)var_stride, tsteps, trace);
  CUDA_OK2(cudaGetLastError());
}

void RawGdaki::destroy() {
  if (!p_->ctx) return;
  cudaDeviceSynchronize();
  for (auto* q : p_->qps) doca_gpu_verbs_destroy_qp_hl(q);
  p_->qps.clear();
  if (p_->ah) doca_verbs_ah_attr_destroy(p_->ah);
  if (p_->ndev) doca_verbs_dev_close(p_->ndev);
  if (p_->gdev) doca_gpu_destroy(p_->gdev);
  if (p_->mr) ibv_dereg_mr(p_->mr);
  if (p_->pd) ibv_dealloc_pd(p_->pd);
  ibv_close_device(p_->ctx);
  p_->ctx = nullptr;
  if (p_->buf) cudaFree(p_->buf);
  if (p_->d_qps) cudaFree(p_->d_qps);
  p_->buf = nullptr;
  p_->d_qps = nullptr;
}

RawGdaki::~RawGdaki() {
  try {
    destroy();
  } catch (...) {
  }
  delete p_;
}

}  // namespace hpoc
