// Raw GDAKI transport (P2b): DOCA GPUNetIO (NCCL's bundled BSD-3 copy, built as build/doca/libhpoc_doca.so)
// without NCCL. GPU-side RC QPs on the inference rank (WQEs and CQs in GPU memory, doorbell rung by GPU threads),
// connected to plain ibverbs QPs of a GPU-less store (VerbsKV with gpu_buffer = false). Same request shape as
// GIN: a post kernel issues every layer's gets and keeps one ticket per (layer, QP); a wait kernel per layer.
#pragma once
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "gin_kv.h"   // Seg

namespace hpoc {

struct RawImpl;
class RawGdaki {
 public:
  RawGdaki(const std::string& dev, int64_t bytes, int nqp, int sq_depth);
  ~RawGdaki();
  std::string local_info() const;          // VerbsKV's format: "lid addr rkey nq qpn... gid[16]"
  void connect(const std::string& remote_info);
  void* buffer() const;
  int64_t bytes() const;
  int nqp() const;
  // tickets: device array of n_layers * nqp uint64 (0 = nothing posted on that QP for the layer)
  // var (optional, device-visible pinned host int): the kernel uses segs + (*var) * var_stride (zero-copy index)
  void post_gets(const Seg* segs, const int* layer_ptr, int n_layers, uint64_t* tickets, uint64_t* t_post,
                 cudaStream_t s, const int* var = nullptr, int64_t var_stride = 0);
  void wait_layer(const uint64_t* tickets, int layer, uint64_t* t_done, cudaStream_t s);
  void poll(const uint64_t* tickets, int n_layers, const uint32_t* seq, uint32_t* ready, cudaStream_t s,
            unsigned long long* trace = nullptr);   // P4 fused; trace: timeline (kvutil.h)
  // per-tile waits: chunked gets posted from each segment's end; tickets [n_layers * max_steps * nqp]; the poller
  // publishes tsteps[l] = steps landed (0x7fffffff when the layer is complete)
  // (the post kernel also polls the completions and publishes tsteps, see raw_post_tiled_kernel)
  void post_gets_tiled(const Seg* segs, const int* layer_ptr, int n_layers, uint64_t* tickets, int max_steps,
                       int64_t tile_bytes, int* tsteps, cudaStream_t s, const int* var = nullptr,
                       int64_t var_stride = 0, unsigned long long* trace = nullptr, int version = 1,
                       uint64_t* prog = nullptr, uint64_t seq = 0, const uint32_t* seq_ptr = nullptr);
  // prog (v2 only): publish prog[l] = seq << 32 | steps (uint64, device memory) instead of tsteps[l], for a post kernel
  // launched outside the request's graph (per-tile v3: no reset of the progress words by the graph); seq_ptr: read the
  // sequence number from device memory instead (a post kernel captured in its own graph)
  // version 2: exclusive QP ownership (no locks/atomics in DOCA's posting calls), a separate publisher warp that
  // publishes each layer as soon as it is posted (raw_post_tiled2_kernel)
  void destroy();

 private:
  RawImpl* p_;
};

}  // namespace hpoc
