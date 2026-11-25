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
  void poll(const uint64_t* tickets, int n_layers, const uint32_t* seq, uint32_t* ready, cudaStream_t s);   // P4 fused
  void destroy();

 private:
  RawImpl* p_;
};

}  // namespace hpoc
