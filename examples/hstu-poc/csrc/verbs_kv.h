// CPU verbs transport (libibverbs, RC QPs, RDMA READ/WRITE posted by a CPU worker thread). No NCCL, and the
// store side needs no GPU: its pool is plain host memory (mmap + mbind to the NIC's NUMA node).
//
// Reader (inference) side: local buffer = GPU memory (cudaMalloc, registered through nvidia_peermem).
// post_request(segs, layer_ptr, seq) hands one request to the worker thread, which posts every layer's
// reads at once (segment i on QP i % nqp, the last read of a layer on each QP signaled), polls the CQ, and when
// layer l has completed on every QP (and every earlier layer has), optionally flushes GPUDirect RDMA writes and
// stores flags[l] = seq in host-mapped memory. The GPU waits with a spin kernel on flags[l] >= seq.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace hpoc {

struct VerbsImpl;
class VerbsKV {
 public:
  // dev: e.g. "mlx5_0". gpu_buffer: local memory on the current CUDA device (else host, NUMA of the NIC).
  // flush_mode: 0 none, 1 cuFlushGPUDirectRDMAWrites after each completed layer.
  VerbsKV(const std::string& dev, int64_t bytes, bool gpu_buffer, int nqp, int max_layers, int flush_mode,
          int cpu_core);
  ~VerbsKV();
  std::string local_info() const;              // serialized QP numbers, LID, address, rkey
  void connect(const std::string& remote_info);
  void* buffer() const;
  int64_t bytes() const;
  uint64_t flags_device_ptr() const;           // GPU-visible pointer to uint32 flags[max_layers]
  // segs: host array of 4 x int64 {remote offset, local offset, bytes, tag}; layer_ptr: [n_layers + 1]
  void post_reads(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq);
  // writes (6' reference): post RDMA WRITEs for segs (local -> remote) after the caller's event; blocking helper
  void write_sync(const int64_t* segs, int n);
  void wait_idle();                            // block until the worker finished every posted request
  // CPU time the worker thread spent (seconds, from CLOCK_THREAD_CPUTIME_ID), and requests served
  double worker_cpu_s() const;
  int64_t requests_done() const;
  int gdr_writes_ordering() const;             // CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
  void destroy();

 private:
  VerbsImpl* p_;
};

}  // namespace hpoc
