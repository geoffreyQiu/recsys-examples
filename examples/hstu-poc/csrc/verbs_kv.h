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
  uint64_t progress_device_ptr() const;        // GPU-visible uint64 progress[max_layers] of tiled reads
  // per-tile waits: like post_reads, every segment split into tile_bytes chunks posted from its end backwards;
  // progress[l] = (seq << 32) | steps of layer l landed (step k = chunk k of every segment of the layer)
  // first_layers > 0 (split posting, inline mode only): post only layers [0, first_layers) now and hand the request to
  // the worker, which polls at once; post_rest() then posts the remaining layers from the calling thread (after the
  // caller has launched the request's graph). Without inline posting the worker posts everything as before.
  void post_reads_tiled(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq, int64_t tile_bytes,
                        int first_layers = 0);
  void post_rest();
  // segs: host array of 4 x int64 {remote offset, local offset, bytes, tag}; layer_ptr: [n_layers + 1]
  void post_reads(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq);
  // W path (6' reference for CPU verbs): layer l's writes are posted once the GPU stores seq into wflags[l]
  // (set_flag_seq kernel after the layer's K/V exist); after the last layer an 8-byte read per QP fences them,
  // then done = seq (the GPU can wait on it with wait_flags_seq(done_ptr, 0, seq)).
  void post_writes(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq, int64_t fence_remote,
                   int64_t fence_local);
  uint64_t wflags_device_ptr() const;
  uint64_t done_device_ptr() const;
  // writes (blocking helper): post RDMA WRITEs for segs (local -> remote) and wait
  void write_sync(const int64_t* segs, int n);
  void wait_idle();                            // block until the worker finished every posted request
  void set_spin(bool on);                      // worker busy-polls for jobs (no wake-up latency; one core busy)
  void set_inline_post(bool on);               // post_reads posts from the calling thread when no job is in flight
  // timeline trace of read requests (CLOCK_MONOTONIC ns): read after wait_idle(); returns
  // [post start, post end (first part when split), worker took the job, first completion, job end, rest posted (split),
  //  n events, (layer, value, time) x n]
  void set_trace(bool on);
  std::vector<int64_t> trace_read() const;
  // CPU time the worker thread spent (seconds, from CLOCK_THREAD_CPUTIME_ID), and requests served
  double worker_cpu_s() const;
  int64_t requests_done() const;
  int gdr_writes_ordering() const;             // CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
  void destroy();

 private:
  VerbsImpl* p_;
};

}  // namespace hpoc
