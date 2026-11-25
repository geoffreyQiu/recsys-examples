// GIN transport for remote KV (NCCL 2.32.7 device API). Plain C++ interface: no NCCL device types leak out,
// so the pybind translation unit never includes nccl_device.h.
//
// Process layout: world = 2N ranks. Ranks [0, N) are inference ranks (local buffer = GPU memory from
// ncclMemAlloc); ranks [N, 2N) are store ranks (local buffer = host memory, cuMemCreate HOST_NUMA on the
// NUMA node of the rank's GPU). Rank i pairs with rank i + N. One window covers each rank's local buffer, so
// a get names (remote window offset, local window offset) in the same window handle.
#pragma once
#include <cstdint>
#include <string>
#include <cuda_runtime.h>

namespace hpoc {

// One transfer: bytes from remote offset to local offset (get), or local to remote (put). 32 bytes.
struct Seg {
  int64_t remote;
  int64_t local;
  int64_t bytes;
  int64_t tag;   // free for callers (user id, layer, ...)
};

std::string gin_unique_id();   // 128-byte ncclUniqueId as a string

struct GinKVImpl;
class GinKV {
 public:
  // n_ctx: GIN contexts (use 0 for reads, 1 for writes); queue_depth >= operations in flight per context.
  GinKV(const std::string& uid, int rank, int nranks, int64_t local_bytes, bool local_on_host, int n_ctx,
        int queue_depth, bool strong_signals, int n_signals);
  ~GinKV();
  void* local_ptr() const;
  int64_t local_bytes() const;
  bool on_host() const;
  int rank() const;
  int nranks() const;

  // Read path (5'a). post_gets: one kernel posts every get; after each layer's gets, one flushAsync per layer
  // stores that layer's ticket (16 B) in tickets[layer]. The kernel returns without waiting.
  //   segs: device array of Seg; layer_ptr: device int array [n_layers + 1] (prefix offsets into segs).
  //   Reads use contexts [ctx0, ctx0 + n_ctx); tickets holds n_layers * n_ctx entries.
  void post_gets(const Seg* segs, const int* layer_ptr, int n_layers, void* tickets, int peer, int ctx0,
                 int n_ctx, uint64_t* t_post, cudaStream_t s);
  // wait_layer: one thread waits on layer's tickets (every read context); optional timestamp (globaltimer ns).
  void wait_layer(void* tickets, int layer, int ctx0, int n_ctx, uint64_t* t_done, cudaStream_t s);

  // Write path (6'). put_segs: put each segment local -> remote (no actions), then flushAsync into
  // *ticket (local completion: sources reusable). commit: strong signal (settles every earlier put on this
  // context to the peer) incrementing signal `sig` at the peer.
  void put_segs(const Seg* segs, int n_segs, void* ticket, int peer, int ctx, int threads, cudaStream_t s);
  void wait_ticket(void* ticket, int ctx, cudaStream_t s);
  void commit(int peer, int ctx, int sig, cudaStream_t s);
  // Store side: wait until signal `sig` reaches `value` (GPU kernel on the store rank's GPU).
  void wait_signal(int ctx, int sig, uint64_t value, cudaStream_t s);
  uint64_t read_signal_host(int ctx, int sig);   // launches a 1-thread kernel and syncs

  void destroy();

 private:
  GinKVImpl* p_;
};

}  // namespace hpoc
