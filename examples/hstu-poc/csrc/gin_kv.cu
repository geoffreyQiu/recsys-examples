// GIN transport (see gin_kv.h). Built against NCCL 2.32.7 (header-only device API, link -lnccl).
#include "gin_kv.h"

#include <cuda.h>
#include <nccl.h>
#include <nccl_device.h>

#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#define NCCL_OK(cmd)                                                                              \
  do {                                                                                            \
    ncclResult_t r_ = (cmd);                                                                      \
    if (r_ != ncclSuccess)                                                                        \
      throw std::runtime_error(std::string("NCCL ") + #cmd + ": " + ncclGetErrorString(r_) + " / " + \
                               ncclGetLastError(nullptr));                                        \
  } while (0)
#define CU_OK(cmd)                                                                \
  do {                                                                            \
    CUresult r_ = (cmd);                                                          \
    if (r_ != CUDA_SUCCESS) {                                                     \
      const char* s_ = nullptr;                                                   \
      cuGetErrorString(r_, &s_);                                                  \
      throw std::runtime_error(std::string("CU ") + #cmd + ": " + (s_ ? s_ : "?")); \
    }                                                                             \
  } while (0)
#define CUDA_OK(cmd)                                                                                       \
  do {                                                                                                     \
    cudaError_t r_ = (cmd);                                                                                \
    if (r_ != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + #cmd + ": " + cudaGetErrorString(r_)); \
  } while (0)

static_assert(sizeof(ncclGinRequest_t) == 16, "ticket size");
static_assert(sizeof(hpoc::Seg) == 32, "seg size");

namespace hpoc {

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// ---------------------------------------------------------------- kernels
// Remote = host (HOST_NUMA) and local = device -> SegmentMixed. Used for both directions.
// Reads are spread over n_ctx GIN contexts (one QP each per peer): warp w posts on context ctx0 + w the
// segments j of each layer with j % n_ctx == w, then its lane 0 stores that context's ticket for the layer in
// tickets[layer * n_ctx + w]. One QP tops out near 42 GB/s for RDMA READ on Eos (P0-a); two reach line rate.
__global__ void gin_post_gets_kernel(ncclDevComm dcomm, ncclWindow_t win, int ctx0, int n_ctx, int peer,
                                     const Seg* segs, const int* layer_ptr, int n_layers, ncclGinRequest_t* tickets,
                                     uint64_t* t_post) {
  if (t_post && threadIdx.x == 0) t_post[0] = globaltimer();
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  ncclGin gin(dcomm, ctx0 + w);
  ncclTeam team = ncclTeamWorld(dcomm);
  for (int l = 0; l < n_layers; ++l) {
    const int s0 = layer_ptr[l], s1 = layer_ptr[l + 1];
    for (int i = s0 + w + n_ctx * lane; i < s1; i += n_ctx * 32) {
      Seg sg = segs[i];
      gin.get(team, peer, win, (size_t)sg.remote, win, (size_t)sg.local, (size_t)sg.bytes, ncclCoopThread{},
              ncclGin_None{}, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
    }
    __syncwarp();   // the warp's gets for layer l are posted before its ticket is taken
    if (lane == 0) gin.flushAsync(team, peer, &tickets[l * n_ctx + w]);
  }
  if (t_post) {
    __syncthreads();
    if (threadIdx.x == 0) t_post[1] = globaltimer();
  }
}

__global__ void gin_wait_kernel(ncclDevComm dcomm, int ctx0, int n_ctx, ncclGinRequest_t* tickets, int layer,
                                uint64_t* t_done) {
  asm volatile("griddepcontrol.launch_dependents;");   // P4 PDL: let the next kernel's prologue start now
  for (int c = 0; c < n_ctx; ++c) {
    ncclGin gin(dcomm, ctx0 + c);
    ncclGinRequest_t r = tickets[layer * n_ctx + c];
    gin.wait(r);
  }
  if (t_done) t_done[layer] = globaltimer();
}

__device__ __forceinline__ void store_release(uint32_t* p, uint32_t v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// P4 fused: one thread waits on each layer's tickets in order and publishes ready[l] = seq for the attention kernel
__global__ void gin_poll_kernel(ncclDevComm dcomm, int ctx0, int n_ctx, ncclGinRequest_t* tickets, int n_layers,
                                const uint32_t* seq, uint32_t* ready) {
  const uint32_t s = *seq;
  for (int l = 0; l < n_layers; ++l) {
    for (int c = 0; c < n_ctx; ++c) {
      ncclGin gin(dcomm, ctx0 + c);
      ncclGinRequest_t r = tickets[l * n_ctx + c];
      gin.wait(r);
    }
    __threadfence();
    store_release(ready + l, s);
  }
}

__global__ void gin_put_kernel(ncclDevComm dcomm, ncclWindow_t win, int ctx, int peer, const Seg* segs, int n,
                               ncclGinRequest_t* ticket) {
  ncclGin gin(dcomm, ctx);
  ncclTeam team = ncclTeamWorld(dcomm);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    Seg sg = segs[i];
    gin.put(team, peer, win, (size_t)sg.remote, win, (size_t)sg.local, (size_t)sg.bytes, ncclGin_None{},
            ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_device,
            cuda::thread_scope_device, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
  }
  __syncthreads();
  if (threadIdx.x == 0 && ticket) gin.flushAsync(team, peer, ticket);
}

// Writes over n_ctx contexts (warp w -> context ctx0 + w), no completion action: the fence below covers them.
__global__ void gin_put_multi_kernel(ncclDevComm dcomm, ncclWindow_t win, int ctx0, int n_ctx, int peer,
                                     const Seg* segs, int n) {
  const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
  ncclGin gin(dcomm, ctx0 + w);
  ncclTeam team = ncclTeamWorld(dcomm);
  for (int i = w + n_ctx * lane; i < n; i += n_ctx * 32) {
    Seg sg = segs[i];
    gin.put(team, peer, win, (size_t)sg.remote, win, (size_t)sg.local, (size_t)sg.bytes, ncclGin_None{},
            ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_device,
            cuda::thread_scope_device, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
  }
}

// Durability fence for writes: on every write context, an 8-byte get after the puts (same QP, so the read is
// served only after the earlier writes have landed), then wait for it. Optionally commit with a strong signal
// (all earlier puts on that context settled at the peer) so the store can see it. t_done: globaltimer.
__global__ void gin_fence_kernel(ncclDevComm dcomm, ncclWindow_t win, int ctx0, int n_ctx, int peer,
                                 int64_t remote_off, int64_t local_off, int sig, uint64_t* t_done) {
  ncclTeam team = ncclTeamWorld(dcomm);
  for (int c = 0; c < n_ctx; ++c) {
    ncclGin gin(dcomm, ctx0 + c);
    gin.get(team, peer, win, (size_t)remote_off, win, (size_t)(local_off + 8 * c), 8, ncclCoopThread{},
            ncclGin_None{}, ncclGinOptFlagsDefault, ncclGin_SegmentMixed{});
    ncclGinRequest_t r;
    gin.flushAsync(team, peer, &r);
    gin.wait(r);
  }
  if (sig >= 0) {
    ncclGin gin(dcomm, ctx0);
    gin.signal(team, peer, ncclGin_StrongSignalInc{(ncclGinSignal_t)sig});
  }
  if (t_done) *t_done = globaltimer();
}

__global__ void gin_wait_ticket_kernel(ncclDevComm dcomm, int ctx, ncclGinRequest_t* ticket) {
  ncclGin gin(dcomm, ctx);
  ncclGinRequest_t r = *ticket;
  gin.wait(r);
}

__global__ void gin_commit_kernel(ncclDevComm dcomm, int ctx, int peer, int sig) {
  ncclGin gin(dcomm, ctx);
  ncclTeam team = ncclTeamWorld(dcomm);
  gin.signal(team, peer, ncclGin_StrongSignalInc{(ncclGinSignal_t)sig});
}

__global__ void gin_wait_signal_kernel(ncclDevComm dcomm, int ctx, int sig, uint64_t value) {
  ncclGin gin(dcomm, ctx);
  gin.waitSignal(ncclCoopThread{}, (ncclGinSignal_t)sig, value);
}

__global__ void gin_read_signal_kernel(ncclDevComm dcomm, int ctx, int sig, uint64_t* out) {
  ncclGin gin(dcomm, ctx);
  *out = gin.readSignal((ncclGinSignal_t)sig);
}

// ---------------------------------------------------------------- host side
static void host_numa_alloc(void** ptr, size_t* size) {
  CUdevice dev;
  int cudaDev = 0, numa = -1;
  CUDA_OK(cudaGetDevice(&cudaDev));
  CU_OK(cuDeviceGet(&dev, cudaDev));
  // NUMA node of the GPU's PCI device from sysfs. CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID is not reported by driver 535
  // (it returned < 0, and a fallback to node 0 put every store pool on one socket: ~230 GB/s per node instead of
  // ~395 with 8 NICs, and half speed for GPUs on the other socket).
  {
    char bus[32] = {0};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), cudaDev) == cudaSuccess) {
      for (char* c = bus; *c; ++c) *c = (char)tolower(*c);
      std::string path = std::string("/sys/bus/pci/devices/") + bus + "/numa_node";
      FILE* f = fopen(path.c_str(), "r");
      if (f) {
        if (fscanf(f, "%d", &numa) != 1) numa = -1;
        fclose(f);
      }
    }
    if (numa < 0) cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev);
    if (numa < 0) numa = 0;
    fprintf(stderr, "hpoc: GIN host pool on NUMA node %d (GPU %d, %s)\n", numa, cudaDev, bus);
  }
  CUmemAllocationProp prop = {};
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = numa;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t gran = 0, gmin = 0;
  CU_OK(cuMemGetAllocationGranularity(&gmin, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CU_OK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  if (const char* e = getenv("HPOC_HOST_GRAN_MB")) gran = (size_t)atol(e) << 20;   // P2 diagnostic
  if (gran < gmin) gran = gmin;
  fprintf(stderr, "hpoc: HOST_NUMA granularity min %zu, recommended/used %zu bytes\n", gmin, gran);
  *size = (*size + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle h;
  CU_OK(cuMemCreate(&h, *size, &prop, 0));
  CU_OK(cuMemAddressReserve((CUdeviceptr*)ptr, *size, gran, 0, 0));
  CU_OK(cuMemMap((CUdeviceptr)*ptr, *size, 0, h, 0));
  CU_OK(cuMemRelease(h));   // the mapping keeps it alive
  CUmemAccessDesc acc = {};
  acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  acc.location.id = cudaDev;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CU_OK(cuMemSetAccess((CUdeviceptr)*ptr, *size, &acc, 1));
  acc.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  acc.location.id = numa;
  CU_OK(cuMemSetAccess((CUdeviceptr)*ptr, *size, &acc, 1));
}

static void host_numa_free(void* ptr, size_t size) {
  if (!ptr) return;
  cuMemUnmap((CUdeviceptr)ptr, size);
  cuMemAddressFree((CUdeviceptr)ptr, size);
}

std::string gin_unique_id() {
  ncclUniqueId id;
  NCCL_OK(ncclGetUniqueId(&id));
  return std::string(id.internal, sizeof(id.internal));
}

struct GinKVImpl {
  ncclComm_t comm = nullptr;
  ncclDevComm dcomm;
  bool have_dcomm = false;
  ncclWindow_t win = nullptr;
  void* buf = nullptr;
  size_t bytes = 0;
  bool on_host = false;
  int rank = 0, nranks = 0;
  uint64_t* sig_out = nullptr;
};

GinKV::GinKV(const std::string& uid, int rank, int nranks, int64_t local_bytes, bool local_on_host, int n_ctx,
             int queue_depth, bool strong_signals, int n_signals)
    : p_(new GinKVImpl) {
  if (uid.size() != NCCL_UNIQUE_ID_BYTES) throw std::runtime_error("bad unique id size");
  ncclUniqueId id;
  memcpy(id.internal, uid.data(), NCCL_UNIQUE_ID_BYTES);
  p_->rank = rank;
  p_->nranks = nranks;
  NCCL_OK(ncclCommInitRank(&p_->comm, nranks, id, rank));

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginContextCount = n_ctx;
  reqs.ginQueueDepth = queue_depth;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginSignalCount = n_signals;
  reqs.ginCounterCount = 1;
  reqs.ginStrongSignalsRequired = strong_signals;
  reqs.ginVaSignalsRequired = false;
  NCCL_OK(ncclDevCommCreate(p_->comm, &reqs, &p_->dcomm));
  p_->have_dcomm = true;

  p_->on_host = local_on_host;
  p_->bytes = (size_t)local_bytes;
  if (local_on_host) {
    host_numa_alloc(&p_->buf, &p_->bytes);
  } else {
    p_->bytes = (p_->bytes + NCCL_WIN_REQUIRED_ALIGNMENT - 1) / NCCL_WIN_REQUIRED_ALIGNMENT * NCCL_WIN_REQUIRED_ALIGNMENT;
    NCCL_OK(ncclMemAlloc(&p_->buf, p_->bytes));
  }
  NCCL_OK(ncclCommWindowRegister(p_->comm, p_->buf, p_->bytes, &p_->win, NCCL_WIN_GIN_ONLY));
  CUDA_OK(cudaMalloc(&p_->sig_out, sizeof(uint64_t)));
}

GinKV::~GinKV() {
  try {
    destroy();
  } catch (...) {
  }
  delete p_;
}

void GinKV::destroy() {
  if (!p_->comm) return;
  cudaDeviceSynchronize();
  if (p_->win) ncclCommWindowDeregister(p_->comm, p_->win);
  if (p_->have_dcomm) ncclDevCommDestroy(p_->comm, &p_->dcomm);
  if (p_->buf) {
    if (p_->on_host) host_numa_free(p_->buf, p_->bytes);
    else ncclMemFree(p_->buf);
  }
  if (p_->sig_out) cudaFree(p_->sig_out);
  ncclCommFinalize(p_->comm);
  ncclCommDestroy(p_->comm);
  p_->comm = nullptr;
  p_->win = nullptr;
  p_->buf = nullptr;
  p_->have_dcomm = false;
  p_->sig_out = nullptr;
}

void* GinKV::local_ptr() const { return p_->buf; }
int64_t GinKV::local_bytes() const { return (int64_t)p_->bytes; }
bool GinKV::on_host() const { return p_->on_host; }
int GinKV::rank() const { return p_->rank; }
int GinKV::nranks() const { return p_->nranks; }

void GinKV::post_gets(const Seg* segs, const int* layer_ptr, int n_layers, void* tickets, int peer, int ctx0,
                      int n_ctx, uint64_t* t_post, cudaStream_t s) {
  gin_post_gets_kernel<<<1, 32 * n_ctx, 0, s>>>(p_->dcomm, p_->win, ctx0, n_ctx, peer, segs, layer_ptr, n_layers,
                                                (ncclGinRequest_t*)tickets, t_post);
  CUDA_OK(cudaGetLastError());
}

void GinKV::wait_layer(void* tickets, int layer, int ctx0, int n_ctx, uint64_t* t_done, cudaStream_t s) {
  gin_wait_kernel<<<1, 1, 0, s>>>(p_->dcomm, ctx0, n_ctx, (ncclGinRequest_t*)tickets, layer, t_done);
  CUDA_OK(cudaGetLastError());
}

void GinKV::poll(void* tickets, int n_layers, int ctx0, int n_ctx, const uint32_t* seq, uint32_t* ready,
                 cudaStream_t s) {
  gin_poll_kernel<<<1, 1, 0, s>>>(p_->dcomm, ctx0, n_ctx, (ncclGinRequest_t*)tickets, n_layers, seq, ready);
  CUDA_OK(cudaGetLastError());
}

void GinKV::put_segs(const Seg* segs, int n_segs, void* ticket, int peer, int ctx, int threads, cudaStream_t s) {
  gin_put_kernel<<<1, threads, 0, s>>>(p_->dcomm, p_->win, ctx, peer, segs, n_segs, (ncclGinRequest_t*)ticket);
  CUDA_OK(cudaGetLastError());
}

void GinKV::put_multi(const Seg* segs, int n_segs, int peer, int ctx0, int n_ctx, cudaStream_t s) {
  if (n_segs <= 0) return;
  gin_put_multi_kernel<<<1, 32 * n_ctx, 0, s>>>(p_->dcomm, p_->win, ctx0, n_ctx, peer, segs, n_segs);
  CUDA_OK(cudaGetLastError());
}

void GinKV::fence(int peer, int ctx0, int n_ctx, int64_t remote_off, int64_t local_off, int sig, uint64_t* t_done,
                  cudaStream_t s) {
  gin_fence_kernel<<<1, 1, 0, s>>>(p_->dcomm, p_->win, ctx0, n_ctx, peer, remote_off, local_off, sig, t_done);
  CUDA_OK(cudaGetLastError());
}

void GinKV::wait_ticket(void* ticket, int ctx, cudaStream_t s) {
  gin_wait_ticket_kernel<<<1, 1, 0, s>>>(p_->dcomm, ctx, (ncclGinRequest_t*)ticket);
  CUDA_OK(cudaGetLastError());
}

void GinKV::commit(int peer, int ctx, int sig, cudaStream_t s) {
  gin_commit_kernel<<<1, 1, 0, s>>>(p_->dcomm, ctx, peer, sig);
  CUDA_OK(cudaGetLastError());
}

void GinKV::wait_signal(int ctx, int sig, uint64_t value, cudaStream_t s) {
  gin_wait_signal_kernel<<<1, 1, 0, s>>>(p_->dcomm, ctx, sig, value);
  CUDA_OK(cudaGetLastError());
}

uint64_t GinKV::read_signal_host(int ctx, int sig) {
  gin_read_signal_kernel<<<1, 1>>>(p_->dcomm, ctx, sig, p_->sig_out);
  CUDA_OK(cudaGetLastError());
  uint64_t v = 0;
  CUDA_OK(cudaMemcpy(&v, p_->sig_out, sizeof(v), cudaMemcpyDeviceToHost));
  return v;
}

}  // namespace hpoc
