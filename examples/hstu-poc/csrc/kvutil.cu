#include "kvutil.h"

#include <cuda.h>
#include <omp.h>

#include <stdexcept>
#include <time.h>
#include <vector>
#include <string>

namespace hpoc {

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

void fill_pattern_host(void* dst, int64_t words, int64_t user, int64_t layer, int64_t word0, int64_t seed) {
  uint64_t* p = (uint64_t*)dst;
#pragma omp parallel for schedule(static)
  for (int64_t w = 0; w < words; ++w) p[w] = kv_pattern(user, layer, word0 + w, seed);
}

// grid: (blocks per row, rows)
__global__ void check_kernel(const char* base, const int64_t* rows, int64_t seed, unsigned long long* bad) {
  const int64_t* r = rows + 4 * blockIdx.y;
  const uint64_t* p = (const uint64_t*)(base + r[0]);
  int64_t words = r[1] / 8, user = r[2], layer = r[3] >> 40, w0 = r[3] & ((1ll << 40) - 1);
  unsigned long long n = 0;
  for (int64_t w = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; w < words; w += (int64_t)gridDim.x * blockDim.x)
    n += (p[w] != kv_pattern(user, layer, w0 + w, seed));
  if (n) atomicAdd(bad, n);
}

__global__ void fill_kernel(char* base, const int64_t* rows, int64_t seed) {
  const int64_t* r = rows + 4 * blockIdx.y;
  uint64_t* p = (uint64_t*)(base + r[0]);
  int64_t words = r[1] / 8, user = r[2], layer = r[3] >> 40, w0 = r[3] & ((1ll << 40) - 1);
  for (int64_t w = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; w < words; w += (int64_t)gridDim.x * blockDim.x)
    p[w] = kv_pattern(user, layer, w0 + w, seed);
}

void check_pattern_gpu(const void* base, const int64_t* rows, int n_rows, int64_t seed, unsigned long long* bad,
                       cudaStream_t s) {
  if (n_rows <= 0) return;
  dim3 grid(64, n_rows);
  check_kernel<<<grid, 256, 0, s>>>((const char*)base, rows, seed, bad);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("check_kernel: ") + cudaGetErrorString(e));
}

void fill_pattern_gpu(void* base, const int64_t* rows, int n_rows, int64_t seed, cudaStream_t s) {
  if (n_rows <= 0) return;
  dim3 grid(64, n_rows);
  fill_kernel<<<grid, 256, 0, s>>>((char*)base, rows, seed);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("fill_kernel: ") + cudaGetErrorString(e));
}

int64_t check_pattern_host(const void* base, const int64_t* rows, int n_rows, int64_t seed) {
  int64_t bad = 0;
  for (int i = 0; i < n_rows; ++i) {
    const int64_t* r = rows + 4 * i;
    const uint64_t* p = (const uint64_t*)((const char*)base + r[0]);
    int64_t words = r[1] / 8, user = r[2], layer = r[3] >> 40, w0 = r[3] & ((1ll << 40) - 1);
#pragma omp parallel for reduction(+ : bad) schedule(static)
    for (int64_t w = 0; w < words; ++w) bad += (p[w] != kv_pattern(user, layer, w0 + w, seed));
  }
  return bad;
}

__global__ void spin_kernel(int64_t ns) {
  uint64_t t0 = gtimer();
  while ((int64_t)(gtimer() - t0) < ns) {
  }
}
__global__ void stamp_kernel(uint64_t* dst) { *dst = gtimer(); }
__global__ void wait_flag_kernel(const volatile uint32_t* flag, uint32_t value, uint64_t* stamp) {
  while (*flag < value) {
  }
  __threadfence_system();
  if (stamp) *stamp = gtimer();
}

void spin_gpu(int64_t ns, cudaStream_t s) { spin_kernel<<<1, 1, 0, s>>>(ns); }
void stamp_gpu(uint64_t* dst, cudaStream_t s) { stamp_kernel<<<1, 1, 0, s>>>(dst); }
void wait_flag_gpu(const volatile uint32_t* flag, uint32_t value, uint64_t* stamp, cudaStream_t s) {
  wait_flag_kernel<<<1, 1, 0, s>>>(flag, value, stamp);
}

__global__ void seq_inc_kernel(uint32_t* seq) { *seq += 1; }
__global__ void wait_flags_seq_kernel(const volatile uint32_t* flags, int layer, const uint32_t* seq, uint64_t* stamp) {
  asm volatile("griddepcontrol.launch_dependents;");   // P4 PDL
  const uint32_t want = *seq;
  while (flags[layer] < want) {
  }
  __threadfence_system();
  if (stamp) stamp[layer] = gtimer();
}
void seq_inc_gpu(uint32_t* seq, cudaStream_t s) { seq_inc_kernel<<<1, 1, 0, s>>>(seq); }
// P4 fused, CPU verbs: relay the CPU's host-mapped per-layer flags into device ready words (one thread polls PCIe,
// the attention CTAs spin on L2)
__device__ __forceinline__ void trace_event(unsigned long long* tr, int& n, int l, long long v) {
  if (tr && n < TRACE_CAP) {
    unsigned long long* e = tr + TRACE_EV + 3 * n;
    e[0] = (unsigned long long)l;
    e[1] = (unsigned long long)v;
    e[2] = gtimer();
    ++n;
  }
}
__global__ void relay_flags_kernel(const volatile uint32_t* host_flags, int n_layers, const uint32_t* seq, uint32_t* ready,
                                   unsigned long long* tr) {
  if (tr) tr[1] = gtimer();
  int n = 0;
  const uint32_t s = *seq;
  for (int l = 0; l < n_layers; ++l) {
    while (host_flags[l] < s) {
    }
    __threadfence_system();
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(ready + l), "r"(s) : "memory");
    trace_event(tr, n, l, 0x7fffffff);
  }
  if (tr) tr[0] = n;
}
void relay_flags_gpu(const volatile uint32_t* host_flags, int n_layers, const uint32_t* seq, uint32_t* ready,
                     cudaStream_t s, unsigned long long* trace) {
  relay_flags_kernel<<<1, 1, 0, s>>>(host_flags, n_layers, seq, ready, trace);
}
__global__ void relay_tiles_kernel(const volatile unsigned long long* prog, const volatile uint32_t* host_flags,
                                   int n_layers, const uint32_t* seq, int32_t* tsteps, unsigned long long* tr) {
  if (tr) tr[1] = gtimer();
  int n = 0;
  const uint32_t s = *seq;
  for (int l = 0; l < n_layers; ++l) {
    int last = 0;
    while (true) {
      const bool full = host_flags[l] >= s;           // read the flag first: then progress below is final too
      const unsigned long long p = prog[l];
      if ((uint32_t)(p >> 32) == s && (int)(p & 0xffffffffull) > last) {
        last = (int)(p & 0xffffffffull);
        __threadfence_system();
        asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(tsteps + l), "r"(last) : "memory");
        trace_event(tr, n, l, last);
      }
      if (full) {
        __threadfence_system();
        asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(tsteps + l), "r"(0x7fffffff) : "memory");
        trace_event(tr, n, l, 0x7fffffff);
        break;
      }
    }
  }
  if (tr) tr[0] = n;
}
// v2: one PCIe read per poll: the worker marks a complete layer in its progress word (steps = 0x7fffffff)
__global__ void relay_tiles2_kernel(const volatile unsigned long long* prog, int n_layers, const uint32_t* seq,
                                    int32_t* tsteps, unsigned long long* tr) {
  if (tr) tr[1] = gtimer();
  int n = 0;
  const uint32_t s = *seq;
  for (int l = 0; l < n_layers; ++l) {
    int last = 0;
    while (last != 0x7fffffff) {
      const unsigned long long p = prog[l];
      const int v = (int)(p & 0xffffffffull);
      if ((uint32_t)(p >> 32) == s && v > last) {
        last = v;
        __threadfence_system();
        asm volatile("st.release.gpu.global.s32 [%0], %1;" ::"l"(tsteps + l), "r"(last) : "memory");
        trace_event(tr, n, l, last);
      }
    }
  }
  if (tr) tr[0] = n;
}
void relay_tiles_gpu(const volatile unsigned long long* prog, const volatile uint32_t* host_flags, int n_layers,
                     const uint32_t* seq, int32_t* tsteps, cudaStream_t s, unsigned long long* trace, int version) {
  if (version == 2)
    relay_tiles2_kernel<<<1, 1, 0, s>>>(prog, n_layers, seq, tsteps, trace);
  else
    relay_tiles_kernel<<<1, 1, 0, s>>>(prog, host_flags, n_layers, seq, tsteps, trace);
}

// clock calibration (see kvutil.h): stop after n values or at the sentinel -1
__global__ void calib_kernel(const volatile long long* hw, long long* out, int n) {
  long long last = hw[0];
  for (int i = 0; i < n;) {
    const long long v = hw[0];
    if (v == last) continue;
    if (v == -1) break;
    out[2 * i] = v;
    out[2 * i + 1] = (long long)gtimer();
    last = v;
    ++i;
  }
}
static int64_t mono_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}
std::vector<int64_t> calibrate_clock(int n) {
  long long* hw = nullptr;
  long long* hw_dev = nullptr;
  long long* out = nullptr;
  cudaHostAlloc((void**)&hw, sizeof(long long), cudaHostAllocMapped);
  cudaHostGetDevicePointer((void**)&hw_dev, hw, 0);
  cudaMalloc((void**)&out, 2 * sizeof(long long) * n);
  cudaMemset(out, 0, 2 * sizeof(long long) * n);
  *(volatile long long*)hw = 0;
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  calib_kernel<<<1, 1, 0, s>>>(hw_dev, out, n);
  const int64_t t_start = mono_ns();
  while (mono_ns() - t_start < 200000) {   // let the kernel start
  }
  for (int i = 0; i < n + 16; ++i) {
    const int64_t t0 = mono_ns();
    while (mono_ns() - t0 < 5000) {         // ~5 us apart: the GPU sees every value
    }
    const int64_t t = mono_ns();
    __atomic_store_n((volatile long long*)hw, (long long)t, __ATOMIC_SEQ_CST);
  }
  __atomic_store_n((volatile long long*)hw, -1ll, __ATOMIC_SEQ_CST);
  cudaStreamSynchronize(s);
  std::vector<int64_t> r(2 * n);
  cudaMemcpy(r.data(), out, 2 * sizeof(long long) * n, cudaMemcpyDeviceToHost);
  cudaStreamDestroy(s);
  cudaFree(out);
  cudaFreeHost(hw);
  return r;
}
__global__ void set_flag_seq_kernel(volatile uint32_t* flags, int layer, const uint32_t* seq) {
  __threadfence_system();
  flags[layer] = *seq;
}
void set_flag_seq_gpu(volatile uint32_t* flags, int layer, const uint32_t* seq, cudaStream_t s) {
  set_flag_seq_kernel<<<1, 1, 0, s>>>(flags, layer, seq);
}
void wait_flags_seq_gpu(const volatile uint32_t* flags, int layer, const uint32_t* seq, uint64_t* stamp,
                        cudaStream_t s) {
  wait_flags_seq_kernel<<<1, 1, 0, s>>>(flags, layer, seq, stamp);
}

__global__ void hog_kernel(int64_t ns) {
  if (threadIdx.x == 0) {
    uint64_t t0 = gtimer();
    while ((int64_t)(gtimer() - t0) < ns) {
    }
  }
  __syncthreads();
}
void hog_gpu(int64_t ns, int blocks, int threads, cudaStream_t s) { hog_kernel<<<blocks, threads, 0, s>>>(ns); }

#define GC_OK(x)                                                                                \
  do {                                                                                          \
    CUresult r_ = (x);                                                                          \
    if (r_ != CUDA_SUCCESS) throw std::runtime_error(std::string("green ctx: ") + #x + " = " + std::to_string((int)r_)); \
  } while (0)
void green_streams(int n_sms, uint64_t* stream_small, uint64_t* stream_rest, int* sms_small, int* sms_rest) {
  int dev = 0;
  cudaGetDevice(&dev);
  CUdevice cd;
  GC_OK(cuDeviceGet(&cd, dev));
  CUdevResource sm{}, part{}, rem{};
  GC_OK(cuDeviceGetDevResource(cd, &sm, CU_DEV_RESOURCE_TYPE_SM));
  unsigned n = 1;
  GC_OK(cuDevSmResourceSplitByCount(&part, &n, &sm, &rem, 0, (unsigned)n_sms));
  CUdevResourceDesc d1, d2;
  GC_OK(cuDevResourceGenerateDesc(&d1, &part, 1));
  GC_OK(cuDevResourceGenerateDesc(&d2, &rem, 1));
  CUgreenCtx g1, g2;
  GC_OK(cuGreenCtxCreate(&g1, d1, cd, CU_GREEN_CTX_DEFAULT_STREAM));
  GC_OK(cuGreenCtxCreate(&g2, d2, cd, CU_GREEN_CTX_DEFAULT_STREAM));
  CUstream s1, s2;
  GC_OK(cuGreenCtxStreamCreate(&s1, g1, CU_STREAM_NON_BLOCKING, 0));
  GC_OK(cuGreenCtxStreamCreate(&s2, g2, CU_STREAM_NON_BLOCKING, 0));
  *stream_small = (uint64_t)s1;
  *stream_rest = (uint64_t)s2;
  *sms_small = (int)part.sm.smCount;
  *sms_rest = (int)rem.sm.smCount;
}

}  // namespace hpoc
