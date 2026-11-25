#include "kvutil.h"

#include <omp.h>

#include <stdexcept>
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
  const uint32_t want = *seq;
  while (flags[layer] < want) {
  }
  __threadfence_system();
  if (stamp) stamp[layer] = gtimer();
}
void seq_inc_gpu(uint32_t* seq, cudaStream_t s) { seq_inc_kernel<<<1, 1, 0, s>>>(seq); }
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

}  // namespace hpoc
