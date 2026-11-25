// Helpers shared by all transports: a seeded KV data pattern (finite BF16 values, distinct per user, layer and
// position), a GPU checker, spin/timestamp kernels for timing tests.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace hpoc {

// 8-byte word `word` of (user, layer) for a given seed. Each 16-bit lane is a finite BF16 in +-[2^-7, 2^-6).
__host__ __device__ inline uint64_t kv_pattern(uint64_t user, uint64_t layer, uint64_t word, uint64_t seed) {
  uint64_t x = seed ^ (user * 0x9E3779B97F4A7C15ull) ^ (layer * 0xC2B2AE3D27D4EB4Full) ^ (word * 0x165667B19E3779F9ull);
  x ^= x >> 31;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 29;
  return (x & 0x807F807F807F807Full) | 0x3C003C003C003C00ull;
}

// Fill `words` words at host pointer `dst` with the pattern of (user, layer) starting at word index `word0`.
void fill_pattern_host(void* dst, int64_t words, int64_t user, int64_t layer, int64_t word0, int64_t seed);

// Check segments of local GPU memory against the pattern. Each check row is 4 int64:
// {local byte offset, bytes, user, (layer << 40) | first word index}. Mismatched words are added to *bad.
void check_pattern_gpu(const void* base, const int64_t* rows, int n_rows, int64_t seed, unsigned long long* bad,
                       cudaStream_t s);
// Same, for a CPU-visible buffer (store side after writes). Returns mismatched words.
int64_t check_pattern_host(const void* base, const int64_t* rows, int n_rows, int64_t seed);
// GPU fill of local memory with the pattern (write tests: the source data).
void fill_pattern_gpu(void* base, const int64_t* rows, int n_rows, int64_t seed, cudaStream_t s);

void spin_gpu(int64_t ns, cudaStream_t s);                     // one thread spins for ns
void stamp_gpu(uint64_t* dst, cudaStream_t s);                 // *dst = globaltimer
// Spin until *flag >= value (flag in host-mapped or device memory), then stamp.
void wait_flag_gpu(const volatile uint32_t* flag, uint32_t value, uint64_t* stamp, cudaStream_t s);

// Request sequence numbers for CPU-posted transports inside CUDA graphs: the graph's first kernel does
// (*seq)++, and the per-layer wait spins until flags[layer] >= *seq (the CPU stores seq into flags[layer]).
void seq_inc_gpu(uint32_t* seq, cudaStream_t s);
void relay_flags_gpu(const volatile uint32_t* host_flags, int n_layers, const uint32_t* seq, uint32_t* ready,
                     cudaStream_t s);   // P4 fused, CPU verbs
// flags[layer] = *seq, after a system-scope fence (host-mapped flags read by a CPU thread)
void set_flag_seq_gpu(volatile uint32_t* flags, int layer, const uint32_t* seq, cudaStream_t s);
void wait_flags_seq_gpu(const volatile uint32_t* flags, int layer, const uint32_t* seq, uint64_t* stamp,
                        cudaStream_t s);

// P6: an SM hog (every block's thread 0 spins for ns while the block stays resident) and green contexts.
void hog_gpu(int64_t ns, int blocks, int threads, cudaStream_t s);
// Split the device's SMs: a green context with n_sms (for transfer kernels) and one with the rest. Returns the two
// streams (CUstream handles usable as cudaStream_t) and the actual SM counts.
void green_streams(int n_sms, uint64_t* stream_small, uint64_t* stream_rest, int* sms_small, int* sms_rest);

}  // namespace hpoc
