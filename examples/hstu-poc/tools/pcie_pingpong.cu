// Host <-> GPU notification latency (SoL constants t_in, t_out, docs/sol.md): a resident 1-thread kernel waits for a
// value the host writes into host-mapped memory and answers through another host-mapped word; the host times the
// round trip (one-way ~ RTT / 2). Also: launch of a 1-kernel CUDA graph until the kernel runs, and a kernel's end
// until cudaEventQuery reports it (the paths the POC's requests use).
//   nvcc -O2 -arch=sm_90 -o /tmp/pp tools/pcie_pingpong.cu && /tmp/pp
#include <cuda_runtime.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <vector>

static long long now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

__global__ void pong(volatile int* h2d, volatile int* d2h, int n) {
  for (int i = 1; i <= n; ++i) {
    while (*h2d != i) {
    }
    *d2h = i;
    __threadfence_system();
  }
}

__global__ void stamp(unsigned long long* t) {
  unsigned long long g;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
  *t = g;
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main() {
  const int n = 2000;
  int *h2d, *d2h, *h2d_d, *d2h_d;
  cudaHostAlloc((void**)&h2d, 64, cudaHostAllocMapped);
  cudaHostAlloc((void**)&d2h, 64, cudaHostAllocMapped);
  *h2d = 0;
  *d2h = 0;
  cudaHostGetDevicePointer((void**)&h2d_d, h2d, 0);
  cudaHostGetDevicePointer((void**)&d2h_d, d2h, 0);
  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  pong<<<1, 1, 0, s>>>(h2d_d, d2h_d, n);
  const long long t_wait = now_ns();
  while (now_ns() - t_wait < 100000) {
  }
  std::vector<double> rtt;
  for (int i = 1; i <= n; ++i) {
    const long long t0 = now_ns();
    __atomic_store_n((volatile int*)h2d, i, __ATOMIC_SEQ_CST);
    while (__atomic_load_n((volatile int*)d2h, __ATOMIC_SEQ_CST) != i) {
    }
    rtt.push_back((now_ns() - t0) / 1e3);
  }
  cudaStreamSynchronize(s);
  std::sort(rtt.begin(), rtt.end());
  printf("host->GPU->host round trip (mapped memory): min %.2f  median %.2f  p99 %.2f us  => one way ~%.2f us\n",
         rtt[0], rtt[rtt.size() / 2], rtt[rtt.size() * 99 / 100], rtt[rtt.size() / 2] / 2);

  // CUDA graph launch -> kernel start, and kernel end -> event seen by a spinning host
  unsigned long long* t_dev;
  cudaMalloc((void**)&t_dev, 8);
  cudaGraph_t g;
  cudaGraphExec_t ge;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  stamp<<<1, 1, 0, s>>>(t_dev);
  cudaStreamEndCapture(s, &g);
  cudaGraphInstantiate(&ge, g, 0);
  // offset between globaltimer and CLOCK_MONOTONIC: from the ping-pong kernel's view is not available here, so the
  // launch latency is taken as (event seen - launch call) - (kernel end -> event seen), both on the host clock
  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  std::vector<double> call, total;
  for (int i = 0; i < 1000; ++i) {
    cudaStreamSynchronize(s);
    const long long t0 = now_ns();
    cudaGraphLaunch(ge, s);
    const long long t1 = now_ns();
    cudaEventRecord(ev, s);
    while (cudaEventQuery(ev) != cudaSuccess) {
    }
    const long long t2 = now_ns();
    call.push_back((t1 - t0) / 1e3);
    total.push_back((t2 - t0) / 1e3);
  }
  printf("1-kernel graph: launch call %.2f us, launch -> event seen %.2f us (median)\n", median(call), median(total));
  return 0;
}
