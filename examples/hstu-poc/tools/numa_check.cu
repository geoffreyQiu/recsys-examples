// Where do CUDA HOST_NUMA allocations land on this driver (535 + CUDA 13 forward compatibility)?
// For each target node n: per-node MemFree before/after cuMemCreate(HOST_NUMA, id=n) of SIZE GiB, the node of sampled
// pages (get_mempolicy / move_pages), and the read bandwidth of a CPU thread pinned on each socket. Control: mmap +
// mbind(n) (the verbs/raw store), and cudaHostAlloc.
//   nvcc -O2 -o numa_check numa_check.cu -lcuda && ./numa_check 16
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CK(x)                                                                 \
  do {                                                                        \
    CUresult r_ = (x);                                                        \
    if (r_ != CUDA_SUCCESS) {                                                 \
      const char* s_ = nullptr;                                               \
      cuGetErrorString(r_, &s_);                                              \
      fprintf(stderr, "%s failed: %s\n", #x, s_ ? s_ : "?");                  \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

static double memfree_gib(int node) {
  char path[128];
  snprintf(path, sizeof path, "/sys/devices/system/node/node%d/meminfo", node);
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  char line[256];
  double v = -1;
  while (fgets(line, sizeof line, f)) {
    char* p = strstr(line, "MemFree:");
    if (p) { v = atol(p + 8) / 1048576.0; break; }
  }
  fclose(f);
  return v;
}

static int node_of(void* a) {   // get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR)
  int node = -1;
  long r = syscall(SYS_get_mempolicy, &node, nullptr, 0, a, 3);
  return r == 0 ? node : -errno;
}

static int node_of_move(void* a) {   // move_pages query mode (nodes == NULL)
  void* pages[1] = {a};
  int status[1] = {-999};
  long r = syscall(SYS_move_pages, 0, 1, pages, nullptr, status, 0);
  return r == 0 ? status[0] : -errno;
}

static double read_gbs(const void* p, size_t bytes, int cpu) {
  double gbs = 0;
  std::thread t([&] {
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof s, &s);
    const uint64_t* q = (const uint64_t*)p;
    size_t n = bytes / 8;
    volatile uint64_t acc = 0;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (size_t i = 0; i < n; i += 4) { a0 += q[i]; a1 += q[i + 1]; a2 += q[i + 2]; a3 += q[i + 3]; }   // warm
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; i += 4) { a0 += q[i]; a1 += q[i + 1]; a2 += q[i + 2]; a3 += q[i + 3]; }
    auto t1 = std::chrono::steady_clock::now();
    acc = a0 + a1 + a2 + a3;
    (void)acc;
    gbs = bytes / std::chrono::duration<double>(t1 - t0).count() / 1e9;
  });
  t.join();
  return gbs;
}

static void report(const char* what, void* p, size_t bytes, double f0, double f1) {
  double d0 = f0 - memfree_gib(0), d1 = f1 - memfree_gib(1);
  printf("%-34s node0 -%.1f GiB, node1 -%.1f GiB | pages:", what, d0, d1);
  for (int k = 0; k < 4; ++k) {
    char* a = (char*)p + bytes / 4 * k;
    printf(" %d/%d", node_of(a), node_of_move(a));
  }
  // 1 GiB read from a core on socket 0 (core 8) and on socket 1 (core 64)
  size_t rb = bytes < (1ull << 30) ? bytes : (1ull << 30);
  printf(" | CPU read GB/s: socket0 %.1f, socket1 %.1f\n", read_gbs(p, rb, 8), read_gbs(p, rb, 64));
  fflush(stdout);
}

int main(int argc, char** argv) {
  size_t gib = argc > 1 ? atol(argv[1]) : 16;
  size_t bytes = gib << 30;
  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  int attr = -7;
  cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev);
  printf("CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID of GPU 0: %d; size %zu GiB\n", attr, gib);
  for (int node = 0; node < 2; ++node) {
    double f0 = memfree_gib(0), f1 = memfree_gib(1);
    CUmemAllocationProp prop = {};
    prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    prop.location.id = node;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gran = 0;
    CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    // NUMA_BIND=1: bind the calling thread's memory policy to the target node around cuMemCreate;
    // NUMA_PIN=1: also pin the thread to a core of that node
    const char* bind = getenv("NUMA_BIND");
    const char* pin = getenv("NUMA_PIN");
    cpu_set_t saved;
    sched_getaffinity(0, sizeof saved, &saved);
    if (pin && atoi(pin)) {
      cpu_set_t s;
      CPU_ZERO(&s);
      CPU_SET(node ? 64 : 8, &s);
      sched_setaffinity(0, sizeof s, &s);
    }
    if (bind && atoi(bind)) {
      unsigned long mask = 1ul << node;
      if (syscall(SYS_set_mempolicy, 2 /*MPOL_BIND*/, &mask, 64) != 0) perror("set_mempolicy");
    }
    CUmemGenericAllocationHandle h;
    CK(cuMemCreate(&h, bytes, &prop, 0));
    if (bind && atoi(bind)) syscall(SYS_set_mempolicy, 0 /*MPOL_DEFAULT*/, nullptr, 0);
    if (pin && atoi(pin)) sched_setaffinity(0, sizeof saved, &saved);
    CUdeviceptr p;
    CK(cuMemAddressReserve(&p, bytes, gran, 0, 0));
    CK(cuMemMap(p, bytes, 0, h, 0));
    CK(cuMemRelease(h));
    CUmemAccessDesc acc[2] = {};
    acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc[0].location.id = 0;
    acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    acc[1].location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    acc[1].location.id = node;
    acc[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CK(cuMemSetAccess(p, bytes, acc, 2));
    memset((void*)p, 1, 1 << 20);
    char what[64];
    snprintf(what, sizeof what, "cuMemCreate HOST_NUMA id=%d", node);
    report(what, (void*)p, bytes, f0, f1);
    CK(cuMemUnmap(p, bytes));
    CK(cuMemAddressFree(p, bytes));

    f0 = memfree_gib(0), f1 = memfree_gib(1);
    void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    unsigned long mask = 1ul << node;
    syscall(SYS_mbind, m, bytes, 2 /*MPOL_BIND*/, &mask, 64, 0);
    memset(m, 1, bytes);
    snprintf(what, sizeof what, "mmap + mbind(%d) (verbs/raw store)", node);
    report(what, m, bytes, f0, f1);
    munmap(m, bytes);
  }
  double f0 = memfree_gib(0), f1 = memfree_gib(1);
  void* hp = nullptr;
  if (cudaHostAlloc(&hp, bytes, cudaHostAllocMapped) == cudaSuccess) {
    report("cudaHostAlloc (calling thread's node)", hp, bytes, f0, f1);
    cudaFreeHost(hp);
  }
  return 0;
}
