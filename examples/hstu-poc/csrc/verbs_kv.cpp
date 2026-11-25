// CPU verbs transport, see verbs_kv.h.
#include "verbs_kv.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace hpoc {

#define VCHECK(x, msg) \
  do {                 \
    if (!(x)) throw std::runtime_error(std::string("verbs: ") + msg + ": " + strerror(errno)); \
  } while (0)
#define CUDACHK(cmd)                                                                                       \
  do {                                                                                                     \
    cudaError_t r_ = (cmd);                                                                                \
    if (r_ != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + #cmd + ": " + cudaGetErrorString(r_)); \
  } while (0)

struct Job {
  std::vector<int64_t> segs;   // 4 per segment
  std::vector<int> lp;
  int n_layers = 0;
  uint32_t seq = 0;
  bool write = false;
};

struct VerbsImpl {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  std::vector<ibv_qp*> qps;
  ibv_mr* mr = nullptr;
  void* buf = nullptr;
  size_t bytes = 0;
  bool gpu = false;
  int cuda_dev = -1;
  int port = 1;
  ibv_port_attr pattr{};
  ibv_gid gid{};
  uint64_t raddr = 0;
  uint32_t rkey = 0;
  uint32_t* flags_host = nullptr;
  uint32_t* flags_dev = nullptr;
  int max_layers = 0, flush_mode = 0, cpu_core = -1, gdr_order = -1;
  int sq_depth = 4096;
  std::thread worker;
  std::mutex mu;
  std::condition_variable cv, cv_idle;
  std::deque<Job> jobs;
  bool stop = false;
  int64_t submitted = 0;
  std::atomic<int64_t> done{0};
  std::atomic<int64_t> cpu_ns{0};
  std::string err;
};

static int nic_numa(const std::string& dev) {
  std::ifstream f("/sys/class/infiniband/" + dev + "/device/numa_node");
  int n = -1;
  if (f) f >> n;
  return n < 0 ? 0 : n;
}

static void* numa_host_alloc(size_t bytes, int node) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  VCHECK(p != MAP_FAILED, "mmap");
  madvise(p, bytes, MADV_HUGEPAGE);
  unsigned long mask = 1ul << node;
  long r = syscall(SYS_mbind, p, bytes, 2 /*MPOL_BIND*/, &mask, 64, 0);
  VCHECK(r == 0, "mbind");
  return p;
}

static int64_t thread_ns() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static void run_job(VerbsImpl* p, const Job& j) {
  const int nqp = (int)p->qps.size();
  std::vector<int> expect(j.n_layers, 0), got(j.n_layers, 0);
  std::vector<ibv_sge> sges;
  std::vector<ibv_send_wr> wrs;
  for (int l = 0; l < j.n_layers; ++l) {
    int s0 = j.lp[l], s1 = j.lp[l + 1];
    for (int q = 0; q < nqp; ++q) {
      sges.clear();
      wrs.clear();
      for (int i = s0 + q; i < s1; i += nqp) {
        const int64_t* s = &j.segs[4 * (size_t)i];
        ibv_sge g{};
        g.addr = (uint64_t)p->buf + (uint64_t)s[1];
        g.length = (uint32_t)s[2];
        g.lkey = p->mr->lkey;
        sges.push_back(g);
      }
      if (sges.empty()) continue;
      wrs.resize(sges.size());
      for (size_t k = 0; k < sges.size(); ++k) {
        int i = s0 + q + (int)k * nqp;
        const int64_t* s = &j.segs[4 * (size_t)i];
        ibv_send_wr& w = wrs[k];
        memset(&w, 0, sizeof(w));
        w.wr_id = ((uint64_t)l << 8) | (uint64_t)q;
        w.sg_list = &sges[k];
        w.num_sge = 1;
        w.opcode = j.write ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
        w.send_flags = (k + 1 == sges.size()) ? IBV_SEND_SIGNALED : 0;
        w.wr.rdma.remote_addr = p->raddr + (uint64_t)s[0];
        w.wr.rdma.rkey = p->rkey;
        w.next = (k + 1 < sges.size()) ? &wrs[k + 1] : nullptr;
      }
      ibv_send_wr* bad = nullptr;
      int r = ibv_post_send(p->qps[q], wrs.data(), &bad);
      if (r) throw std::runtime_error("ibv_post_send failed: " + std::string(strerror(r)));
      expect[l]++;
    }
  }
  int next = 0;
  ibv_wc wc[64];
  while (next < j.n_layers) {
    while (next < j.n_layers && got[next] >= expect[next]) {
      if (!j.write && p->flush_mode == 1) {
        cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                   CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER);
      }
      if (!j.write) __atomic_store_n(&p->flags_host[next], j.seq, __ATOMIC_RELEASE);
      ++next;
    }
    if (next >= j.n_layers) break;
    int n = ibv_poll_cq(p->cq, 64, wc);
    if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
    for (int k = 0; k < n; ++k) {
      if (wc[k].status != IBV_WC_SUCCESS)
        throw std::runtime_error(std::string("work completion error: ") + ibv_wc_status_str(wc[k].status));
      got[wc[k].wr_id >> 8]++;
    }
  }
}

static void worker_main(VerbsImpl* p) {
  if (p->cpu_core >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(p->cpu_core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
  if (p->gpu) cudaSetDevice(p->cuda_dev);
  while (true) {
    Job j;
    {
      std::unique_lock<std::mutex> lk(p->mu);
      p->cv.wait(lk, [p] { return p->stop || !p->jobs.empty(); });
      if (p->stop && p->jobs.empty()) return;
      j = std::move(p->jobs.front());
      p->jobs.pop_front();
    }
    int64_t t0 = thread_ns();
    try {
      run_job(p, j);
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lk(p->mu);
      p->err = e.what();
    }
    p->cpu_ns += thread_ns() - t0;
    p->done++;
    p->cv_idle.notify_all();
  }
}

VerbsKV::VerbsKV(const std::string& dev, int64_t bytes, bool gpu_buffer, int nqp, int max_layers, int flush_mode,
                 int cpu_core)
    : p_(new VerbsImpl) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  VCHECK(list, "ibv_get_device_list");
  for (int i = 0; i < n; ++i)
    if (dev == ibv_get_device_name(list[i])) p_->ctx = ibv_open_device(list[i]);
  ibv_free_device_list(list);
  if (!p_->ctx) throw std::runtime_error("verbs: device not found: " + dev);
  VCHECK(ibv_query_port(p_->ctx, p_->port, &p_->pattr) == 0, "ibv_query_port");
  ibv_query_gid(p_->ctx, p_->port, 0, &p_->gid);
  p_->pd = ibv_alloc_pd(p_->ctx);
  VCHECK(p_->pd, "ibv_alloc_pd");
  p_->cq = ibv_create_cq(p_->ctx, p_->sq_depth * nqp + 16, nullptr, nullptr, 0);
  VCHECK(p_->cq, "ibv_create_cq");
  for (int q = 0; q < nqp; ++q) {
    ibv_qp_init_attr ia{};
    ia.send_cq = p_->cq;
    ia.recv_cq = p_->cq;
    ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = p_->sq_depth;
    ia.cap.max_recv_wr = 1;
    ia.cap.max_send_sge = 1;
    ia.cap.max_recv_sge = 1;
    ibv_qp* qp = ibv_create_qp(p_->pd, &ia);
    VCHECK(qp, "ibv_create_qp");
    p_->qps.push_back(qp);
  }
  p_->bytes = (size_t)bytes;
  p_->gpu = gpu_buffer;
  if (gpu_buffer) {
    CUDACHK(cudaGetDevice(&p_->cuda_dev));
    CUDACHK(cudaMalloc(&p_->buf, p_->bytes));
    CUdevice cd;
    cuDeviceGet(&cd, p_->cuda_dev);
    cuDeviceGetAttribute(&p_->gdr_order, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, cd);
  } else {
    p_->buf = numa_host_alloc(p_->bytes, nic_numa(dev));
    memset(p_->buf, 0, p_->bytes);   // first touch on the bound node
  }
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  p_->mr = ibv_reg_mr(p_->pd, p_->buf, p_->bytes, access);
  VCHECK(p_->mr, "ibv_reg_mr");
  p_->max_layers = max_layers;
  p_->flush_mode = flush_mode;
  p_->cpu_core = cpu_core;
  if (gpu_buffer) {
    CUDACHK(cudaHostAlloc((void**)&p_->flags_host, max_layers * sizeof(uint32_t), cudaHostAllocMapped));
    memset(p_->flags_host, 0, max_layers * sizeof(uint32_t));
    CUDACHK(cudaHostGetDevicePointer((void**)&p_->flags_dev, p_->flags_host, 0));
  }
  p_->worker = std::thread(worker_main, p_);
}

std::string VerbsKV::local_info() const {
  std::ostringstream o;
  o << p_->pattr.lid << " " << (uint64_t)p_->buf << " " << p_->mr->rkey << " " << p_->qps.size();
  for (auto* q : p_->qps) o << " " << q->qp_num;
  for (int i = 0; i < 16; ++i) o << " " << (int)p_->gid.raw[i];
  return o.str();
}

void VerbsKV::connect(const std::string& remote) {
  std::istringstream in(remote);
  uint32_t lid, nq;
  uint64_t addr;
  in >> lid >> addr >> p_->rkey >> nq;
  p_->raddr = addr;
  if (nq != p_->qps.size()) throw std::runtime_error("verbs: QP count mismatch");
  std::vector<uint32_t> rq(nq);
  for (auto& x : rq) in >> x;
  ibv_gid rgid{};
  for (int i = 0; i < 16; ++i) {
    int v;
    in >> v;
    rgid.raw[i] = (uint8_t)v;
  }
  const bool roce = p_->pattr.link_layer == IBV_LINK_LAYER_ETHERNET;
  for (size_t q = 0; q < nq; ++q) {
    ibv_qp* qp = p_->qps[q];
    ibv_qp_attr a{};
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = p_->port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    VCHECK(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0, "INIT");
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = IBV_MTU_4096;
    a.dest_qp_num = rq[q];
    a.rq_psn = 0;
    a.max_dest_rd_atomic = 16;
    a.min_rnr_timer = 12;
    a.ah_attr.dlid = (uint16_t)lid;
    a.ah_attr.sl = 0;
    a.ah_attr.port_num = p_->port;
    if (roce) {
      a.ah_attr.is_global = 1;
      a.ah_attr.grh.dgid = rgid;
      a.ah_attr.grh.sgid_index = 0;
      a.ah_attr.grh.hop_limit = 255;
    }
    VCHECK(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                     IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0, "RTR");
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.timeout = 14;
    a.retry_cnt = 7;
    a.rnr_retry = 7;
    a.sq_psn = 0;
    a.max_rd_atomic = 16;
    VCHECK(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                     IBV_QP_MAX_QP_RD_ATOMIC) == 0, "RTS");
  }
}

void* VerbsKV::buffer() const { return p_->buf; }
int64_t VerbsKV::bytes() const { return (int64_t)p_->bytes; }
uint64_t VerbsKV::flags_device_ptr() const { return (uint64_t)p_->flags_dev; }
int VerbsKV::gdr_writes_ordering() const { return p_->gdr_order; }

static void submit(VerbsImpl* p, Job&& j) {
  {
    std::lock_guard<std::mutex> lk(p->mu);
    if (!p->err.empty()) throw std::runtime_error("verbs worker failed earlier: " + p->err);
    p->jobs.push_back(std::move(j));
    p->submitted++;
  }
  p->cv.notify_one();
}

void VerbsKV::post_reads(const int64_t* segs, const int* layer_ptr, int n_layers, uint32_t seq) {
  if (n_layers > p_->max_layers) throw std::runtime_error("verbs: too many layers");
  Job j;
  j.n_layers = n_layers;
  j.lp.assign(layer_ptr, layer_ptr + n_layers + 1);
  j.segs.assign(segs, segs + 4 * (size_t)layer_ptr[n_layers]);
  j.seq = seq;
  submit(p_, std::move(j));
}

void VerbsKV::write_sync(const int64_t* segs, int n) {
  Job j;
  j.n_layers = 1;
  j.lp = {0, n};
  j.segs.assign(segs, segs + 4 * (size_t)n);
  j.write = true;
  submit(p_, std::move(j));
  wait_idle();
}

void VerbsKV::wait_idle() {
  std::unique_lock<std::mutex> lk(p_->mu);
  p_->cv_idle.wait(lk, [this] { return p_->done.load() >= p_->submitted || !p_->err.empty(); });
  if (!p_->err.empty()) throw std::runtime_error("verbs worker: " + p_->err);
}

double VerbsKV::worker_cpu_s() const { return p_->cpu_ns.load() / 1e9; }
int64_t VerbsKV::requests_done() const { return p_->done.load(); }

void VerbsKV::destroy() {
  if (!p_->ctx) return;
  {
    std::lock_guard<std::mutex> lk(p_->mu);
    p_->stop = true;
  }
  p_->cv.notify_all();
  if (p_->worker.joinable()) p_->worker.join();
  for (auto* q : p_->qps) ibv_destroy_qp(q);
  p_->qps.clear();
  if (p_->mr) ibv_dereg_mr(p_->mr);
  if (p_->cq) ibv_destroy_cq(p_->cq);
  if (p_->pd) ibv_dealloc_pd(p_->pd);
  ibv_close_device(p_->ctx);
  p_->ctx = nullptr;
  if (p_->buf) {
    if (p_->gpu) cudaFree(p_->buf);
    else munmap(p_->buf, p_->bytes);
  }
  if (p_->flags_host) cudaFreeHost(p_->flags_host);
  p_->buf = nullptr;
  p_->flags_host = nullptr;
}

VerbsKV::~VerbsKV() {
  try {
    destroy();
  } catch (...) {
  }
  delete p_;
}

}  // namespace hpoc
