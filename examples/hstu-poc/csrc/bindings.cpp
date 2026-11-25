// Python bindings for the POC transports and helpers. Tensors in, raw pointers to the C++ layer.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <optional>

#include "gin_kv.h"
#include "kvutil.h"
#include "verbs_kv.h"
#include "raw_gdaki.h"

namespace py = pybind11;
using namespace hpoc;

static cudaStream_t cur() { return at::cuda::getCurrentCUDAStream().stream(); }
template <typename T>
static T* ptr_or_null(const std::optional<torch::Tensor>& t) {
  return t.has_value() ? (T*)t->data_ptr() : nullptr;
}
static void need_cuda(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), what, " must be a contiguous CUDA tensor");
}

static torch::Tensor gin_local_tensor(GinKV& g) {
  auto opts = torch::TensorOptions().dtype(torch::kUInt8);
  if (g.on_host()) {
    opts = opts.device(torch::kCPU);
  } else {
    int dev = 0;
    cudaGetDevice(&dev);
    opts = opts.device(torch::kCUDA, dev);
  }
  return torch::from_blob(g.local_ptr(), {g.local_bytes()}, [](void*) {}, opts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gin_unique_id", []() { return py::bytes(gin_unique_id()); });

  py::class_<GinKV>(m, "GinKV")
      .def(py::init([](py::bytes uid, int rank, int nranks, int64_t bytes, bool on_host, int n_ctx, int qdepth,
                       bool strong, int n_signals) {
             return new GinKV(std::string(uid), rank, nranks, bytes, on_host, n_ctx, qdepth, strong, n_signals);
           }),
           py::arg("uid"), py::arg("rank"), py::arg("nranks"), py::arg("local_bytes"), py::arg("on_host"),
           py::arg("n_ctx") = 2, py::arg("queue_depth") = 1024, py::arg("strong_signals") = true,
           py::arg("n_signals") = 4)
      .def("local", &gin_local_tensor)
      .def("rank", &GinKV::rank)
      .def("nranks", &GinKV::nranks)
      .def("post_gets",
           [](GinKV& g, torch::Tensor segs, torch::Tensor layer_ptr, torch::Tensor tickets, int peer, int ctx0,
              int n_ctx, std::optional<torch::Tensor> t_post) {
             need_cuda(segs, "segs");
             need_cuda(layer_ptr, "layer_ptr");
             need_cuda(tickets, "tickets");
             TORCH_CHECK(segs.scalar_type() == torch::kInt64 && segs.dim() == 2 && segs.size(1) == 4, "segs [n,4] int64");
             TORCH_CHECK(layer_ptr.scalar_type() == torch::kInt32, "layer_ptr int32");
             TORCH_CHECK(tickets.numel() * tickets.element_size() >= 16 * n_ctx * (layer_ptr.numel() - 1),
                         "tickets too small");
             TORCH_CHECK(n_ctx >= 1 && n_ctx <= 32, "n_ctx in [1, 32]");
             g.post_gets((const Seg*)segs.data_ptr(), layer_ptr.data_ptr<int>(), (int)layer_ptr.numel() - 1,
                         tickets.data_ptr(), peer, ctx0, n_ctx, ptr_or_null<uint64_t>(t_post), cur());
           },
           py::arg("segs"), py::arg("layer_ptr"), py::arg("tickets"), py::arg("peer"), py::arg("ctx0") = 0,
           py::arg("n_ctx") = 2, py::arg("t_post") = py::none())
      .def("wait_layer",
           [](GinKV& g, torch::Tensor tickets, int layer, int ctx0, int n_ctx, std::optional<torch::Tensor> t_done) {
             g.wait_layer(tickets.data_ptr(), layer, ctx0, n_ctx, ptr_or_null<uint64_t>(t_done), cur());
           },
           py::arg("tickets"), py::arg("layer"), py::arg("ctx0") = 0, py::arg("n_ctx") = 2,
           py::arg("t_done") = py::none())
      .def("put_segs",
           [](GinKV& g, torch::Tensor segs, std::optional<torch::Tensor> ticket, int peer, int ctx, int threads) {
             need_cuda(segs, "segs");
             g.put_segs((const Seg*)segs.data_ptr(), (int)segs.size(0), ptr_or_null<void>(ticket), peer, ctx, threads,
                        cur());
           },
           py::arg("segs"), py::arg("ticket"), py::arg("peer"), py::arg("ctx") = 1, py::arg("threads") = 32)
      .def("put_multi",
           [](GinKV& g, torch::Tensor segs, int peer, int ctx0, int n_ctx) {
             need_cuda(segs, "segs");
             g.put_multi((const Seg*)segs.data_ptr(), (int)segs.size(0), peer, ctx0, n_ctx, cur());
           },
           py::arg("segs"), py::arg("peer"), py::arg("ctx0"), py::arg("n_ctx") = 2)
      .def("fence",
           [](GinKV& g, int peer, int ctx0, int n_ctx, int64_t remote_off, int64_t local_off, int sig,
              std::optional<torch::Tensor> t_done) {
             g.fence(peer, ctx0, n_ctx, remote_off, local_off, sig, ptr_or_null<uint64_t>(t_done), cur());
           },
           py::arg("peer"), py::arg("ctx0"), py::arg("n_ctx"), py::arg("remote_off"), py::arg("local_off"),
           py::arg("sig") = -1, py::arg("t_done") = py::none())
      .def("wait_ticket", [](GinKV& g, torch::Tensor ticket, int ctx) { g.wait_ticket(ticket.data_ptr(), ctx, cur()); },
           py::arg("ticket"), py::arg("ctx") = 1)
      .def("commit", [](GinKV& g, int peer, int ctx, int sig) { g.commit(peer, ctx, sig, cur()); }, py::arg("peer"),
           py::arg("ctx") = 1, py::arg("sig") = 0)
      .def("wait_signal", [](GinKV& g, int ctx, int sig, uint64_t v) { g.wait_signal(ctx, sig, v, cur()); },
           py::arg("ctx"), py::arg("sig"), py::arg("value"))
      .def("read_signal", &GinKV::read_signal_host, py::arg("ctx"), py::arg("sig"))
      .def("destroy", &GinKV::destroy);

  m.def("fill_pattern_host",
        [](torch::Tensor dst, int64_t user, int64_t layer, int64_t word0, int64_t seed) {
          TORCH_CHECK(!dst.is_cuda() && dst.is_contiguous() && dst.nbytes() % 8 == 0, "host uint8 view, 8-B multiple");
          py::gil_scoped_release nogil;
          fill_pattern_host(dst.data_ptr(), (int64_t)dst.nbytes() / 8, user, layer, word0, seed);
        },
        py::arg("dst"), py::arg("user"), py::arg("layer"), py::arg("word0"), py::arg("seed"));
  m.def("check_pattern_gpu",
        [](torch::Tensor base, torch::Tensor rows, int64_t seed, torch::Tensor bad) {
          need_cuda(rows, "rows");
          check_pattern_gpu(base.data_ptr(), rows.data_ptr<int64_t>(), (int)rows.size(0), seed,
                            (unsigned long long*)bad.data_ptr(), cur());
        },
        py::arg("base"), py::arg("rows"), py::arg("seed"), py::arg("bad"));
  m.def("fill_pattern_gpu",
        [](torch::Tensor base, torch::Tensor rows, int64_t seed) {
          need_cuda(rows, "rows");
          fill_pattern_gpu(base.data_ptr(), rows.data_ptr<int64_t>(), (int)rows.size(0), seed, cur());
        },
        py::arg("base"), py::arg("rows"), py::arg("seed"));
  m.def("check_pattern_host",
        [](torch::Tensor base, torch::Tensor rows, int64_t seed) {
          TORCH_CHECK(!rows.is_cuda(), "rows on CPU");
          py::gil_scoped_release nogil;
          return check_pattern_host(base.data_ptr(), rows.data_ptr<int64_t>(), (int)rows.size(0), seed);
        },
        py::arg("base"), py::arg("rows"), py::arg("seed"));
  py::class_<VerbsKV>(m, "VerbsKV")
      .def(py::init<const std::string&, int64_t, bool, int, int, int, int>(), py::arg("dev"), py::arg("bytes"),
           py::arg("gpu_buffer"), py::arg("nqp") = 2, py::arg("max_layers") = 64, py::arg("flush_mode") = 0,
           py::arg("cpu_core") = -1)
      .def("local_info", &VerbsKV::local_info)
      .def("connect", &VerbsKV::connect)
      .def("local", [](VerbsKV& v) {
        auto opts = torch::TensorOptions().dtype(torch::kUInt8);
        if (v.flags_device_ptr()) {
          int dev = 0;
          cudaGetDevice(&dev);
          opts = opts.device(torch::kCUDA, dev);
        }
        return torch::from_blob(v.buffer(), {v.bytes()}, [](void*) {}, opts);
      })
      .def("flags_ptr", &VerbsKV::flags_device_ptr)
      .def("post_reads",
           [](VerbsKV& v, torch::Tensor segs, torch::Tensor layer_ptr, uint32_t seq) {
             TORCH_CHECK(!segs.is_cuda() && segs.scalar_type() == torch::kInt64 && segs.is_contiguous(), "segs cpu int64");
             TORCH_CHECK(!layer_ptr.is_cuda() && layer_ptr.scalar_type() == torch::kInt32, "layer_ptr cpu int32");
             v.post_reads(segs.data_ptr<int64_t>(), layer_ptr.data_ptr<int>(), (int)layer_ptr.numel() - 1, seq);
           })
      .def("post_writes",
           [](VerbsKV& vk, torch::Tensor segs, torch::Tensor layer_ptr, uint32_t seq, int64_t fr, int64_t fl) {
             TORCH_CHECK(!segs.is_cuda() && segs.scalar_type() == torch::kInt64 && segs.is_contiguous(), "segs cpu int64");
             vk.post_writes(segs.data_ptr<int64_t>(), layer_ptr.data_ptr<int>(), (int)layer_ptr.numel() - 1, seq, fr, fl);
           })
      .def("wflags_ptr", &VerbsKV::wflags_device_ptr)
      .def("done_ptr", &VerbsKV::done_device_ptr)
      .def("write_sync",
           [](VerbsKV& v, torch::Tensor segs) {
             py::gil_scoped_release nogil;
             v.write_sync(segs.data_ptr<int64_t>(), (int)segs.size(0));
           })
      .def("wait_idle", [](VerbsKV& v) {
        py::gil_scoped_release nogil;
        v.wait_idle();
      })
      .def("worker_cpu_s", &VerbsKV::worker_cpu_s)
      .def("requests_done", &VerbsKV::requests_done)
      .def("gdr_writes_ordering", &VerbsKV::gdr_writes_ordering)
      .def("destroy", &VerbsKV::destroy);
  py::class_<RawGdaki>(m, "RawGdaki")
      .def(py::init<const std::string&, int64_t, int, int>(), py::arg("dev"), py::arg("bytes"), py::arg("nqp") = 4,
           py::arg("sq_depth") = 4096)
      .def("local_info", &RawGdaki::local_info)
      .def("connect", &RawGdaki::connect)
      .def("local", [](RawGdaki& r) {
        int dev = 0;
        cudaGetDevice(&dev);
        return torch::from_blob(r.buffer(), {r.bytes()}, [](void*) {},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, dev));
      })
      .def("nqp", &RawGdaki::nqp)
      .def("post_gets",
           [](RawGdaki& r, torch::Tensor segs, torch::Tensor layer_ptr, torch::Tensor tickets,
              std::optional<torch::Tensor> t_post) {
             need_cuda(segs, "segs");
             need_cuda(layer_ptr, "layer_ptr");
             TORCH_CHECK(tickets.numel() >= (int64_t)r.nqp() * (layer_ptr.numel() - 1), "tickets too small");
             r.post_gets((const Seg*)segs.data_ptr(), layer_ptr.data_ptr<int>(), (int)layer_ptr.numel() - 1,
                         (uint64_t*)tickets.data_ptr(), ptr_or_null<uint64_t>(t_post), cur());
           },
           py::arg("segs"), py::arg("layer_ptr"), py::arg("tickets"), py::arg("t_post") = py::none())
      .def("wait_layer",
           [](RawGdaki& r, torch::Tensor tickets, int layer, std::optional<torch::Tensor> t_done) {
             r.wait_layer((const uint64_t*)tickets.data_ptr(), layer, ptr_or_null<uint64_t>(t_done), cur());
           },
           py::arg("tickets"), py::arg("layer"), py::arg("t_done") = py::none())
      .def("destroy", &RawGdaki::destroy);
  m.def("seq_inc", [](torch::Tensor seq) { seq_inc_gpu((uint32_t*)seq.data_ptr(), cur()); });
  m.def("set_flag_seq", [](int64_t flags_addr, int layer, torch::Tensor seq) {
    set_flag_seq_gpu((volatile uint32_t*)flags_addr, layer, (const uint32_t*)seq.data_ptr(), cur());
  });
  m.def("wait_flags_seq",
        [](int64_t flags_addr, int layer, torch::Tensor seq, std::optional<torch::Tensor> stamp) {
          wait_flags_seq_gpu((const volatile uint32_t*)flags_addr, layer, (const uint32_t*)seq.data_ptr(),
                             ptr_or_null<uint64_t>(stamp), cur());
        },
        py::arg("flags_addr"), py::arg("layer"), py::arg("seq"), py::arg("stamp") = py::none());
  m.def("spin", [](int64_t ns) { spin_gpu(ns, cur()); });
  m.def("stamp", [](torch::Tensor dst) { stamp_gpu((uint64_t*)dst.data_ptr(), cur()); });
  m.def("wait_flag",
        [](int64_t flag_addr, uint32_t value, std::optional<torch::Tensor> stamp) {
          wait_flag_gpu((const volatile uint32_t*)flag_addr, value, ptr_or_null<uint64_t>(stamp), cur());
        },
        py::arg("flag_addr"), py::arg("value"), py::arg("stamp") = py::none());
}
