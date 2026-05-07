#include "native_host_kvcache_manager_impl.h"
#include "gpu_kvcache_manager_impl.h"

PYBIND11_MODULE(kvcache_cpp, m) {
  py::class_<kvcache::HostKVStorageImpl>(m, "HostKVStorageImpl")
    .def(py::init<int, int, int, int, int64_t, int64_t, int64_t, int64_t, int>(), 
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_tokens_per_chunk"),
         py::arg("capacity_per_layer"),
         py::arg("max_batch_size"),
         py::arg("max_sequence_length"),
         py::arg("device_idx"))
    .def("register_gpu_cache_table", &kvcache::HostKVStorageImpl::register_gpu_cache_table)
    .def("get_kvdata_length", &kvcache::HostKVStorageImpl::get_kvdata_length)
    .def("get_kvdata_tensor", &kvcache::HostKVStorageImpl::get_kvdata_tensor)
    .def("onload_kvcache", &kvcache::HostKVStorageImpl::onload_kvcache)
    .def("offload_kvcache", &kvcache::HostKVStorageImpl::offload_kvcache)
    .def("finish_offload", &kvcache::HostKVStorageImpl::finish_offload)
    .def("cancel_offload", &kvcache::HostKVStorageImpl::cancel_offload)
  ;

  py::class_<kvcache::GPUKVCacheMangerImpl>(m, "GPUKVCacheMangerImpl")
    .def(py::init<int, int, int, int, int, int, int, int, int, int>(),
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_tokens_per_chunk"),
         py::arg("num_primary_cache_pages"),
         py::arg("num_buffer_pages"),
         py::arg("max_batch_size"),
         py::arg("max_sequence_length"),
         py::arg("device_idx"))
    .def("lookup", &kvcache::GPUKVCacheMangerImpl::lookup)
    .def("allocate", &kvcache::GPUKVCacheMangerImpl::allocate)
    .def("evict", &kvcache::GPUKVCacheMangerImpl::evict)
    .def("evict_all", &kvcache::GPUKVCacheMangerImpl::evict_all)
    .def("check_for_offload", &kvcache::GPUKVCacheMangerImpl::check_for_offload)
    .def("acquire_offload_pages", &kvcache::GPUKVCacheMangerImpl::acquire_offload_pages)
    .def("release_offload_pages", &kvcache::GPUKVCacheMangerImpl::release_offload_pages)
  ;

  py::class_<kvcache::KVOnloadHandle>(m, "KVOnloadHandle")
    .def(py::init<int>(), py::arg("num_layers"))
    // .def("reset", &kvcache::KVOnloadHandle::reset)
    .def("wait_host", &kvcache::KVOnloadHandle::wait_host)
  ;

  py::class_<kvcache::KVOffloadHandle>(m, "KVOffloadHandle")
    .def(py::init<int>(), py::arg("num_layers"))
    // .def("reset", &kvcache::KVOffloadHandle::reset)
  ;
}