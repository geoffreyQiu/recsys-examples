// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Fixed-slot adapter lifecycle only. Queueing, transfer progress, completion,
// and shutdown execute in corelib's compiled CacheOwner implementation.
using kvcache_manager::CacheOwner;

thread_local std::unique_ptr<CacheOwner> cache_owner;

std::vector<at::Tensor> owner_start(at::Tensor anchor) {
  TORCH_CHECK(!cache_owner&&pending.empty(),"owner already active or request pending");
  cache_owner=std::make_unique<CacheOwner>(anchor.get_device());
  try {
    cache_owner->initialize(ints({}));
    resident_pools=cache_owner->sync([]{return call("kvcache_manager_ops::get_cache_tables",{ints({})})[0].toTensorVector();});
    resident_fastpath_available=c10::Dispatcher::singleton().findSchema(
        {"kvcache_manager_ops::try_resident_slots",""}).has_value();
    if(resident_fastpath_available) {
      resident_cpu_slots=at::empty({request_batch_limit},
          at::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true));
      resident_gpu_slots=at::empty({request_batch_limit},anchor.options().dtype(at::kInt));
    }
    resident_stats.fill(0);
    return resident_pools;
  } catch(...) {
    cache_owner.reset();resident_pools.clear();resident_cpu_slots=at::Tensor();
    resident_gpu_slots=at::Tensor();resident_fastpath_available=false;throw;
  }
}
void owner_flush(at::Tensor anchor) {
  TORCH_CHECK(cache_owner&&pending.empty(),"owner flush requires retired requests");
  auto* owner=cache_owner.get();owner->sync([owner]{owner->flush();});
}
at::Tensor owner_stats(at::Tensor anchor) {
  TORCH_CHECK(cache_owner,"owner is not active");
  auto* owner=cache_owner.get();return owner->sync([owner]{return owner->stats();});
}
void owner_evict(at::Tensor ids) {
  TORCH_CHECK(cache_owner&&pending.empty(),"owner eviction requires retired requests");
  auto* owner=cache_owner.get();owner->sync([owner,ids]{
    owner->flush();call("kvcache_manager_ops::evict_kvcache",{ids,true,ids});
  });
}
void owner_stop(at::Tensor anchor) {
  TORCH_CHECK(cache_owner,"owner is not active");
  TORCH_CHECK(pending.empty(),"retire all requests before stopping the owner");
  auto* owner=cache_owner.get();std::exception_ptr error;
  try {owner->sync([owner]{owner->flush();});}catch(...){error=std::current_exception();}
  try {owner->sync([owner,anchor]{owner->shutdown(anchor);},true);}catch(...){if(!error)error=std::current_exception();}
  cache_owner.reset();
  clear_overlap_graphs();
  resident_pools.clear();resident_cpu_slots=at::Tensor();resident_gpu_slots=at::Tensor();
  resident_fastpath_available=false;
  if(error)std::rethrow_exception(error);
}
