// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <torch/library.h>
#include <torch/types.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <unordered_map>
#include <memory>
#include <vector>
#include <map>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <array>
#include <unordered_set>
#include <cuda_runtime_api.h>
#include "cache_owner.h"

void single_fill_cuda(at::Tensor,at::Tensor,at::Tensor,at::Tensor);
std::tuple<at::Tensor,at::Tensor> single_gather_cuda(at::Tensor,at::Tensor);

namespace {
using Stack=std::vector<c10::IValue>;
Stack call(const char* name,Stack args) {
  c10::Dispatcher::singleton().findSchemaOrThrow(name,"").callBoxed(&args);
  return args;
}
at::Tensor ints(const std::vector<int64_t>& values) {
  auto t=at::empty({(int64_t)values.size()},at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  std::copy(values.begin(),values.end(),t.data_ptr<int64_t>());
  return t;
}
// Transfer ownership and progress live in the shared corelib library. Only
// converted-graph bookkeeping remains specific to the fixed-slot adapter.
struct Request : kvcache_manager::OwnerTransferState {
  std::vector<bool> filled;
  int64_t misses=0,pack_calls=0,wait_calls=0,gather_calls=0;
  bool ready=false,finished=false,offloaded=false;
};
thread_local std::unordered_map<int64_t,std::shared_ptr<Request>> pending;
thread_local int64_t serial=1;
thread_local int64_t request_batch_limit=8;
thread_local bool legacy_offload_active=false;
thread_local std::map<int64_t,std::vector<int64_t>> snapshot_contexts;
// Capture records a GPU body for one (batch, missing-row-count) shape. Every replay creates a
// real request/lease, validates context, waits for transfers and retires it.
thread_local std::shared_ptr<Request> capture_request;
thread_local at::Tensor capture_slots,capture_rows,capture_missing_rows;
thread_local std::optional<std::tuple<at::Tensor,at::Tensor,at::Tensor>> resumed_prepare;
// Serialized requests own these reusable slots until retire() has fenced GPU
// completion. Pending offloads never retain references to these staging buffers.
thread_local bool resident_fastpath=true,resident_fastpath_available=false;
thread_local at::Tensor resident_cpu_slots,resident_gpu_slots;
thread_local std::vector<at::Tensor> resident_pools;
thread_local std::array<int64_t,5> resident_stats{};
Request& request(at::Tensor ticket);
#include "single_overlap.h"
#include "single_owner.h"
Request& request(at::Tensor ticket) {
  TORCH_CHECK(ticket.device().is_cpu()&&ticket.scalar_type()==at::kLong&&ticket.numel()==1,"invalid request ticket");
  if(ticket.data_ptr<int64_t>()[0]==-1&&capture_request)return *capture_request;
  auto it=pending.find(ticket.data_ptr<int64_t>()[0]);
  TORCH_CHECK(it!=pending.end(),"unknown/retired request ticket");
  return *it->second;
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> prepare_impl(
    at::Tensor ids,at::Tensor anchor,int64_t length,int64_t banks,
    const std::optional<at::Tensor>& context,
    const std::optional<at::Tensor>& resident_destination) {
  if(capture_request) {
    TORCH_CHECK(ids.numel()==capture_slots.numel(),"capture batch changed");
    return {ints({-1}),capture_rows,capture_slots};
  }
  if(resumed_prepare) {
    auto result=*resumed_prepare;
    auto& r=request(std::get<0>(result));
    TORCH_CHECK(ids.equal(r.input_ids)&&banks==(int64_t)r.pools.size()&&length==r.pools[0].size(2),"resumed request contract changed");
    resumed_prepare.reset();return result;
  }
  TORCH_CHECK(pending.empty(),"single-program runtime requires serialized requests; retire the previous ticket");
  TORCH_CHECK(ids.device().is_cpu()&&ids.scalar_type()==at::kLong&&ids.dim()==1&&ids.numel()>=1&&ids.numel()<=request_batch_limit,"CPU snapshot IDs exceed the configured request batch limit");
  TORCH_CHECK(anchor.is_cuda()&&banks>0,"CUDA anchor and positive bank count required");
  ids=ids.contiguous();
  if(context.has_value()) {
    TORCH_CHECK(context->dim()>=1&&context->size(0)==ids.numel(),"cache context must be batched");
    auto values=context->to(at::TensorOptions().dtype(at::kLong).device(at::kCPU)).contiguous().reshape({ids.numel(),-1});
    for(int64_t i=0;i<ids.numel();++i) {
      auto first=values.data_ptr<int64_t>()+i*values.size(1);
      std::vector<int64_t> key(first,first+values.size(1));
      auto identity=ids.data_ptr<int64_t>()[i];
      auto previous=snapshot_contexts.find(identity);
      TORCH_CHECK(previous==snapshot_contexts.end()||previous->second==key,
                  "snapshot_context_changed: use a new snapshot_id when the attention normalization context changes");
      snapshot_contexts[identity]=std::move(key);
    }
  }
  auto r=std::make_shared<Request>();
  r->input_ids=ids;
  std::unordered_map<int64_t,int64_t> seen;
  std::vector<int64_t> unique,first,inverse;
  for(int64_t i=0;i<ids.numel();++i) {
    auto id=ids.data_ptr<int64_t>()[i];
    if(!seen.count(id)){seen[id]=unique.size();unique.push_back(id);first.push_back(i);}
    inverse.push_back(seen[id]);
  }
  r->ids=ints(unique);r->lengths=at::full_like(r->ids,length);
  r->count=unique.size();r->filled.resize(banks,false);
  auto metadata=[&] {
    r->found=call("kvcache_manager_ops::lookup",{r->ids,r->lengths,ids})[0].toTensorVector();
    r->pools=call("kvcache_manager_ops::get_cache_tables",{ids})[0].toTensorVector();
    r->allocation=call("kvcache_manager_ops::allocate",{r->ids,r->lengths,r->found[1],r->found[5]})[0].toTensorVector();
  };
  bool compact=false;
  at::Tensor compact_cpu,compact_slots;
  if(cache_owner) {
    auto* owner=cache_owner.get();
    const bool try_compact=resident_fastpath&&resident_fastpath_available;
    if(try_compact) {
      ++resident_stats[0];
      compact_cpu=resident_cpu_slots.narrow(0,0,ids.numel());
      if(resident_destination.has_value()) {
        const auto& dst=*resident_destination;
        TORCH_CHECK(dst.device()==anchor.device()&&dst.scalar_type()==at::kInt&&
                    dst.dim()==1&&dst.numel()==ids.numel()&&dst.is_contiguous(),
                    "resident replay slot buffer has invalid geometry/device/dtype");
      }
    }
    owner->sync([&] {
      if(try_compact)
        compact=call("kvcache_manager_ops::try_resident_slots",{ids,length,banks,compact_cpu})[0].toBool();
      if(compact)return;
      owner->admission(r->ids);
      metadata();
      r->metadata.record(at::cuda::getCurrentCUDAStream());
    });
    if(compact) {
      // Runtime pool references are caller-thread-local, not owner TLS.
      r->pools=resident_pools;
      ++resident_stats[1];
      r->compact_resident=true;
      r->onboard_complete=true;
      compact_slots=resident_destination.has_value()?*resident_destination:
          resident_gpu_slots.narrow(0,0,ids.numel());
      // Owner performed only CPU metadata work. Upload directly on the model
      // stream, optionally into its captured slot buffer: no cross-stream event
      // or intermediate GPU slot copy is needed on a complete resident hit.
      compact_slots.copy_(compact_cpu,/*non_blocking=*/true);
      ++resident_stats[resident_destination.has_value()?3:4];
    } else {
      if(try_compact)++resident_stats[2];
      r->metadata.block(at::cuda::getCurrentCUDAStream());
    }
  } else {
    if(legacy_offload_active) {
      auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
      do {
        auto completed=call("kvcache_manager_ops::offload_reap_completed",{ids})[0].toTensor();
        for(int64_t i=0;i<completed.size(0);++i)
          TORCH_CHECK(completed.data_ptr<int64_t>()[3*i+2]==0,"FlexKV offload failed");
        legacy_offload_active=call("kvcache_manager_ops::offload_pending",{ids})[0].toInt()>0;
        if(call("kvcache_manager_ops::fixed_allocation_available",{r->ids})[0].toBool())break;
        TORCH_CHECK(legacy_offload_active&&std::chrono::steady_clock::now()<deadline,"cache admission cannot obtain an unpinned slot");
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      } while(true);
    }
    metadata();
  }
  std::vector<int64_t> rows;
  if(!compact)for(int64_t i=0;i<r->count;++i){
    auto cached=r->found[1].data_ptr<int>()[i];
    TORCH_CHECK(cached==0||cached==length,"partial immutable snapshot is unsupported");
    if(cached==0)rows.push_back(first[i]);
    if(r->found[5].data_ptr<int>()[i]>0)++r->cpu_hits;
  }
  r->misses=rows.size();
  TORCH_CHECK((int64_t)r->pools.size()==banks,"compiled bank count differs from native runtime");
  TORCH_CHECK(compact||r->allocation[0].numel()==r->count,"fixed-slot allocator must return one slot per unique history");
  if(cache_owner) {
    if(!r->cpu_hits)r->onboard_complete=true;
    else {auto* owner=cache_owner.get();owner->post([owner,r]{owner->onboard(r);});}
  } else {
    auto launch=call("kvcache_manager_ops::onboard_launch",{r->ids,r->lengths,c10::List<at::Tensor>(r->found),r->allocation[0],r->allocation[2]});
    r->tasks=launch[2].toTensor();
  }
  auto slots=compact?compact_slots:(r->count==ids.numel()?r->allocation[0]:
      r->allocation[0].index_select(0,ints(inverse).to(anchor.device())));
  // Fixed-size effectful output. The pure packing op below owns the unbacked
  // missing-row dimension; this avoids with_effects shape-binding failures.
  rows.resize(ids.numel(),-1);
  auto id=serial++;
  pending[id]=std::move(r);
  return {ints({id}),ints(rows),slots};
}


std::tuple<at::Tensor,at::Tensor,at::Tensor> prepare(
    at::Tensor ids,at::Tensor anchor,int64_t length,int64_t banks,
    const std::optional<at::Tensor>& context) {
  return prepare_impl(ids,anchor,length,banks,context,std::nullopt);
}
void set_resident_fastpath(bool enabled) {
  TORCH_CHECK(pending.empty(),"change resident fast path only between retired requests");
  resident_fastpath=enabled;resident_stats.fill(0);
}
bool resident_fastpath_enabled() {
  return resident_fastpath&&resident_fastpath_available&&cache_owner!=nullptr;
}
at::Tensor resident_fastpath_stats() {
  return ints(std::vector<int64_t>(resident_stats.begin(),resident_stats.end()));
}

std::vector<at::Tensor> pack(std::vector<at::Tensor> xs,at::Tensor rows,at::Tensor slots,at::Tensor ticket) {
  auto& r=request(ticket);++r.pack_calls;
  TORCH_CHECK(!xs.empty()&&rows.device().is_cpu()&&rows.numel()==xs[0].size(0)&&slots.numel()==rows.numel(),"invalid missing-row mapping");
  auto indices=(capture_request && &r==capture_request.get())?capture_missing_rows:
      rows.narrow(0,0,r.misses).to(xs[0].device());
  std::vector<at::Tensor> out;out.reserve(xs.size()+1);
  for(const auto& x:xs)out.push_back(x.index_select(0,indices));
  out.push_back(slots.index_select(0,indices));
  return out;
}

at::Tensor fill(at::Tensor pool,at::Tensor slots,at::Tensor key,at::Tensor value,at::Tensor ticket,int64_t bank) {
  auto& r=request(ticket);
  TORCH_CHECK(bank>=0&&bank<(int64_t)r.filled.size()&&!r.filled[bank],"bank filled twice or invalid bank");
  TORCH_CHECK(pool.data_ptr()==r.pools[bank].data_ptr(),"fill must write registered pool storage, not a functionalization copy");
  TORCH_CHECK(key.size(0)==r.misses,"producer rows disagree with lookup");
  single_fill_cuda(pool,slots,key,value);
  r.filled[bank]=true;
  return ticket.clone();
}

at::Tensor wait(at::Tensor ticket,at::Tensor filled,at::Tensor ordering) {
  auto& r=request(ticket);++r.wait_calls;
  // Mark the actual first history-read dependency in the captured GPU DAG.
  // This is an empty graph node, never a host wait or a spinning GPU kernel.
  if(capture_request && &r==capture_request.get() && r.wait_calls==1)
    mark_overlap_boundary();
  TORCH_CHECK(filled.equal(ticket),"fill ticket mismatch");
  if(!r.ready){
    if(cache_owner) {
      // AOTI's Python boxed runner may hold the GIL. Owner-side TensorImpl
      // reference changes can need it, so never retain it while waiting.
      // Declare before the request lock: unlock before restoring the GIL.
      kvcache_manager::OwnerWaitGuard release_gil;
      cache_owner->check();
      std::unique_lock<std::mutex> lock(r.mutex);
      TORCH_CHECK(r.cv.wait_for(lock,std::chrono::seconds(30),[&]{return r.onboard_complete||r.error;}),"onboarding readiness timed out");
      if(r.error)std::rethrow_exception(r.error);
    } else call("kvcache_manager_ops::onboard_wait",{r.tasks,ordering});
    r.ready=true;
  }
  return ticket.clone();
}

std::tuple<at::Tensor,at::Tensor> gather(at::Tensor pool,at::Tensor slots,at::Tensor ready,int64_t bank) {
  auto& r=request(ready);++r.gather_calls;
  TORCH_CHECK(r.ready&&bank>=0&&bank<(int64_t)r.filled.size()&&r.filled[bank],"gather before fill/readiness");
  TORCH_CHECK(pool.data_ptr()==r.pools[bank].data_ptr(),"gather must read registered pool storage");
  return single_gather_cuda(pool,slots);
}

at::Tensor offload(at::Tensor ticket,at::Tensor ordering,at::Tensor enabled) {
  auto& r=request(ticket);
  TORCH_CHECK(!r.finished,"request finished twice");
  for(bool filled:r.filled)TORCH_CHECK(filled,"cannot publish an incomplete snapshot");
  TORCH_CHECK(enabled.device().is_cpu()&&enabled.scalar_type()==at::kBool&&enabled.numel()==1,"CPU offload flag required");
  if(capture_request && &r==capture_request.get()) {
    TORCH_CHECK(!enabled.data_ptr<bool>()[0],"publication must run outside CUDA Graph capture");
    r.finished=true;return ticket.clone();
  }
  r.done.record(at::cuda::getCurrentCUDAStream());
  if(enabled.data_ptr<bool>()[0]) {
    // Publish only after the producer stream has completed all bank writes.
    // This runtime uses serialized requests and an explicit completion fence.
    if(cache_owner) cache_owner->submit_offload(pending.at(ticket.data_ptr<int64_t>()[0]));
    else {
      r.done.synchronize();
      std::vector<at::Tensor> mapping{ints({-1})};
      call("kvcache_manager_ops::offload_launch",{r.ids,r.lengths,r.found[1],r.found[5],r.found[2],r.found[3],r.allocation[0],r.allocation[2],c10::List<at::Tensor>(mapping),ordering});
      legacy_offload_active=true;
    }
    r.offloaded=true;
  }
  r.finished=true;
  return ticket.clone();
}

at::Tensor retire(at::Tensor ticket) {
  auto& r=request(ticket);
  TORCH_CHECK(r.finished,"cannot retire unfinished request");
  {
    kvcache_manager::OwnerWaitGuard release_gil;
    r.done.synchronize();
  }
  auto stats=ints({r.count,r.misses,r.cpu_hits,r.pack_calls,r.wait_calls,r.gather_calls,(int64_t)r.filled.size(),r.offloaded});
  pending.erase(ticket.data_ptr<int64_t>()[0]);
  return stats;
}
void reset(at::Tensor anchor) {
  TORCH_CHECK(pending.empty(),"failed/incomplete request exists; discard this process/runtime");
}
void reset_context(at::Tensor anchor) {
  TORCH_CHECK(pending.empty(),"cannot reset context during an active request");
  snapshot_contexts.clear();
}
void set_request_limit(int64_t limit) {
  TORCH_CHECK(!cache_owner&&pending.empty()&&limit>=1&&limit<=256,"set batch limit before opening an idle runtime; range 1..256");
  request_batch_limit=limit;
}

at::Tensor prepared_info(at::Tensor ticket) {
  auto& r=request(ticket);return ints({r.count,r.misses,r.cpu_hits});
}
void resume_prepare(at::Tensor ticket,at::Tensor rows,at::Tensor slots) {
  request(ticket);
  TORCH_CHECK(!capture_request&&!resumed_prepare,"prepare interception already active");
  resumed_prepare=std::make_tuple(ticket,rows,slots);
}
void ready_for_replay(at::Tensor ticket) {
  auto& r=request(ticket);
  wait(ticket,ticket,r.pools[0]);
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> prepare_replay(
    at::Tensor ids,at::Tensor anchor,int64_t length,int64_t banks,
    const std::optional<at::Tensor>& context,
    const std::optional<at::Tensor>& resident_destination) {
  auto result=prepare_impl(ids,anchor,length,banks,context,resident_destination);
  auto ticket=std::get<0>(result);
  auto info=prepared_info(ticket);
  // Preserve the replay wrapper's existing readiness boundary on every route.
  // Combining control dispatch does not advance consumers past an onboard wait.
  ready_for_replay(ticket);
  return {ticket,std::get<1>(result),std::get<2>(result),info};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> prepare_replay_async(
    at::Tensor ids,at::Tensor anchor,int64_t length,int64_t banks,
    const std::optional<at::Tensor>& context,
    const std::optional<at::Tensor>& resident_destination) {
  auto result=prepare_impl(ids,anchor,length,banks,context,resident_destination);
  auto ticket=std::get<0>(result);
  return {ticket,std::get<1>(result),std::get<2>(result),prepared_info(ticket)};
}
at::Tensor onboard_status(at::Tensor ticket) {
  auto& r=request(ticket);
  std::lock_guard<std::mutex> lock(r.mutex);
  if(r.error)std::rethrow_exception(r.error);
  return ints({r.ready,r.onboard_complete,r.cpu_hits});
}
void capture_barrier(at::Tensor ticket) {
  auto& r=request(ticket);
  TORCH_CHECK(r.ready,"capture barrier requires completed onboarding");
  // Capture is a one-time initialization for each batch shape. Quiesce older
  // transfers so global CUDA capture cannot conflict with the owner's stream.
  if(cache_owner) {
    auto* owner=cache_owner.get();owner->sync([owner]{owner->flush();});
  } else if(call("kvcache_manager_ops::offload_pending",{ticket})[0].toInt()>0) {
    call("kvcache_manager_ops::offload_flush",{ticket});
  }
}
void capture_begin(at::Tensor ticket,at::Tensor slots,at::Tensor missing_rows) {
  auto& r=request(ticket);
  TORCH_CHECK(r.ready&&!capture_request&&!resumed_prepare,"capture requires completed onboarding");
  TORCH_CHECK(missing_rows.is_cuda()&&missing_rows.scalar_type()==at::kLong&&missing_rows.numel()==r.misses,"capture missing-row shape mismatch");
  capture_request=std::make_shared<Request>();
  capture_request->pools=r.pools;capture_request->filled.resize(r.pools.size(),false);
  capture_request->ready=true;capture_request->count=r.count;
  capture_request->misses=r.misses;capture_missing_rows=missing_rows;
  capture_slots=slots;capture_rows=at::full({slots.numel()},-1,at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}
void capture_end(at::Tensor anchor) {
  capture_request.reset();capture_slots=at::Tensor();capture_rows=at::Tensor();capture_missing_rows=at::Tensor();
}
at::Tensor replay_finish(at::Tensor ticket,at::Tensor ordering,at::Tensor enabled) {
  auto& r=request(ticket);
  TORCH_CHECK(r.ready&&!capture_request,"invalid replay completion");
  for(size_t i=0;i<r.filled.size();++i)r.filled[i]=true;
  return offload(ticket,ordering,enabled);
}
}

TORCH_LIBRARY(history_single,m) {
  m.def("prepare(Tensor ids, Tensor anchor, int length, int banks, Tensor? context=None) -> (Tensor, Tensor, Tensor)");
  m.def("pack(Tensor[] inputs, Tensor rows, Tensor slots, Tensor ticket) -> Tensor[]");
  m.def("fill(Tensor(a!) pool, Tensor slots, Tensor key, Tensor value, Tensor ticket, int bank) -> Tensor");
  m.def("onboard_wait(Tensor ticket, Tensor filled, Tensor ordering) -> Tensor");
  m.def("gather(Tensor pool, Tensor slots, Tensor ready, int bank) -> (Tensor, Tensor)");
  m.def("offload_launch(Tensor ticket, Tensor ordering, Tensor enabled) -> Tensor");
  m.def("retire(Tensor ticket) -> Tensor");
  m.def("check_idle(Tensor anchor) -> ()");
  m.def("reset_context(Tensor anchor) -> ()");
  m.def("set_request_limit(int limit) -> ()");
  m.def("owner_start(Tensor anchor) -> Tensor[]");
  m.def("owner_flush(Tensor anchor) -> ()");
  m.def("owner_evict(Tensor ids) -> ()");
  m.def("owner_stats(Tensor anchor) -> Tensor");
  m.def("owner_stop(Tensor anchor) -> ()");
  m.def("prepared_info(Tensor ticket) -> Tensor");
  m.def("prepare_replay(Tensor ids, Tensor anchor, int length, int banks, Tensor? context=None, Tensor(a!)? resident_destination=None) -> (Tensor, Tensor, Tensor(a), Tensor)");
  m.def("prepare_replay_async(Tensor ids, Tensor anchor, int length, int banks, Tensor? context=None, Tensor(a!)? resident_destination=None) -> (Tensor, Tensor, Tensor(a), Tensor)");
  m.def("onboard_status(Tensor ticket) -> Tensor");
  m.def("capture_overlap_mark(bool enabled) -> ()");
  m.def("split_graph(int graph_ptr) -> Tensor");
  m.def("replay_prefix(int handle) -> ()");
  m.def("replay_suffix(int handle, Tensor ticket) -> ()");
  m.def("destroy_split(int handle) -> ()");
  m.def("set_resident_fastpath(bool enabled) -> ()");
  m.def("resident_fastpath_enabled() -> bool");
  m.def("resident_fastpath_stats() -> Tensor");
  m.def("resume_prepare(Tensor ticket, Tensor rows, Tensor slots) -> ()");
  m.def("ready_for_replay(Tensor ticket) -> ()");
  m.def("capture_barrier(Tensor ticket) -> ()");
  m.def("capture_begin(Tensor ticket, Tensor slots, Tensor missing_rows) -> ()");
  m.def("capture_end(Tensor anchor) -> ()");
  m.def("replay_finish(Tensor ticket, Tensor ordering, Tensor enabled) -> Tensor");
}
TORCH_LIBRARY_IMPL(history_single,CompositeExplicitAutograd,m) {
  m.impl("prepare",prepare);m.impl("pack",pack);m.impl("fill",fill);
  m.impl("onboard_wait",wait);m.impl("gather",gather);m.impl("offload_launch",offload);
  m.impl("retire",retire);m.impl("check_idle",reset);
  m.impl("reset_context",reset_context);
  m.impl("set_request_limit",set_request_limit);
  m.impl("owner_start",owner_start);m.impl("owner_flush",owner_flush);
  m.impl("owner_evict",owner_evict);m.impl("owner_stats",owner_stats);m.impl("owner_stop",owner_stop);
  m.impl("prepared_info",prepared_info);m.impl("resume_prepare",resume_prepare);
  m.impl("prepare_replay",prepare_replay);m.impl("set_resident_fastpath",set_resident_fastpath);
  m.impl("prepare_replay_async",prepare_replay_async);m.impl("onboard_status",onboard_status);
  m.impl("capture_overlap_mark",capture_overlap_mark);m.impl("split_graph",split_graph);
  m.impl("replay_prefix",replay_prefix);m.impl("replay_suffix",replay_suffix);m.impl("destroy_split",destroy_split);
  m.impl("resident_fastpath_enabled",resident_fastpath_enabled);m.impl("resident_fastpath_stats",resident_fastpath_stats);
  m.impl("ready_for_replay",ready_for_replay);m.impl("capture_begin",capture_begin);
  m.impl("capture_barrier",capture_barrier);
  m.impl("capture_end",capture_end);m.impl("replay_finish",replay_finish);
}
