// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Included after Request and request registries. The original torch CUDAGraph
// must outlive these executable fragments: it owns their captured tensor pool.
// One serialized request is allowed, exactly as in the surrounding runtime.
void overlap_cuda(cudaError_t status,const char* operation) {
  TORCH_CHECK(status==cudaSuccess,"CUDA graph overlap ",operation,": ",cudaGetErrorString(status));
}
struct OverlapGraph {
  cudaGraph_t prefix=nullptr,suffix=nullptr;
  cudaGraphExec_t prefix_exec=nullptr,suffix_exec=nullptr;
  int device=-1;
  int64_t prefix_nodes=0,suffix_nodes=0,total_nodes=0;
  bool prefix_launched=false;
  cudaStream_t launch_stream=nullptr;
  int64_t active_ticket=0;
  ~OverlapGraph() {
    // Explicit release is only permitted after request retirement. During
    // interpreter failure teardown, CUDA owns completion of any queued work.
    if(prefix_exec)cudaGraphExecDestroy(prefix_exec);
    if(suffix_exec)cudaGraphExecDestroy(suffix_exec);
    if(prefix)cudaGraphDestroy(prefix);
    if(suffix)cudaGraphDestroy(suffix);
  }
};
thread_local bool overlap_mark_capture=false;
thread_local std::unordered_map<cudaGraph_t,cudaGraphNode_t> overlap_markers;
thread_local std::unordered_map<int64_t,std::unique_ptr<OverlapGraph>> overlap_graphs;
thread_local int64_t overlap_serial=1;

void capture_overlap_mark(bool enabled) {
  // Disabling must retain the marker until split_graph() consumes it after
  // capture_end(). Native/eager warmups never insert a marker.
  overlap_mark_capture=enabled;
}
void mark_overlap_boundary() {
  if(!overlap_mark_capture)return;
  auto stream=at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus status;
  cudaGraph_t graph=nullptr;
  const cudaGraphNode_t* dependencies=nullptr;
  size_t count=0;
#if CUDART_VERSION >= 13000
  overlap_cuda(cudaStreamGetCaptureInfo(stream,&status,nullptr,&graph,&dependencies,nullptr,&count),"capture info");
#else
  overlap_cuda(cudaStreamGetCaptureInfo_v2(stream,&status,nullptr,&graph,&dependencies,&count),"capture info");
#endif
  if(status!=cudaStreamCaptureStatusActive)return;
  TORCH_CHECK(graph&&count,"overlap boundary requires an active, nonempty GPU prefix");
  TORCH_CHECK(!overlap_markers.count(graph),"overlap graph has more than one first-read marker");
  cudaGraphNode_t marker=nullptr;
  overlap_cuda(cudaGraphAddEmptyNode(&marker,graph,dependencies,count),"add first-read marker");
#if CUDART_VERSION >= 13000
  overlap_cuda(cudaStreamUpdateCaptureDependencies(stream,&marker,nullptr,1,cudaStreamSetCaptureDependencies),"set first-read dependency");
#else
  overlap_cuda(cudaStreamUpdateCaptureDependencies(stream,&marker,1,cudaStreamSetCaptureDependencies),"set first-read dependency");
#endif
  overlap_markers[graph]=marker;
}
std::vector<cudaGraphNode_t> overlap_dependencies(cudaGraphNode_t node) {
  size_t count=0;
#if CUDART_VERSION >= 13000
  overlap_cuda(cudaGraphNodeGetDependencies(node,nullptr,nullptr,&count),"dependency count");
#else
  overlap_cuda(cudaGraphNodeGetDependencies(node,nullptr,&count),"dependency count");
#endif
  std::vector<cudaGraphNode_t> dependencies(count);
  if(count) {
#if CUDART_VERSION >= 13000
    overlap_cuda(cudaGraphNodeGetDependencies(node,dependencies.data(),nullptr,&count),"dependencies");
#else
    overlap_cuda(cudaGraphNodeGetDependencies(node,dependencies.data(),&count),"dependencies");
#endif
  }
  dependencies.resize(count);return dependencies;
}
std::vector<cudaGraphNode_t> overlap_nodes(cudaGraph_t graph) {
  size_t count=0;
  overlap_cuda(cudaGraphGetNodes(graph,nullptr,&count),"node count");
  std::vector<cudaGraphNode_t> nodes(count);
  if(count)overlap_cuda(cudaGraphGetNodes(graph,nodes.data(),&count),"nodes");
  nodes.resize(count);return nodes;
}
void check_overlap_node_types(cudaGraph_t graph) {
  for(auto node:overlap_nodes(graph)) {
    cudaGraphNodeType type;
    overlap_cuda(cudaGraphNodeGetType(node,&type),"node type");
    switch(type) {
      case cudaGraphNodeTypeKernel: case cudaGraphNodeTypeMemcpy:
      case cudaGraphNodeTypeMemset: case cudaGraphNodeTypeEmpty:
      case cudaGraphNodeTypeEventRecord: case cudaGraphNodeTypeWaitEvent:
        break;
      case cudaGraphNodeTypeGraph: {
        cudaGraph_t child=nullptr;
        overlap_cuda(cudaGraphChildGraphNodeGetGraph(node,&child),"child graph");
        check_overlap_node_types(child);break;
      }
      default:
        TORCH_CHECK(false,"overlap cannot split graph node type ",(int)type,
                    "; allocation/free/host/conditional nodes require an explicit lifetime design");
    }
  }
}
at::Tensor split_graph(int64_t graph_ptr) {
  TORCH_CHECK(graph_ptr>0&&!capture_request,"split requires a completed captured graph");
  auto original=reinterpret_cast<cudaGraph_t>(static_cast<uintptr_t>(graph_ptr));
  auto found=overlap_markers.find(original);
  TORCH_CHECK(found!=overlap_markers.end(),"captured graph has no explicit first-history-read marker");
  auto marker=found->second;
  auto nodes=overlap_nodes(original);
  std::unordered_set<cudaGraphNode_t> all(nodes.begin(),nodes.end());
  TORCH_CHECK(all.count(marker),"first-history-read marker is not in captured graph");
  check_overlap_node_types(original);
  std::unordered_set<cudaGraphNode_t> before;
  auto visit=overlap_dependencies(marker);
  while(!visit.empty()) {
    auto n=visit.back();visit.pop_back();
    TORCH_CHECK(all.count(n)&&n!=marker,"invalid/cyclic first-read dependency");
    if(!before.insert(n).second)continue;
    auto deps=overlap_dependencies(n);visit.insert(visit.end(),deps.begin(),deps.end());
  }
  TORCH_CHECK(!before.empty()&&before.size()+1<nodes.size(),"overlap split requires nonempty prefix and suffix");
  auto split=std::make_unique<OverlapGraph>();
  overlap_cuda(cudaGetDevice(&split->device),"device");
  split->prefix_nodes=before.size();split->suffix_nodes=nodes.size()-before.size()-1;split->total_nodes=nodes.size();
  overlap_cuda(cudaGraphClone(&split->prefix,original),"clone prefix");
  overlap_cuda(cudaGraphClone(&split->suffix,original),"clone suffix");
  for(auto n:nodes) {
    cudaGraphNode_t cloned=nullptr;
    if(!before.count(n)) {
      overlap_cuda(cudaGraphNodeFindInClone(&cloned,n,split->prefix),"find prefix node");
      overlap_cuda(cudaGraphDestroyNode(cloned),"prune prefix");
    }
    if(before.count(n)||n==marker) {
      overlap_cuda(cudaGraphNodeFindInClone(&cloned,n,split->suffix),"find suffix node");
      overlap_cuda(cudaGraphDestroyNode(cloned),"prune suffix");
    }
  }
  TORCH_CHECK(overlap_nodes(split->prefix).size()==(size_t)split->prefix_nodes&&
              overlap_nodes(split->suffix).size()==(size_t)split->suffix_nodes,"graph split lost nodes");
  // Removing cross-cut edges is safe: a prefix contains every ancestor of the
  // marker, and both executable fragments are launched in order on one stream.
  // No suffix consumer is queued until actual owner readiness has completed.
  overlap_cuda(cudaGraphInstantiateWithFlags(&split->prefix_exec,split->prefix,0),"instantiate prefix");
  overlap_cuda(cudaGraphInstantiateWithFlags(&split->suffix_exec,split->suffix,0),"instantiate suffix");
  auto handle=overlap_serial++;
  auto result=ints({handle,split->prefix_nodes,split->suffix_nodes,split->total_nodes});
  overlap_graphs[handle]=std::move(split);overlap_markers.erase(found);
  return result;
}
OverlapGraph& overlap_graph(int64_t handle) {
  auto found=overlap_graphs.find(handle);
  TORCH_CHECK(found!=overlap_graphs.end(),"unknown/retired split graph handle");
  int device;overlap_cuda(cudaGetDevice(&device),"current device");
  TORCH_CHECK(device==found->second->device,"split graph launched on a different GPU");
  return *found->second;
}
void replay_prefix(int64_t handle) {
  auto& split=overlap_graph(handle);
  TORCH_CHECK(pending.size()==1&&!capture_request&&!split.prefix_launched,"prefix requires one active serialized request");
  split.launch_stream=at::cuda::getCurrentCUDAStream();
  split.active_ticket=pending.begin()->first;
  overlap_cuda(cudaGraphLaunch(split.prefix_exec,split.launch_stream),"launch prefix");
  split.prefix_launched=true;
}
void replay_suffix(int64_t handle,at::Tensor ticket) {
  auto& split=overlap_graph(handle);auto& r=request(ticket);
  TORCH_CHECK(split.prefix_launched&&ticket.data_ptr<int64_t>()[0]==split.active_ticket,
              "suffix must follow this request's prefix");
  TORCH_CHECK(r.ready,"suffix requires ready=true after completed CPU onboarding");
  TORCH_CHECK(split.launch_stream==at::cuda::getCurrentCUDAStream().stream(),"prefix/suffix must use the same CUDA stream");
  overlap_cuda(cudaGraphLaunch(split.suffix_exec,split.launch_stream),"launch suffix");
  split.prefix_launched=false;split.active_ticket=0;
}
void destroy_split(int64_t handle) {
  auto found=overlap_graphs.find(handle);
  if(found==overlap_graphs.end())return; // owner_stop() may already have cleared it.
  TORCH_CHECK(pending.empty()&&!found->second->prefix_launched,"retire request before destroying split graphs");
  overlap_graphs.erase(found);
}
void clear_overlap_graphs() {
  TORCH_CHECK(pending.empty(),"retire all requests before closing split graphs");
  overlap_graphs.clear();overlap_markers.clear();overlap_mark_capture=false;
}
