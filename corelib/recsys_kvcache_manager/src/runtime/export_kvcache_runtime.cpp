#include "export_kvcache_runtime.h"
#include "kvcache_runtime_config.h"
#include "cache_owner.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <string>
#include <thread>

using namespace kvcache;

namespace kvcache_manager {

namespace {

#ifndef KVCACHE_MANAGER_RUNTIME_DEBUG
#define KVCACHE_MANAGER_RUNTIME_DEBUG 0
#endif

void log_runtime_message(const std::string& message) {
#if KVCACHE_MANAGER_RUNTIME_DEBUG
    std::cout << "[KVCACHE][runtime] " << message << std::endl;
#else
    (void)message;
#endif
}

} // namespace

ExportKVCacheRuntime::ExportKVCacheRuntime() {
    log_runtime_message(
        "thread=" + c10::str(std::this_thread::get_id()) + " constructor enter"
    );
    const auto config_path = find_config_path();
    const auto config = load_yaml_config(config_path);
    const int num_layers = get_config_int(config, "num_layers");
    const int num_kv_heads = get_config_int(config, "num_kv_heads");
    const int head_size = get_config_int(config, "head_size");
    const int tokens_per_page = get_config_int(config, "tokens_per_page");
    const int tokens_per_chunk = get_config_int(config, "tokens_per_chunk");
    const int num_primary_cache_pages = get_config_int(config, "num_primary_cache_pages");
    const int num_buffer_pages = get_config_int(config, "num_buffer_pages");
    const int max_batch_size = get_config_int(config, "max_batch_size");
    const int max_sequence_length = get_config_int(config, "max_sequence_length");
    const int device_idx = get_config_int(config, "device_idx");
    fixed_history_slots_ = config.count("fixed_history_slots") &&
        get_config_int(config, "fixed_history_slots") != 0;
    batch_lookup_ = config.count("batch_lookup") && get_config_int(config,"batch_lookup") != 0;
    onboard_child_tasks_ = config.count("onboard_child_tasks") &&
        get_config_int(config, "onboard_child_tasks") != 0;
    fast_metadata_ = fixed_history_slots_ && config.count("fast_metadata") &&
        get_config_int(config,"fast_metadata") != 0;
    TORCH_CHECK(!fixed_history_slots_ || (tokens_per_page == max_sequence_length &&
                tokens_per_chunk == tokens_per_page),
                "Fixed history slots require one complete snapshot per page/chunk");
    const at::ScalarType cache_dtype = get_config_dtype(config);
    num_layers_ = num_layers;

    log_runtime_message(
        "config_path=" + config_path.string()
        + " layers=" + std::to_string(num_layers)
        + " kv_heads=" + std::to_string(num_kv_heads)
        + " head_size=" + std::to_string(head_size)
        + " page=" + std::to_string(tokens_per_page)
        + " chunk=" + std::to_string(tokens_per_chunk)
        + " primary_pages=" + std::to_string(num_primary_cache_pages)
        + " buffer_pages=" + std::to_string(num_buffer_pages)
        + " max_bs=" + std::to_string(max_batch_size)
        + " max_seq=" + std::to_string(max_sequence_length)
        + " device=" + std::to_string(device_idx)
        + " dtype=" + c10::str(cache_dtype)
    );

    log_runtime_message("creating GPUKVCacheManagerImpl");
    gpu_kvcache_ = std::make_unique<GPUKVCacheManagerImpl>(
        /*num_layers=*/num_layers,
        /*num_kv_heads=*/num_kv_heads,
        /*kv_headdim=*/head_size,
        /*num_tokens_per_page=*/tokens_per_page,
        /*num_tokens_per_chunk=*/tokens_per_chunk,
        /*num_primary_cache_pages=*/num_primary_cache_pages,
        /*num_buffer_pages=*/num_buffer_pages,
        /*max_batch_size=*/max_batch_size,
        /*max_sequence_length=*/max_sequence_length,
        /*device_idx=*/device_idx);

    auto server_recv_port = get_required_config_string(config, "server_recv_port");
    auto gpu_register_port = get_required_config_string(config, "gpu_register_port");
    log_runtime_message(
        "creating FlexKVCppClient server_recv_port=" + server_recv_port
    );
    flexkv_client_ = std::make_unique<FlexKVCppClient>(server_recv_port, /*dpClientId=*/0, /*tpSize=*/1);

    log_runtime_message("start_server_and_register begin");
    flexkv_client_->start_server_and_register();
    log_runtime_message("start_server_and_register done");

    log_runtime_message(
        "creating GPU registrator port=" + gpu_register_port
    );
    flexkv_gpu_registrator_ = std::make_unique<FlexKVGPURegistrator>(gpu_register_port, /*dpClientId=*/0, /*device_id=*/device_idx);
    FlexKVCacheLayoutSpec layout{
        FlexKVLayoutType::LayerFirst,
        num_layers, num_primary_cache_pages, tokens_per_page, num_kv_heads, head_size,
        false,
    };
    cache_tensor_ = at::empty(
        {num_layers, num_primary_cache_pages, 2, tokens_per_page, num_kv_heads, head_size},
        at::TensorOptions().dtype(cache_dtype).device(at::Device(at::kCUDA, device_idx)));
    cache_tables_ = cache_tensor_.unbind(0);
    log_runtime_message(
        "register_to_server begin cache_tensor=" + c10::str(cache_tensor_.sizes())
    );
    flexkv_gpu_registrator_->register_to_server({cache_tensor_}, layout);
    log_runtime_message("register_to_server done");

    auto start_time = std::chrono::steady_clock::now();
    while (!flexkv_client_->is_ready()) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > std::chrono::seconds(30)) {
            throw std::runtime_error("FlexKV client failed to become ready within 30 seconds");
        }
        log_runtime_message(
            "waiting for FlexKV client ready elapsed_ms="
            + std::to_string(
                std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
            )
        );
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    log_runtime_message(
        "FlexKV client ready=" + std::to_string(flexkv_client_->is_ready())
    );

    lookup_token_index_ = at::arange(0, max_sequence_length, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
    token_slot_mappings_in_page_ = at::arange(0, gpu_kvcache_->num_tokens_per_page, at::TensorOptions().dtype(torch::kInt64).device(at::kCPU));
    log_runtime_message("constructor exit");
}

ExportKVCacheRuntime::~ExportKVCacheRuntime() {
    log_runtime_message("destructor enter");
    if (flexkv_client_) {
        flexkv_client_->shutdown();
    }
    log_runtime_message("destructor exit");
}

std::vector<at::Tensor>
ExportKVCacheRuntime::lookup_kvcache(at::Tensor user_ids, at::Tensor seqlens) {
    auto gpu_lookup_result = gpu_kvcache_->lookup(user_ids);
    auto gpu_cached_startpos = gpu_lookup_result[0];
    auto gpu_cached_lengths = gpu_lookup_result[1];

    const auto batch_size = user_ids.size(0);
    auto task_ids =
        at::empty({batch_size}, at::dtype(torch::kInt64).device(at::kCPU));
    auto host_cached_startpos = at::zeros_like(
        gpu_cached_startpos, at::dtype(torch::kInt32).device(at::kCPU));
    auto host_cached_lengths = at::empty_like(
        gpu_cached_lengths, at::dtype(torch::kInt32).device(at::kCPU));

    const auto *gpu_cached_startpos_ptr =
        gpu_cached_startpos.data_ptr<int32_t>();
    const auto *gpu_cached_lengths_ptr = gpu_cached_lengths.data_ptr<int32_t>();
    const auto *user_ids_ptr = user_ids.data_ptr<int64_t>();
    const auto *seqlens_ptr = seqlens.data_ptr<int64_t>();
    auto *task_ids_ptr = task_ids.data_ptr<int64_t>();
    auto *host_cached_lengths_ptr = host_cached_lengths.data_ptr<int32_t>();

    std::vector<int> remote_rows;
    std::vector<at::Tensor> remote_tokens;
    std::vector<std::vector<std::string>> remote_namespaces;
    auto matched_length = [this](const at::Tensor& mask) {
        if (!fast_metadata_) return mask.sum().item<int>();
        TORCH_CHECK(mask.device().is_cpu() && mask.scalar_type() == at::kBool && mask.is_contiguous(),
                    "FlexKV match mask must be a contiguous CPU boolean tensor");
        const auto* values = mask.data_ptr<bool>();
        return static_cast<int>(std::count(values, values + mask.numel(), true));
    };
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        // A cache entry rooted at token zero is already directly usable. Avoid
        // the synchronous FlexKV/CPU lookup in this common GPU-cache-hit path.
        // A zero host result and -1 task ID also make onboarding a no-op for
        // this sequence.
        if (gpu_cached_startpos_ptr[seq_idx] == 0 &&
            gpu_cached_lengths_ptr[seq_idx] > 0) {
            task_ids_ptr[seq_idx] = -1;
            host_cached_lengths_ptr[seq_idx] = 0;
            continue;
        }

        int64_t uid = user_ids_ptr[seq_idx];
        int64_t seqlen = seqlens_ptr[seq_idx];

        auto user_namespace = "uid:" + std::to_string(uid);
        if(batch_lookup_) {
            remote_rows.push_back(seq_idx);
            remote_tokens.push_back(lookup_token_index_.slice(0,0,seqlen));
            remote_namespaces.push_back({user_namespace});
            continue;
        }
        auto [task_id, cache_result] = flexkv_client_->get_match(
            lookup_token_index_.slice(0, 0, seqlen), std::nullopt, num_layers_,
            {user_namespace});
        task_ids_ptr[seq_idx] = task_id;

        int cached_length = matched_length(cache_result);
        host_cached_lengths_ptr[seq_idx] = cached_length;
    }
    if(!remote_rows.empty()) {
        auto matches=flexkv_client_->get_match_batch(remote_tokens,num_layers_,remote_namespaces);
        for(size_t i=0;i<remote_rows.size();++i) {
            task_ids_ptr[remote_rows[i]]=matches[i].first;
            host_cached_lengths_ptr[remote_rows[i]]=matched_length(matches[i].second);
        }
    }

    {
        // get_match creates a server task even for an empty match. Neither
        // paged nor fixed-slot onboarding launches it; cancel empty matches
        // so continuous misses do not retain server-side task records.
        std::vector<int64_t> empty_matches;
        for (int64_t row = 0; row < batch_size; ++row) {
            if (host_cached_lengths_ptr[row] == 0 && task_ids_ptr[row] >= 0)
                empty_matches.push_back(task_ids_ptr[row]);
        }
        if (!empty_matches.empty()) flexkv_client_->cancel(empty_matches);
        // Preserve lookup's IDs for diagnostics. Onboarding masks rows with
        // zero matched length before using task IDs; owner misses launch none.
    }

    if (fast_metadata_) {
        // Preserve the generic merge formula while avoiding many tiny CPU
        // Tensor operations for the fixed-slot runtime.
        auto merged_start = at::empty_like(gpu_cached_startpos);
        auto merged_length = at::empty_like(gpu_cached_lengths);
        auto* starts = merged_start.data_ptr<int32_t>();
        auto* lengths = merged_length.data_ptr<int32_t>();
        for (int64_t i = 0; i < batch_size; ++i) {
            const int gpu_start = gpu_cached_startpos_ptr[i];
            const int gpu_length = gpu_cached_lengths_ptr[i];
            const int host_length = host_cached_lengths_ptr[i];
            int start = host_length > 0 ? 0 : gpu_start;
            int length = gpu_length > 0 ? std::max(gpu_start, 0) + gpu_length - start : host_length;
            if ((gpu_start > host_length && gpu_length > 0) || start > 0) start = length = 0;
            starts[i] = start;
            lengths[i] = length;
        }
        return {merged_start, merged_length, gpu_cached_startpos, gpu_cached_lengths,
                host_cached_startpos, host_cached_lengths, task_ids};
    }

    auto merge_cached_startpos = at::where(
        host_cached_lengths > 0, host_cached_startpos, gpu_cached_startpos);
    auto merge_cached_lengths =
        at::where(gpu_cached_lengths > 0,
                  at::max(gpu_cached_startpos, host_cached_startpos) +
                      gpu_cached_lengths - merge_cached_startpos,
                  host_cached_lengths);
    auto non_valid_caching =
        (gpu_cached_startpos > host_cached_lengths) & (gpu_cached_lengths > 0);
    non_valid_caching = non_valid_caching | (merge_cached_startpos > 0);
    merge_cached_lengths =
        at::where(non_valid_caching, 0, merge_cached_lengths);
    merge_cached_startpos =
        at::where(non_valid_caching, 0, merge_cached_startpos);

    return {
        merge_cached_startpos,
        merge_cached_lengths,
        gpu_cached_startpos,
        gpu_cached_lengths,
        host_cached_startpos,
        host_cached_lengths,
        task_ids,
    };
}

std::vector<at::Tensor>
ExportKVCacheRuntime::allocate_kvcache(at::Tensor user_ids, at::Tensor seqlens,
                                       at::Tensor merged_cached_lengths,
                                       at::Tensor host_cached_lengths) {
    const auto batch_size = user_ids.size(0);
    if (fixed_history_slots_) {
        const int length = gpu_kvcache_->num_tokens_per_page;
        TORCH_CHECK(batch_size > 0 && batch_size <= gpu_kvcache_->num_primary_cache_pages &&
                    batch_size <= gpu_kvcache_->max_batch_size, "Invalid fixed-slot batch");
        const auto* ids = user_ids.data_ptr<int64_t>();
        const auto* lens = seqlens.data_ptr<int64_t>();
        const auto* cached = merged_cached_lengths.data_ptr<int>();
        const auto* host = host_cached_lengths.data_ptr<int>();
        const std::unordered_set<int64_t> protected_ids(ids, ids + batch_size);
        TORCH_CHECK(protected_ids.size() == batch_size, "Deduplicate fixed-slot identities first");
        auto cpu = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
        const int64_t slot_stride = (batch_size + 3) / 4 * 4;
        at::Tensor staging;
        if (fast_metadata_) staging = at::zeros({slot_stride + 5 * batch_size + 4}, cpu.pinned_memory(true));
        auto slots = fast_metadata_ ? staging.narrow(0, 0, batch_size) : at::empty({batch_size}, cpu);
        auto metadata = fast_metadata_ ? staging.narrow(0, slot_stride, 5 * batch_size + 4)
                                      : at::zeros({5 * batch_size + 4}, cpu);
        auto* m = metadata.data_ptr<int>();
        int new_tokens = 0;
        for (int i = 0; i < batch_size; ++i) {
            TORCH_CHECK(lens[i] == length && (cached[i] == 0 || cached[i] == length) &&
                        (host[i] == 0 || host[i] == length), "Partial fixed-history snapshot");
            const bool resident=gpu_kvcache_->_uid_to_page_id.count(ids[i]);
            const int old_offloaded=resident && gpu_kvcache_->_uid_to_offloaded_length[ids[i]]>=length;
            // A GPU hit skips the host lookup: host[i]==0 means unknown,
            // not "the prior CPU publication disappeared". Keep its eviction
            // bookkeeping. A later CPU lookup still verifies actual residency.
            if(!resident || host[i]>0)gpu_kvcache_->_uid_to_offloaded_length[ids[i]]=host[i];
            const auto& pages = gpu_kvcache_->alloc_single_sequence(ids[i], length, host[i], protected_ids);
            TORCH_CHECK(pages.size() == 1, "Expected one GPU slot per history");
            const int new_offloaded=gpu_kvcache_->_uid_to_offloaded_length[ids[i]]>=length;
            gpu_kvcache_->total_offloaded_pages+=new_offloaded-old_offloaded;
            slots.data_ptr<int>()[i] = pages[0];
            m[i + 1] = i + 1;
            m[batch_size + 1 + i] = length;
            m[2 * batch_size + 1 + i] = length;
            m[3 * batch_size + 2 + i] = (i + 1) * length;
            new_tokens += cached[i] == 0 ? length : 0;
            m[4 * batch_size + 4 + i] = new_tokens;
        }
        m[4 * batch_size + 2] = new_tokens;
        at::Tensor gpu_meta, gpu_slots;
        if (fast_metadata_) {
            // PyTorch's pinned allocator retains staging storage until this
            // asynchronous copy completes. The existing metadata CUDA event
            // orders the consuming stream after the owner stream's upload.
            auto uploaded = staging.to(gpu_kvcache_->device, /*non_blocking=*/true);
            gpu_slots = uploaded.narrow(0, 0, batch_size);
            gpu_meta = uploaded.narrow(0, slot_stride, 5 * batch_size + 4);
        } else {
            gpu_meta = metadata.to(gpu_kvcache_->device);
            gpu_slots = slots.to(gpu_kvcache_->device);
        }
        auto empty = at::empty({0}, gpu_meta.options());
        return {gpu_slots, gpu_meta,
            gpu_meta.narrow(0, 0, batch_size + 1),
            gpu_meta.narrow(0, batch_size + 1, batch_size),
            gpu_meta.narrow(0, 2 * batch_size + 1, batch_size),
            gpu_meta.narrow(0, 3 * batch_size + 1, batch_size + 1),
            gpu_meta.narrow(0, 4 * batch_size + 3, batch_size + 1), empty, empty,
            at::tensor({new_tokens}, cpu), gpu_meta.narrow(0, 4 * batch_size + 2, 1),
            at::full({batch_size}, length, cpu), at::arange(batch_size + 1, cpu) * length};
    }
    int num_total_pages;
    int num_new_tokens;
    {
        nvtx3::scoped_range range{
            "hstu.aoti.kvcache.allocate.shape_calculation"};
        num_total_pages =
            at::ceil(seqlens / this->gpu_kvcache_->num_tokens_per_page)
                .to(torch::kInt32)
                .sum()
                .item<int>();
        num_new_tokens = (seqlens - merged_cached_lengths).sum().item<int>();
    }

    at::Tensor page_ids_gpu_buffer;
    at::Tensor metadata_gpu_buffer;
    {
        nvtx3::scoped_range range{"hstu.aoti.kvcache.allocate.cuda_buffers"};
        page_ids_gpu_buffer = at::empty(
            {num_total_pages}, at::dtype(torch::kInt32).device(at::kCUDA));
        metadata_gpu_buffer =
            at::empty({5 * batch_size + 4 + num_new_tokens * 2},
                      at::dtype(torch::kInt32).device(at::kCUDA));
    }

    {
        nvtx3::scoped_range range{"hstu.aoti.kvcache.allocate.gpu_manager"};
        gpu_kvcache_->allocate(user_ids, seqlens, host_cached_lengths,
                               page_ids_gpu_buffer, metadata_gpu_buffer);
    }

    auto new_history_nnz =
        at::tensor({num_new_tokens}, at::dtype(torch::kInt32).device(at::kCPU));

    return {
        page_ids_gpu_buffer,
        metadata_gpu_buffer,
        metadata_gpu_buffer.narrow(0, 0, batch_size + 1), // page_indptr
        metadata_gpu_buffer.narrow(0, batch_size + 1,
                                   batch_size), // last_page_lens
        metadata_gpu_buffer.narrow(0, batch_size * 2 + 1,
                                   batch_size), // total_history_lengths
        metadata_gpu_buffer.narrow(0, batch_size * 3 + 1,
                                   batch_size + 1), // total_history_offsets
        metadata_gpu_buffer.narrow(0, batch_size * 4 + 3,
                                   batch_size + 1), // new_history_offsets
        metadata_gpu_buffer.narrow(0, batch_size * 5 + 4,
                                   num_new_tokens), // batch_indices
        metadata_gpu_buffer.narrow(0, batch_size * 5 + 4 + num_new_tokens,
                                   num_new_tokens), // positions

        new_history_nnz,
        metadata_gpu_buffer.narrow(0, batch_size * 4 + 2,
                                   1), // new_history_nnz_cuda

        at::empty({batch_size},
                  at::dtype(torch::kInt32).device(at::kCPU)), // kv_seqlens
        at::empty(
            {batch_size + 1},
            at::dtype(torch::kInt32).device(at::kCPU)), // kv_seqlen_offsets
    };
}

// The fixed-slot allocator is CPU-authoritative. Transfers run on the same
// owner as allocation; offloads pin their slots before a later allocation can
// recycle them. Reconstruct slot mappings directly, without GPU->CPU metadata
// copies or one synchronizing GPU scalar read per sequence boundary.
std::vector<at::Tensor> ExportKVCacheRuntime::fixed_slot_mappings(
    at::Tensor user_ids, at::Tensor seqlens) {
    TORCH_CHECK(fixed_history_slots_, "fixed slot mapping requires fixed history storage");
    const auto length = gpu_kvcache_->num_tokens_per_page;
    const auto* ids = user_ids.data_ptr<int64_t>();
    const auto* lengths = seqlens.data_ptr<int64_t>();
    std::vector<at::Tensor> mappings;
    mappings.reserve(user_ids.numel());
    for (int64_t row = 0; row < user_ids.numel(); ++row) {
        const auto found = gpu_kvcache_->_uid_to_page_id.find(ids[row]);
        TORCH_CHECK(lengths[row] == length && found != gpu_kvcache_->_uid_to_page_id.end()
                    && found->second.size() == 1, "Missing or non-fixed transfer allocation");
        auto mapping = at::empty({length}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
        auto* values = mapping.data_ptr<int64_t>();
        const int64_t first = static_cast<int64_t>(found->second[0]) * length;
        for (int64_t position = 0; position < length; ++position) values[position] = first + position;
        mappings.push_back(std::move(mapping));
    }
    return mappings;
}

std::vector<at::Tensor> ExportKVCacheRuntime::onboard_kvcache_launch(
    at::Tensor user_ids, at::Tensor seqlens, at::Tensor merged_cached_lengths,
    at::Tensor host_cached_lengths, at::Tensor gpu_cached_startpos,
    at::Tensor gpu_cached_lengths,
    at::Tensor &task_ids, // from `get_match` in lookup and reuse in onboarding
                          // only, returned as masked
    at::Tensor kv_page_indices, at::Tensor kv_page_indptr) {
    const auto batch_size = user_ids.size(0);

    bool full_gpu_hit_no_new_history = batch_size > 0;
    const auto *seqlens_ptr = seqlens.data_ptr<int64_t>();
    const auto *merged_cached_lengths_ptr =
        merged_cached_lengths.data_ptr<int32_t>();
    const auto *host_cached_lengths_ptr =
        host_cached_lengths.data_ptr<int32_t>();
    const auto *gpu_cached_startpos_ptr =
        gpu_cached_startpos.data_ptr<int32_t>();
    const auto *gpu_cached_lengths_ptr = gpu_cached_lengths.data_ptr<int32_t>();
    auto *task_ids_ptr = task_ids.data_ptr<int64_t>();
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
        full_gpu_hit_no_new_history =
            full_gpu_hit_no_new_history &&
            host_cached_lengths_ptr[seq_idx] == 0 &&
            gpu_cached_startpos_ptr[seq_idx] == 0 &&
            gpu_cached_lengths_ptr[seq_idx] == seqlens_ptr[seq_idx] &&
            merged_cached_lengths_ptr[seq_idx] == seqlens_ptr[seq_idx];
    }
    if (full_gpu_hit_no_new_history) {
        nvtx3::scoped_range range{"hstu.aoti.kvcache.onboard_launch."
                                  "full_gpu_hit_no_new_history"};
        std::vector<at::Tensor> empty_slot_mappings;
        empty_slot_mappings.reserve(batch_size);
        for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
            task_ids_ptr[seq_idx] = -1;
            empty_slot_mappings.push_back(
                at::empty({0}, at::dtype(torch::kInt64).device(at::kCPU)));
        }
        return empty_slot_mappings;
    }

    std::vector<at::Tensor> slot_mappings;
    if (fixed_history_slots_ && fast_metadata_) {
        slot_mappings = fixed_slot_mappings(user_ids, seqlens);
    } else {
        at::Tensor kv_page_indices_cpu = kv_page_indices.cpu();
        slot_mappings.reserve(batch_size);
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            const auto seqlen = seqlens_ptr[seq_idx];
            auto page_ids =
                kv_page_indices_cpu.slice(0, kv_page_indptr[seq_idx].item<int>(),
                                          kv_page_indptr[seq_idx + 1].item<int>());

            if (page_ids.numel() > 0) {
                auto slot_mapping = page_ids.unsqueeze(1) *
                                        this->gpu_kvcache_->num_tokens_per_page +
                                    token_slot_mappings_in_page_.unsqueeze(0);
                slot_mappings.push_back(
                    slot_mapping.reshape(-1).slice(0, 0, seqlen));
            } else {
                slot_mappings.push_back(
                    at::empty({0}, at::dtype(torch::kInt64).device(at::kCPU)));
            }
        }
    }

    std::vector<int64_t> task_ids_tolaunch;
    std::vector<at::Tensor> slot_mappings_tolaunch;
    task_ids_tolaunch.reserve(batch_size);
    slot_mappings_tolaunch.reserve(batch_size);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        if (merged_cached_lengths_ptr[seq_idx] == 0 ||
            host_cached_lengths_ptr[seq_idx] == 0) {
            // -1 indicates no cache to onboard for this sequence.
            task_ids_ptr[seq_idx] = -1;
            continue;
        }

        const bool should_launch = host_cached_lengths_ptr[seq_idx] >
                                       gpu_cached_startpos_ptr[seq_idx] +
                                           gpu_cached_lengths_ptr[seq_idx] ||
                                   gpu_cached_startpos_ptr[seq_idx] > 0;

        if (should_launch) {
            task_ids_tolaunch.push_back(task_ids_ptr[seq_idx]);
            slot_mappings_tolaunch.push_back(slot_mappings[seq_idx]);
        } else {
            task_ids_ptr[seq_idx] = -1;
        }
    }

    if (!task_ids_tolaunch.empty()) {
        const auto launched_ids = flexkv_client_->launch_tasks(
            task_ids_tolaunch, slot_mappings_tolaunch, true);
        // Newer servers retire merged children and return parent handles.
        // Some pinned servers retain children; observing their parent alone
        // would leave dangling child records. Keep original child IDs only
        // when that backend contract is explicitly selected in the config.
        if (!onboard_child_tasks_) {
            std::fill(task_ids_ptr, task_ids_ptr + batch_size, -1);
            std::copy(launched_ids.begin(), launched_ids.end(), task_ids_ptr);
        }
    }

    return slot_mappings;
}

void ExportKVCacheRuntime::onboard_kvcache_wait(
    at::Tensor task_ids // masked task ids from onboarding launch
) {
    const auto batch_size = task_ids.size(0);
    const auto *task_ids_ptr = task_ids.data_ptr<int64_t>();

    std::vector<int64_t> task_ids_launched;
    task_ids_launched.reserve(batch_size);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        if (task_ids_ptr[seq_idx] == -1) {
            continue;
        }
        task_ids_launched.push_back(task_ids_ptr[seq_idx]);
    }
    if (task_ids_launched.empty()) {
        return; // no onboard task launched, skip waiting
    }

    auto wait_results = flexkv_client_->wait(task_ids_launched, 1000.f, true);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        if (task_ids_ptr[seq_idx] == -1) {
            continue;
        }
        const int64_t task_id = task_ids_ptr[seq_idx];
        auto it = wait_results.find(task_id);
        if (it == wait_results.end()) {
            throw std::runtime_error(
                "Did not receive result for onboard task_id " +
                std::to_string(task_id));
        }
        auto &result = it->second;
        if (result.status != KVResponseStatus::SUCCESS) {
            throw std::runtime_error(
                "Onboard task_id " + std::to_string(task_id) +
                " failed with status: " +
                std::to_string(static_cast<int>(result.status)));
        }
    }

    // skipped returning status for now
}

// This runs only on the manager owner after the preceding inference retired.
// A resident mapping is therefore complete, and no allocation/eviction can run
// until this request completes. Background D2H reads the same immutable slots.
// No lookup tensors, allocation metadata, CUDA work or FlexKV RPC are needed.
bool ExportKVCacheRuntime::try_resident_slots(
    at::Tensor user_ids, int64_t length, int64_t banks, at::Tensor cpu_slots) {
    TORCH_CHECK(user_ids.device().is_cpu() && user_ids.scalar_type() == at::kLong &&
                    user_ids.dim() == 1 && user_ids.is_contiguous(),
                "resident slots require contiguous CPU int64 IDs");
    TORCH_CHECK(cpu_slots.device().is_cpu() && cpu_slots.scalar_type() == at::kInt &&
                    cpu_slots.dim() == 1 && cpu_slots.is_contiguous() &&
                    cpu_slots.numel() == user_ids.numel() && cpu_slots.is_pinned(),
                "resident slots require a matching pinned CPU int32 output buffer");
    const auto batch = user_ids.numel();
    if (!fixed_history_slots_ || batch < 1 || batch > gpu_kvcache_->max_batch_size ||
            length != gpu_kvcache_->num_tokens_per_page || banks != num_layers_)
        return false;
    const auto* ids = user_ids.data_ptr<int64_t>();
    auto* slots = cpu_slots.data_ptr<int32_t>();

    // Validate the entire batch before touching either output or LRU state.
    // Partial residency falls back to normal lookup/allocation/onboarding.
    for (int64_t i = 0; i < batch; ++i) {
        const auto pages = gpu_kvcache_->_uid_to_page_id.find(ids[i]);
        const auto start = gpu_kvcache_->_uid_to_paged_cache_startpos.find(ids[i]);
        const auto count = gpu_kvcache_->_uid_to_paged_cache_length.find(ids[i]);
        if (pages == gpu_kvcache_->_uid_to_page_id.end() || pages->second.size() != 1 ||
                start == gpu_kvcache_->_uid_to_paged_cache_startpos.end() || start->second != 0 ||
                count == gpu_kvcache_->_uid_to_paged_cache_length.end() || count->second != length ||
                !gpu_kvcache_->_lru_lookup_table.count(ids[i]) ||
                pages->second[0] < 0 || pages->second[0] >= gpu_kvcache_->num_primary_cache_pages)
            return false;
    }
    for (int64_t i = 0; i < batch; ++i) {
        slots[i] = gpu_kvcache_->_uid_to_page_id.at(ids[i])[0];
        bool first = true;
        for (int64_t j = 0; j < i; ++j) {
            if (ids[j] == ids[i]) { first = false; break; }
        }
        if (first) {
            // Match deduplicated allocation's access order without allocating
            // a replacement list node. Existing iterators remain valid.
            gpu_kvcache_->_lru_list.splice(gpu_kvcache_->_lru_list.end(),
                gpu_kvcache_->_lru_list, gpu_kvcache_->_lru_lookup_table.at(ids[i]));
        }
    }
    return true;
}

// Admission check for one fixed slot per immutable history. Pending D2H
// transfers pin slots; allocation must wait rather than evict their storage.
bool ExportKVCacheRuntime::fixed_allocation_available(at::Tensor user_ids) {
    TORCH_CHECK(fixed_history_slots_, "fixed-slot admission requires fixed history slots");
    std::unordered_set<int64_t> wanted;
    for (int64_t i=0; i<user_ids.numel(); ++i) wanted.insert(user_ids.data_ptr<int64_t>()[i]);
    int64_t need = 0;
    for (auto id : wanted) if (!gpu_kvcache_->_uid_to_page_id.count(id)) ++need;
    int64_t available = gpu_kvcache_->_empty_pages.size();
    for (const auto& entry : gpu_kvcache_->_uid_to_page_id) {
        if (!wanted.count(entry.first) && !gpu_kvcache_->_uid_offload_lock.count(entry.first))
            available += entry.second.size();
    }
    return available >= need;
}

// Nonblocking transfer-status query. The socket round trip remains on the
// owner thread; completed IDs are consumed once and replaced by -1.
bool ExportKVCacheRuntime::onboard_kvcache_poll(at::Tensor task_ids) {
    auto *ids = task_ids.data_ptr<int64_t>();
    std::vector<int64_t> active;
    for (int64_t i=0; i<task_ids.numel(); ++i) if (ids[i]>=0) active.push_back(ids[i]);
    if (active.empty()) return true;
    std::unordered_map<int, KVTaskResponse> results;
    if (onboard_child_tasks_) {
        // These child handles share one parent. Early readiness SUCCESS does
        // not reap records on retained-child servers; require terminal parent
        // completion before consuming the first child. A zero timeout keeps
        // this an owner progress poll, including the singleton case.
        results = flexkv_client_->wait({active.front()}, 0.0f, true);
        const auto first = results.find(active.front());
        if (first == results.end() || first->second.status == KVResponseStatus::UNREADY ||
            first->second.status == KVResponseStatus::TIMEOUT) return false;
        TORCH_CHECK(first->second.status == KVResponseStatus::SUCCESS,
                    "onboarding failed, task=", active.front());
        if (active.size() > 1) {
            // The parent's terminal state now guarantees these observations
            // also release their child records and finally the parent record.
            std::vector<int64_t> remaining(active.begin() + 1, active.end());
            auto completed = flexkv_client_->try_wait(remaining);
            results.insert(completed.begin(), completed.end());
        }
    } else {
        results = flexkv_client_->try_wait(active);
    }
    bool ready = true;
    for (int64_t i=0; i<task_ids.numel(); ++i) {
        if (ids[i]<0) continue;
        auto it = results.find(ids[i]);
        if (it == results.end() || it->second.status == KVResponseStatus::UNREADY) {
            ready = false;
            continue;
        }
        TORCH_CHECK(it->second.status == KVResponseStatus::SUCCESS,
                    "onboarding failed, task=", ids[i], " status=", static_cast<int>(it->second.status));
        ids[i] = -1;
    }
    return ready;
}

at::Tensor ExportKVCacheRuntime::offload_kvcache_launch(
    at::Tensor user_ids,
    at::Tensor seqlens,
    at::Tensor merged_cached_lengths,
    at::Tensor host_cached_lengths,
    at::Tensor gpu_cached_startpos,
    at::Tensor gpu_cached_lengths,
    at::Tensor kv_page_indices,
    at::Tensor kv_page_indptr,
    std::vector<at::Tensor> slot_mappings
) {
    // FlexKV reads this storage from another process/stream. In legacy mode
    // there is no owner event to establish producer completion, so publication
    // must synchronously retire the caller's CUDA writes. The shared owner has
    // already queried its producer event before reaching this method.
    if (!cache_owner_worker_thread()) {
        at::cuda::getCurrentCUDAStream().synchronize();
    }
    auto [offload_user_ids, offload_start_indices, offload_page_indices_list] = gpu_kvcache_->acquire_offload_pages(user_ids, host_cached_lengths, true);

    std::vector<at::Tensor> new_slot_mappings;
    if (slot_mappings.size() == 1 && slot_mappings[0].data_ptr<int64_t>()[0] == -1) {
        if (fixed_history_slots_ && fast_metadata_) {
            new_slot_mappings = fixed_slot_mappings(user_ids, seqlens);
        } else {
            for (int seq_idx = 0; seq_idx < user_ids.size(0); seq_idx++) {
                auto seqlen = seqlens.data_ptr<int64_t>()[seq_idx];
                auto page_ids = kv_page_indices.cpu().slice(0, kv_page_indptr[seq_idx].item<int>(), kv_page_indptr[seq_idx + 1].item<int>());
                if (page_ids.numel() > 0) {
                    auto slot_mapping = page_ids.unsqueeze(1) * this->gpu_kvcache_->num_tokens_per_page + token_slot_mappings_in_page_.unsqueeze(0);
                    new_slot_mappings.push_back(slot_mapping.reshape(-1).slice(0, 0, seqlen));
                } else {
                    new_slot_mappings.push_back(at::empty({0}, at::dtype(torch::kInt64).device(at::kCPU)));
                }
            }
        }
    } else if (slot_mappings.size() != user_ids.size(0)) {
        throw std::runtime_error("slot_mappings size does not match batch size");
    } else {
        new_slot_mappings = slot_mappings;
    }

    auto batch_size = user_ids.size(0);
    at::Tensor offload_task_ids = at::empty({batch_size}, at::dtype(torch::kInt64).device(at::kCPU));
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        int64_t seqlen = seqlens.data_ptr<int64_t>()[seq_idx];
        int64_t valid_len = int64_t(seqlen / this->gpu_kvcache_->num_tokens_per_page) * this->gpu_kvcache_->num_tokens_per_page; // align to page boundary
        if (valid_len == 0) {
            offload_task_ids.data_ptr<int64_t>()[seq_idx] = -1;  // -1 indicates no cache to offload for this sequence
            this->gpu_kvcache_->release_offload_pages(
                at::from_blob(&uid, {1}, at::dtype(torch::kInt64).device(at::kCPU)),
                at::from_blob(&offload_start_indices.data_ptr<int>()[seq_idx], {1}, at::dtype(torch::kInt32).device(at::kCPU)),
                at::from_blob(&valid_len, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
                { true, }
            );
            continue;
        }  // skip sequences without cache to offload for this sequence
        auto user_namespace = "uid:" + std::to_string(uid);
        int64_t task_id = flexkv_client_->put_async(
            lookup_token_index_.slice(0, 0, valid_len),
            new_slot_mappings[seq_idx].slice(0, 0, valid_len),
            std::nullopt,
            {user_namespace}
        );
        offload_tasks_.push_back(
            std::make_tuple(
                task_id,
                user_ids.data_ptr<int64_t>()[seq_idx],
                offload_start_indices.data_ptr<int>()[seq_idx],
                valid_len
            )
        );
        offload_task_ids.data_ptr<int64_t>()[seq_idx] = task_id;
    }
    return offload_task_ids;
}

at::Tensor ExportKVCacheRuntime::offload_kvcache_reap_completed() {
    std::vector<int64_t> completed_tasks;
    std::list<std::tuple<int64_t, int64_t, int32_t, int32_t>> remaining_tasks;
    std::vector<int64_t> active_ids;
    for (const auto& task : offload_tasks_) active_ids.push_back(std::get<0>(task));
    if (active_ids.empty()) return at::empty({0, 3}, at::dtype(torch::kInt64).device(at::kCPU));
    auto wait_results = flexkv_client_->try_wait(active_ids);
    for (auto& task : offload_tasks_) {
        auto [task_id, user_id, offload_start_index, offload_length] = task;
        if (wait_results.find(task_id) == wait_results.end()) {
            remaining_tasks.push_back(task);  // not completed yet, keep waiting
            continue;
        }
        if (wait_results[task_id].status == KVResponseStatus::UNREADY) {
            remaining_tasks.push_back(task);  // not completed yet, keep waiting
            continue;  // completed successfully, can be reaped
        }

        int offload_succeeded = (wait_results[task_id].status == KVResponseStatus::SUCCESS);
        completed_tasks.push_back(user_id);
        completed_tasks.push_back(task_id);
        completed_tasks.push_back(static_cast<int64_t>(wait_results[task_id].status));

        gpu_kvcache_->release_offload_pages(
            at::from_blob(&user_id, {1}, at::dtype(torch::kInt64).device(at::kCPU)),
            at::from_blob(&offload_start_index, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            at::from_blob(&offload_length, {1}, at::dtype(torch::kInt32).device(at::kCPU)),
            { offload_succeeded, }
        );
    }
    offload_tasks_ = std::move(remaining_tasks);
    if (completed_tasks.empty()) {
        return at::empty({0, 3}, at::dtype(torch::kInt64).device(at::kCPU));  // return empty tensor with shape (0, 3) if no completed task
    }
    return at::from_blob(completed_tasks.data(), {(int64_t)completed_tasks.size()}, at::dtype(torch::kInt64).device(at::kCPU)).clone().reshape({-1, 3});  // reshape to Nx3 for easier parsing (user_id, task_id, status)
}

void ExportKVCacheRuntime::shutdown_runtime() {
    // The owner normally drains before reaching this raw runtime. Direct
    // standard-op callers need the same guarantee before GPU storage is freed.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    int64_t failed_task = -1;
    int64_t failed_status = 0;
    while (!offload_tasks_.empty()) {
        auto completed = offload_kvcache_reap_completed();
        const auto* rows = completed.data_ptr<int64_t>();
        for (int64_t i = 0; i < completed.size(0); ++i) {
            if (rows[3 * i + 2] != static_cast<int64_t>(KVResponseStatus::SUCCESS) && failed_task < 0) {
                failed_task = rows[3 * i + 1];
                failed_status = rows[3 * i + 2];
            }
        }
        if (offload_tasks_.empty()) break;
        TORCH_CHECK(std::chrono::steady_clock::now() < deadline,
                    "KV-cache shutdown timed out with ", offload_tasks_.size(),
                    " pending offloads; runtime storage retained for retry");
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    // Report a terminal transfer failure only after the other transfers have
    // also retired. RPC errors propagate immediately; the context retains this
    // runtime on failure instead of releasing potentially live GPU storage.
    TORCH_CHECK(failed_task < 0, "FlexKV offload failed during shutdown, task=",
                failed_task, " status=", failed_status);
}

std::vector<at::Tensor> ExportKVCacheRuntime::get_kvcache_tables() {
    return cache_tables_;
}

int64_t ExportKVCacheRuntime::get_num_offload_tasks() {
    return offload_tasks_.size();
}

void ExportKVCacheRuntime::evict_kvcache(at::Tensor user_ids, bool evict_gpu_only) {
    const auto batch_size = user_ids.size(0);
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t uid = user_ids.data_ptr<int64_t>()[seq_idx];
        gpu_kvcache_->evict(uid);

        if (!evict_gpu_only) {
        }
    }
}

} // namespace kvcache_manager
