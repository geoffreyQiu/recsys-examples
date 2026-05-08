import time

import torch
import kvcache_cpp

from recsys_kvcache_manager.kvcache_manager import KVCacheManager
from recsys_kvcache_manager.kvcache_config import get_kvcache_config


if __name__ == "__main__":
    kvcache_config = get_kvcache_config(
        num_layers=3,
        num_heads=4,
        head_dim=128,
        page_size=32,
        offload_chunksize=64,
        num_primary_cache_pages=4096,
        num_buffer_pages=256,
        host_capacity_per_layer=40960 * 2 * 32 * 4 * 128 * 2,
        max_batch_size=8,
        max_seq_len=1024,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        secondary_backend="native",
        offload_mode="lazy",
        offload_timeout_ms=100.0,
    )
    print(f"[DEBUG] KVCache GPU Memory Usage: {\
        (kvcache_config.num_layers * \
        kvcache_config.num_primary_cache_pages * \
        kvcache_config.page_size * \
        2 * kvcache_config.num_heads * \
        kvcache_config.head_dim * 2) / (1024. ** 3) \
        } GiB.")
    print(f"[DEBUG] KVCache GPU Memory Usage: {\
        kvcache_config.host_capacity_per_layer / (1024. ** 3) \
        } GiB.")
    
    kvcache_mgr = KVCacheManager.from_config(kvcache_config)
    print(kvcache_mgr)
    print(f"[DEBUG] Created KVCache Manager") 


    if False:
        user_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sequence_lengths = torch.tensor([100, 64, 88, 97], dtype=torch.int32)

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.cached_start_indices}")
        print(f"  \t{lookup_res.cached_lengths}")

        kvcache_metadata = kvcache_mgr.allocate_kvcache(
            index_meta, lookup_res)
        print(f"[DEBUG] Allocates Results for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.kv_indices}")
        print(f"  \t{kvcache_metadata.kv_indptr}")
        print(f"  \t{kvcache_metadata.kv_last_page_len}")
        print(f"  \t{kvcache_metadata.total_history_lengths}")
        print(f"  \t{kvcache_metadata.total_history_offsets}")
        print(f"  \t{kvcache_metadata.new_history_offsets}")

        print(f"[DEBUG] Appending Metadata for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.batch_indices}")
        print(f"  \t{kvcache_metadata.position}")
        print(f"  \t{kvcache_metadata.new_history_nnz_cuda}")
        print(f"  \t{kvcache_metadata.new_history_nnz}")

        sequence_lengths *= 2
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.cached_start_indices}")
        print(f"  \t{lookup_res.cached_lengths}")

        kvcache_metadata = kvcache_mgr.allocate_kvcache(
            index_meta, lookup_res)
        print(f"[DEBUG] Allocates Results for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.kv_indices}")
        print(f"  \t{kvcache_metadata.kv_indptr}")
        print(f"  \t{kvcache_metadata.kv_last_page_len}")
        print(f"  \t{kvcache_metadata.total_history_lengths}")
        print(f"  \t{kvcache_metadata.total_history_offsets}")
        print(f"  \t{kvcache_metadata.new_history_offsets}")

        print(f"[DEBUG] Appending Metadata for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.batch_indices}")
        print(f"  \t{kvcache_metadata.position}")
        print(f"  \t{kvcache_metadata.new_history_nnz_cuda}")
        print(f"  \t{kvcache_metadata.new_history_nnz}")
    
    if False:
        user_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sequence_lengths = torch.tensor([100, 64, 88, 97], dtype=torch.int32)

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.cached_start_indices}")
        print(f"  \t{lookup_res.cached_lengths}")

        kvcache_metadata = kvcache_mgr.allocate_kvcache(
            index_meta, lookup_res)
        print(f"[DEBUG] Allocates Results for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.kv_indices}")
        print(f"  \t{kvcache_metadata.kv_indptr}")
        print(f"  \t{kvcache_metadata.kv_last_page_len}")
        print(f"  \t{kvcache_metadata.total_history_lengths}")
        print(f"  \t{kvcache_metadata.total_history_offsets}")
        print(f"  \t{kvcache_metadata.new_history_offsets}")

        print(f"[DEBUG] Appending Metadata for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.batch_indices}")
        print(f"  \t{kvcache_metadata.position}")
        print(f"  \t{kvcache_metadata.new_history_nnz_cuda}")
        print(f"  \t{kvcache_metadata.new_history_nnz}")

        ret = kvcache_mgr.eager_offboard_kvcache(index_meta)
        assert ret == None, "[ERROR] Lazy mode does not trigger eager offload"
        offload_handle = kvcache_mgr.lazy_offload_kvcache(index_meta)

        while True:
            kvcache_mgr.finish_or_cancel_kvcache_ops()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
        
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.cached_start_indices}")
        print(f"  \t{lookup_res.cached_lengths}")


    if True:
        user_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        sequence_lengths = torch.tensor([100, 64, 88, 97], dtype=torch.int32)
        key = torch.randn((3, torch.sum(sequence_lengths).item(), 4, 128), dtype=torch.bfloat16).cuda()
        value = torch.randn((3, torch.sum(sequence_lengths).item(), 4, 128), dtype=torch.bfloat16).cuda()

        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.cached_start_indices}")
        print(f"  \t{lookup_res.cached_lengths}")

        kvcache_metadata = kvcache_mgr.allocate_kvcache(
            index_meta, lookup_res)
        print(f"[DEBUG] Allocates Results for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.kv_indices}")
        print(f"  \t{kvcache_metadata.kv_indptr}")
        print(f"  \t{kvcache_metadata.kv_last_page_len}")
        print(f"  \t{kvcache_metadata.total_history_lengths}")
        print(f"  \t{kvcache_metadata.total_history_offsets}")
        print(f"  \t{kvcache_metadata.new_history_offsets}")

        print(f"[DEBUG] Appending Metadata for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.batch_indices}")
        print(f"  \t{kvcache_metadata.position}")
        print(f"  \t{kvcache_metadata.new_history_nnz_cuda}")
        print(f"  \t{kvcache_metadata.new_history_nnz}")

        for layer_idx in range(3):
            kvcache_mgr.gpu_kvcache_mgr.put(key[layer_idx], value[layer_idx], layer_idx, kvcache_metadata)
        torch.cuda.synchronize()
        for i in range(len(user_ids)):
            print(f"[DEBUG] Check for uid:{user_ids[i].item()} ...")
            start, end = kvcache_metadata.total_history_offsets[i], kvcache_metadata.total_history_offsets[i+1]
            k, v = key[:, start:end, ...], value[:, start:end, ...]
            for layer_idx in range(3):
                page_ids = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[i]:kvcache_metadata.kv_indptr[i+1]]
                last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
                cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(page_ids, last_page_lens, layer_idx)
                print(f"[DEBUG]   Layer_{layer_idx} keys: {torch.allclose(cached_k, k[layer_idx])} , values: {torch.allclose(cached_v, v[layer_idx])}")

        ret = kvcache_mgr.eager_offboard_kvcache(index_meta)
        assert ret == None, "[ERROR] Lazy mode does not trigger eager offload"
        offload_handle = kvcache_mgr.lazy_offload_kvcache(index_meta)

        while True:
            kvcache_mgr.finish_or_cancel_kvcache_ops()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
        
        for i in range(len(user_ids)):
            uid = user_ids[i].item()
            print(f"[DEBUG] Check for uid:{user_ids[i].item()} ...")
            start, end = kvcache_metadata.total_history_offsets[i], kvcache_metadata.total_history_offsets[i+1]
            k, v = key[:, start:end, ...], value[:, start:end, ...]
            
            kvdata = kvcache_mgr.secondary_kvcache_manager.kvcache_mananger_impl.get_kvdata_tensor([uid,], False)[0]
            cached_k, cached_v = kvdata.unbind(dim=2)
            cached_k = cached_k.reshape(cached_k.size(0), -1, cached_k.size(3), cached_k.size(4))
            cached_v = cached_v.reshape(cached_v.size(0), -1, cached_v.size(3), cached_v.size(4))
            print(f"        \tK: {torch.allclose(cached_k.cuda(), k[:, :cached_k.size(1), ...])}")
            print(f"        \tV: {torch.allclose(cached_v.cuda(), v[:, :cached_v.size(1), ...])}")


        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.gpu_cached_start_indices}")
        print(f"  \t{lookup_res.gpu_cached_lengths}")
        print(f"  \t{lookup_res.host_cached_start_indices}")
        print(f"  \t{lookup_res.host_cached_lengths}")

        print(f"[DEBUG] == Eviction for GPU ==")
        kvcache_mgr.evict(user_ids, for_gpu=True)
        # kvcache_mgr.evict_all(for_gpu=True)

        # sequence_lengths *= 2
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.gpu_cached_start_indices}")
        print(f"  \t{lookup_res.gpu_cached_lengths}")
        print(f"  \t{lookup_res.host_cached_start_indices}")
        print(f"  \t{lookup_res.host_cached_lengths}")


        kvcache_metadata = kvcache_mgr.allocate_kvcache(
            index_meta, lookup_res)
        print(f"[DEBUG] Allocates Results for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.kv_indices}")
        print(f"  \t{kvcache_metadata.kv_indptr}")
        print(f"  \t{kvcache_metadata.kv_last_page_len}")
        print(f"  \t{kvcache_metadata.total_history_lengths}")
        print(f"  \t{kvcache_metadata.total_history_offsets}")
        print(f"  \t{kvcache_metadata.new_history_offsets}")

        print(f"[DEBUG] Appending Metadata for {user_ids.tolist()}:")
        print(f"  \t{kvcache_metadata.batch_indices}")
        print(f"  \t{kvcache_metadata.position}")
        print(f"  \t{kvcache_metadata.new_history_nnz_cuda}")
        print(f"  \t{kvcache_metadata.new_history_nnz}")


        kvcache_metadata.kv_onload_handle = kvcache_mgr.onboard_launch_kvcache(
            index_meta,
            lookup_res,
            kvcache_metadata
        )

        for layer_idx in range(3):
            kvcache_metadata.kv_onload_handle.handle.wait_host(layer_idx)
            print(f"[DEBUG] Layer {layer_idx} event recorded.")
        
        # torch.cuda.synchronize()

        for i in range(len(user_ids)):
            uid = user_ids[i].item()
            print(f"[DEBUG] Check for uid:{user_ids[i].item()} ...")
            start = kvcache_metadata.total_history_offsets[i]
            end = start + lookup_res.host_cached_lengths[i]
            k, v = key[:, start:end, ...], value[:, start:end, ...]
            
            for layer_idx in range(3):
                page_ids = kvcache_metadata.kv_indices[kvcache_metadata.kv_indptr[i]:kvcache_metadata.kv_indptr[i+1]]
                last_page_lens = kvcache_metadata.kv_last_page_len[i].item()
                cached_k, cached_v = kvcache_mgr.gpu_kvcache_mgr.get(page_ids, last_page_lens, layer_idx)
                cached_k = cached_k[:lookup_res.host_cached_lengths[i], ...]
                cached_v = cached_v[:lookup_res.host_cached_lengths[i], ...]
                print(f"[DEBUG]   Layer_{layer_idx} keys: {torch.allclose(cached_k, k[layer_idx])} , values: {torch.allclose(cached_v, v[layer_idx])}")
        


