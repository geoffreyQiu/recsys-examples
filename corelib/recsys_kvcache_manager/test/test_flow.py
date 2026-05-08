import atexit
import time

import torch
import kvcache_cpp

from recsys_kvcache_manager.kvcache_manager import KVCacheManager
from recsys_kvcache_manager.kvcache_config import get_kvcache_config


def _shutdown_flexkv_client(kvcache_mgr) -> None:
    if kvcache_mgr is None:
        return
    secondary_mgr = getattr(kvcache_mgr, "secondary_kvcache_manager", None)
    if secondary_mgr is None:
        return
    client = getattr(secondary_mgr, "_client", None)
    if client is None:
        return
    try:
        client.shutdown()
        # Prevent duplicated shutdown during interpreter teardown.
        secondary_mgr._client = None
        print("[DEBUG] FlexKV client shutdown completed.")
    except Exception as e:
        print(f"[WARN] FlexKV shutdown failed: {e}")


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
        secondary_backend="flexkv",
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
    atexit.register(_shutdown_flexkv_client, kvcache_mgr)


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

        # - token_ids: random int64 tokens per user
        # - token_mask: valid token range by each sequence length
        batch_size = int(user_ids.numel())
        max_seq_len = int(sequence_lengths.max().item())
        token_ids = torch.randint(
            low=0,
            high=100000,
            size=(batch_size, max_seq_len),
            dtype=torch.int64,
        )
        token_mask = torch.arange(max_seq_len, dtype=torch.int32).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        index_meta.token_ids = token_ids
        index_meta.token_mask = token_mask

        ret = kvcache_mgr.eager_offboard_kvcache(index_meta)
        assert ret == None, "[ERROR] Lazy mode does not trigger eager offload"
        offload_handle = kvcache_mgr.lazy_offload_kvcache(index_meta, kvcache_metadata)

        while True:
            kvcache_mgr.finish_or_cancel_kvcache_ops()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
        
        index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
            user_ids, sequence_lengths)
        print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
        print(f"  \t{lookup_res.cached_start_indices}")
        print(f"  \t{lookup_res.cached_lengths}")
        print("offload status:", offload_handle.status if offload_handle else None)
        print("ongoing offload tasks:", len(kvcache_mgr.ongoing_offload_tasks))


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

        # FlexKV offload input: prepare per-user tokens.
        batch_size = int(user_ids.numel())
        max_seq_len = int(sequence_lengths.max().item())
        token_ids = torch.randint(
            low=0,
            high=100000,
            size=(batch_size, max_seq_len),
            dtype=torch.int64,
        )
        token_mask = torch.arange(max_seq_len, dtype=torch.int32).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        index_meta.token_ids = token_ids
        index_meta.token_mask = token_mask

        ret = kvcache_mgr.eager_offboard_kvcache(index_meta)
        assert ret == None, "[ERROR] Lazy mode does not trigger eager offload"
        offload_handle = kvcache_mgr.lazy_offload_kvcache(index_meta, kvcache_metadata)

        while True:
            kvcache_mgr.finish_or_cancel_kvcache_ops()
            if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                break
        
        for i in range(len(user_ids)):
            uid = user_ids[i].item()
            print(f"[DEBUG] Check for uid:{user_ids[i].item()} ...")
            start, end = kvcache_metadata.total_history_offsets[i], kvcache_metadata.total_history_offsets[i+1]
            k, v = key[:, start:end, ...], value[:, start:end, ...]
            
            if kvcache_config.secondary_backend == "native":
                impl = kvcache_mgr.secondary_kvcache_manager.kvcache_mananger_impl
                kvdata = impl.get_kvdata_tensor([uid], False)[0]
                cached_k, cached_v = kvdata.unbind(dim=2)
                cached_k = cached_k.reshape(cached_k.size(0), -1, cached_k.size(3), cached_k.size(4))
                cached_v = cached_v.reshape(cached_v.size(0), -1, cached_v.size(3), cached_v.size(4))
                print(f"        \tK: {torch.allclose(cached_k.cuda(), k[:, :cached_k.size(1), ...])}")
                print(f"        \tV: {torch.allclose(cached_v.cuda(), v[:, :cached_v.size(1), ...])}")
            elif kvcache_config.secondary_backend == "flexkv":
                # FlexKV path: verify offloaded tokens can be queried back.
                # Do host-only lookup via secondary manager directly.
                # KVCacheManager.lookup_kvcache() intentionally masks host lookup
                # when GPU has full hit, which would always return zero here.
                namespace = (
                    index_meta.namespaces[i]
                    if hasattr(index_meta, "namespaces")
                    and index_meta.namespaces is not None
                    and i < len(index_meta.namespaces)
                    else f"uid:{uid}"
                )
                flex_cached_len = 0
                for _ in range(20):
                    sec_index_meta = kvcache_mgr.secondary_kvcache_manager.build_index_meta(
                        user_ids[i : i + 1],
                        sequence_lengths[i : i + 1],
                    )
                    if hasattr(sec_index_meta, "token_ids"):
                        sec_index_meta.token_ids = token_ids[i : i + 1]
                    if hasattr(sec_index_meta, "token_mask"):
                        sec_index_meta.token_mask = token_mask[i : i + 1]
                    if hasattr(sec_index_meta, "namespaces"):
                        sec_index_meta.namespaces = [namespace]
                    flex_lookup = kvcache_mgr.secondary_kvcache_manager.lookup_kvcache(sec_index_meta)
                    flex_cached_len = int(flex_lookup.host_cached_lengths[0].item())
                    if flex_cached_len > 0:
                        break
                    time.sleep(0.05)
                print(f"        \tFlexKV cached length: {flex_cached_len}")
                assert flex_cached_len > 0, (
                    f"[ERROR] FlexKV offload missing for uid {uid}, namespace={namespace}"
                )
            else:
                raise ValueError(
                    f"Unsupported secondary backend: {kvcache_config.secondary_backend}"
                )


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
        


    if False:
        print("[DEBUG] ===== Section 4: FlexKV offload A/B page-tail check =====")
        user_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        ab_cases = [
            ("A_aligned", torch.tensor([96, 64, 96, 64], dtype=torch.int32)),
            ("B_plus_one", torch.tensor([97, 65, 97, 64], dtype=torch.int32)),
        ]

        for case_name, sequence_lengths in ab_cases:
            print(f"\n[DEBUG] ---- {case_name} ----")
            print(f"[DEBUG] sequence_lengths: {sequence_lengths.tolist()}")

            # Keep each case isolated to make D2H size comparison reliable.
            kvcache_mgr.evict_all(for_gpu=True, for_host=True)

            index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)
            kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

            # Same token generation style as benchmark_single_batch.py.
            batch_size = int(user_ids.numel())
            max_seq_len = int(sequence_lengths.max().item())
            token_ids = torch.randint(
                low=0,
                high=100000,
                size=(batch_size, max_seq_len),
                dtype=torch.int64,
            )
            token_mask = torch.arange(max_seq_len, dtype=torch.int32).unsqueeze(0) < sequence_lengths.unsqueeze(1)
            index_meta.token_ids = token_ids
            index_meta.token_mask = token_mask

            ret = kvcache_mgr.eager_offboard_kvcache(index_meta)
            assert ret == None, "[ERROR] Lazy mode does not trigger eager offload"
            offload_handle = kvcache_mgr.lazy_offload_kvcache(index_meta, kvcache_metadata)

            while True:
                kvcache_mgr.finish_or_cancel_kvcache_ops()
                if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                    break

            total_tokens = int(sequence_lengths.sum().item())
            aligned_tokens = sum((int(x) // 32) * 32 for x in sequence_lengths.tolist())
            tail_tokens = total_tokens - aligned_tokens
            print(f"[DEBUG] offload status: {offload_handle.status if offload_handle else None}")
            print(f"[DEBUG] token summary: total={total_tokens}, aligned={aligned_tokens}, tail={tail_tokens}")

    _shutdown_flexkv_client(kvcache_mgr)
