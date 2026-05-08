import time

import torch
import kvcache_cpp

from recsys_kvcache_manager.kvcache_manager import KVCacheManager
from recsys_kvcache_manager.kvcache_config import get_kvcache_config


def _shutdown_flexkv_client(kvcache_mgr) -> None:
    if kvcache_mgr is None:
        return
    secondary_mgr = kvcache_mgr.secondary_kvcache_manager
    client = getattr(secondary_mgr, "_client", None)
    if client is not None:
        try:
            client.shutdown()
        except Exception as e:
            print(f"[WARN] FlexKV shutdown failed: {e}")
        finally:
            # Avoid running FlexKV client destructor during interpreter teardown.
            try:
                secondary_mgr._client = None
            except Exception:
                pass


if __name__ == "__main__":
    kvcache_mgr = None
    try:
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
        
        if True:
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

            # Prepare valid token payload so FlexKV offload request is not skipped.
            max_len = int(sequence_lengths.max().item())
            token_ids = torch.zeros((user_ids.size(0), max_len), dtype=torch.int64)
            token_mask = torch.zeros((user_ids.size(0), max_len), dtype=torch.bool)
            for i in range(user_ids.size(0)):
                seq_len_i = int(sequence_lengths[i].item())
                token_ids[i, :seq_len_i] = torch.arange(seq_len_i, dtype=torch.int64)
                token_mask[i, :seq_len_i] = True
            index_meta.token_ids = token_ids
            index_meta.token_mask = token_mask
            index_meta.namespaces = [f"uid:{int(uid.item())}" for uid in user_ids]

            ret = kvcache_mgr.eager_offboard_kvcache(index_meta)
            assert ret == None, "[ERROR] Lazy mode does not trigger eager offload"
            offload_handle = kvcache_mgr.lazy_offload_kvcache(index_meta, kvcache_metadata)
            assert offload_handle is not None, "[ERROR] offload handle should not be None"
            assert offload_handle.status == offload_handle.status.LAUNCHED, (
                f"[ERROR] Expected LAUNCHED offload task, got {offload_handle.status}"
            )

            while True:
                kvcache_mgr.finish_or_cancel_kvcache_ops()
                if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                    break
            
            # Strict validation: force GPU miss, then verify secondary readback.
            kvcache_mgr.evict_all(for_gpu=True)
            index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
                user_ids,
                sequence_lengths,
                token_ids=token_ids,
                token_mask=token_mask,
                namespaces=[f"uid:{int(uid.item())}" for uid in user_ids],
            )
            print(f"[DEBUG] Lookup Results after GPU evict for {user_ids.tolist()}:")
            print(f"  \t{lookup_res.cached_start_indices}")
            print(f"  \t{lookup_res.cached_lengths}")
            print(f"  \tGPU cached lengths: {lookup_res.gpu_cached_lengths}")
            print(f"  \tHost cached lengths: {lookup_res.host_cached_lengths}")
            print(f"  \tLookup extra: {lookup_res.extra}")
            print("offload status:", offload_handle.status if offload_handle else None)
            print("ongoing offload tasks:", len(kvcache_mgr.ongoing_offload_tasks))
            assert lookup_res.gpu_cached_lengths is not None, "[ERROR] missing gpu_cached_lengths"
            assert lookup_res.host_cached_lengths is not None, "[ERROR] missing host_cached_lengths"
            assert torch.all(lookup_res.gpu_cached_lengths == 0), (
                f"[ERROR] GPU should be evicted, got {lookup_res.gpu_cached_lengths}"
            )
            page_size = int(kvcache_mgr.gpu_kvcache_mgr.page_size)
            expected_host = (sequence_lengths // page_size) * page_size
            expected_host = expected_host.to(
                device=lookup_res.host_cached_lengths.device,
                dtype=lookup_res.host_cached_lengths.dtype,
            )
            assert torch.equal(lookup_res.host_cached_lengths, expected_host), (
                "[ERROR] Secondary readback mismatch (block-aligned expectation), "
                f"expected {expected_host}, got {lookup_res.host_cached_lengths}, extra={lookup_res.extra}"
            )

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
            offload_handle = kvcache_mgr.lazy_offload_kvcache(index_meta, kvcache_metadata)

            while True:
                kvcache_mgr.finish_or_cancel_kvcache_ops()
                if len(kvcache_mgr.ongoing_offload_tasks) == 0:
                    break
            
            index_meta, lookup_res = kvcache_mgr.lookup_kvcache(
                user_ids, sequence_lengths)
            print(f"[DEBUG] Lookup Results for {user_ids.tolist()}:")
            print(f"  \t{lookup_res.gpu_cached_start_indices}")
            print(f"  \t{lookup_res.gpu_cached_lengths}")
            print(f"  \t{lookup_res.host_cached_start_indices}")
            print(f"  \t{lookup_res.host_cached_lengths}")


            print(f"[DEBUG] == Eviction for GPU ==")
            kvcache_mgr.evict(user_ids, for_gpu=True)
            kvcache_mgr.evict_all(for_gpu=True)

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
    finally:
        _shutdown_flexkv_client(kvcache_mgr)

        


