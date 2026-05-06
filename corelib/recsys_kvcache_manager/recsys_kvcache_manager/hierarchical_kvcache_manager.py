import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from configs import KVCacheMetadata
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from typing import Any, List, Dict, Optional, Tuple, Union
from uuid import uuid4
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum


class SecondaryTaskStatus(Enum):
    UNINITIALIZED = "uninitialized"
    SKIPPED = "skipped"
    LAUNCHED = "launched"
    READY = "ready"
    EVENT_READY = "event_ready"
    TIMEOUT = "timeout"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SecondaryErrorCode(str, Enum):
    SDK_IMPORT_FAILED = "sdk_import_failed"
    SDK_INIT_FAILED = "sdk_init_failed"
    LOOKUP_FAILED = "lookup_failed"
    LOOKUP_MISSING_TOKENS = "lookup_missing_tokens"
    ONBOARD_TASK_NOT_FOUND = "onboard_task_not_found"
    ONBOARD_WAIT_FAILED = "onboard_wait_failed"
    ONBOARD_TIMEOUT = "onboard_timeout"
    OFFLOAD_TASK_NOT_FOUND = "offload_task_not_found"
    OFFLOAD_WAIT_FAILED = "offload_wait_failed"
    OFFLOAD_TIMEOUT = "offload_timeout"
    CANCEL_FAILED = "cancel_failed"

@dataclass
class SecondaryTaskHandle:
    backend: str
    user_ids: Optional[torch.Tensor] = None
    handle: Optional[Any]
    status: SecondaryTaskStatus = SecondaryTaskStatus.UNINITIALIZED
    metadata: Optional[Dict[str, Any]] = None
    time_launched: Optional[float] = None

@dataclass
class SecondaryWaitResult:
    status: SecondaryTaskStatus
    ready: bool
    error_code: Optional[str] = None
    message: str = ""
    failed_mask: Optional[torch.Tensor] = None
    failed_user_ids: Optional[List[int]] = None

class SecondaryKVCacheManagerBase(ABC):
    @abstractmethod
    def lookup_kvcache(self, index_meta: KVIndexMeta) -> KVLookupResult:
        ...

    @abstractmethod
    def onboard_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        kvcache_metadata: KVCacheMetadata,
    ) -> SecondaryTaskHandle:
        ...

    @abstractmethod
    def onboard_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        ...

    @abstractmethod
    def offload_launch_kvcache(
        self,
        index_meta: KVIndexMeta,
        append_slot_mapping: Optional[torch.Tensor],
        append_slot_indptr: Optional[torch.Tensor] = None,
    ) -> SecondaryTaskHandle:
        ...

    @abstractmethod
    def offload_wait_kvcache(self, task_handle: SecondaryTaskHandle) -> SecondaryWaitResult:
        ...

    @abstractmethod
    def cancel_task(self, task_handle: SecondaryTaskHandle) -> None:
        ...

    @abstractmethod
    def register_gpu_cache_tensors(self, cache_table_list: List[torch.Tensor]) -> None:
        ...
    
    @abstractmethod
    def build_index_meta(self, user_ids: torch.Tensor, history_sequence_lengths: torch.Tensor) -> KVIndexMeta:
        ...


class NopSecondaryKVCacheManager(SecondaryKVCacheManagerBase):
    def lookup_kvcache(self, index_meta: KVIndexMeta):
        return {"backend": "nop", "hit_mask": None}

    def onboard_launch_kvcache(self, index_meta, restore_slot_mapping):
        return SecondaryTaskHandle(
            backend="nop",
            handle=None,
            status=SecondaryTaskStatus.SKIPPED,
            metadata={"reason": "nop backend"},
        )

    def onboard_wait_kvcache(self, task_handle):
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.READY,
            ready=True,
            message="nop onboard wait ready",
        )

    def offload_launch_kvcache(
        self,
        index_meta,
        append_slot_mapping,
        append_slot_indptr=None,
    ):
        return SecondaryTaskHandle(
            backend="nop",
            handle=None,
            status=SecondaryTaskStatus.SKIPPED,
            metadata={"reason": "nop backend"},
        )

    def offload_wait_kvcache(self, task_handle):
        return SecondaryWaitResult(
            status=SecondaryTaskStatus.READY,
            ready=True,
            message="nop offload wait ready",
        )

    def cancel_task(self, task_handle):
        return None