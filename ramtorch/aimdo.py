"""Optional AIMDO adapters for inference-time weight residency and NVMe I/O.

This module deliberately imports neither ``torch`` nor ``comfy_aimdo`` at
module import time.  AIMDO must be initialized before PyTorch, so callers use
the ``ramtorch-aimdo`` launcher (or perform the same bootstrap themselves)
before constructing an :class:`ramtorch.OffloadModel` with an AIMDO backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Dict, Iterable, Optional


def require_aimdo(device):
    """Load and validate the optional AIMDO runtime for ``device``.

    Importing after Torch is only safe when the process was bootstrapped first;
    checking the initialized device context gives users a deterministic error
    instead of an eventual invalid raw-pointer access.
    """
    try:
        import comfy_aimdo.control as control
        from comfy_aimdo.model_vbar import (
            ModelVBAR,
            vbar_fault,
            vbar_signature_compare,
            vbar_unpin,
        )
        from comfy_aimdo.torch import aimdo_to_tensor
        from comfy_aimdo.host_buffer import read_file_to_device
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "AIMDO backend requested but comfy-aimdo is not installed; "
            "install it and launch through ramtorch-aimdo."
        ) from exc

    index = device.index
    if index is None:
        raise ValueError("AIMDO requires an explicit CUDA device index")
    if control.lib is None or not control.devctxs:
        raise RuntimeError(
            "AIMDO was not initialized before PyTorch. Launch with "
            "'ramtorch-aimdo --devices %d -- <script> ...'." % index
        )
    try:
        control.get_devctx(index)
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            "AIMDO was not initialized for %s; include it in --devices." % device
        ) from exc
    return _AimdoApi(
        ModelVBAR, vbar_fault, vbar_signature_compare, vbar_unpin,
        aimdo_to_tensor, read_file_to_device,
    )


@dataclass(frozen=True)
class _AimdoApi:
    ModelVBAR: object
    vbar_fault: object
    vbar_signature_compare: object
    vbar_unpin: object
    aimdo_to_tensor: object
    read_file_to_device: object


@dataclass
class _TensorSlot:
    alloc: object
    tensor: object
    signature: Optional[object] = None


class AimdoResidency:
    """A VBAR cache for inference-only ``OffloadModel`` weights.

    AIMDO owns the virtual address range and may evict unpinned pages under
    pressure.  A slot is unpinned only after the compute-stream event recorded
    by :meth:`release` completes, preventing native eviction while a queued
    PyTorch kernel still reads the raw allocation.
    """

    def __init__(self, states: Iterable[object], device):
        import torch

        self.device = device
        self._torch = torch
        self.api = require_aimdo(device)
        total = sum(
            tensor.numel() * tensor.element_size()
            for state in states for tensor in state.tensors.values()
        )
        self.vbar = self.api.ModelVBAR(total, device.index)
        self._lock = threading.Lock()
        self._slots: Dict[int, Dict[str, _TensorSlot]] = {}
        self._pending = []
        self._loaded: Dict[int, list] = {}

        for index, state in enumerate(states):
            slots = {}
            for name, source in state.tensors.items():
                alloc = self.vbar.alloc(source.numel() * source.element_size())
                tensor = self.api.aimdo_to_tensor(alloc, device).view(source.dtype)
                slots[name] = _TensorSlot(alloc, tensor.view(source.shape))
            self._slots[index] = slots

    def _reap(self) -> None:
        keep = []
        for event, allocs in self._pending:
            if event.query():
                for alloc in allocs:
                    self.api.vbar_unpin(alloc)
            else:
                keep.append((event, allocs))
        self._pending = keep

    def materialize(self, index: int, source: Dict[str, object]) -> Dict[str, object]:
        with self._lock:
            self._reap()
            out = {}
            pinned = []
            for name, cpu_tensor in source.items():
                slot = self._slots[index][name]
                signature = self.api.vbar_fault(slot.alloc)
                if signature is None:
                    # Keep the established streamed path for a weight AIMDO
                    # could not retain under current pressure.
                    out[name] = cpu_tensor.to(self.device, non_blocking=True)
                    continue
                if not self.api.vbar_signature_compare(signature, slot.signature):
                    slot.tensor.copy_(cpu_tensor, non_blocking=True)
                    slot.signature = signature
                out[name] = slot.tensor
                pinned.append(slot.alloc)
            self._loaded[index] = pinned
            return out

    def release(self, index: int) -> None:
        with self._lock:
            allocs = self._loaded.pop(index, ())
            if not allocs:
                return
            event = self._torch.cuda.Event()
            event.record(self._torch.cuda.current_stream(self.device))
            self._pending.append((event, allocs))
            self._reap()

    def close(self) -> None:
        # Synchronization is only at explicit teardown; hot-path releases are
        # event-driven and never block the caller.
        with self._lock:
            for event, allocs in self._pending:
                event.synchronize()
                for alloc in allocs:
                    self.api.vbar_unpin(alloc)
            for allocs in self._loaded.values():
                for alloc in allocs:
                    self.api.vbar_unpin(alloc)
            self._pending = []
            self._loaded = {}
