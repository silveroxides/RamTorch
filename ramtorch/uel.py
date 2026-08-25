"""UEL-backed asynchronous safetensors transport for RamTorch inference."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Sequence


def _require_uel():
    try:
        from unifiedefficientloader import (
            IncrementalSafetensorsWriter,
            UnifiedSafetensorsLoader,
        )
    except ImportError as exc:  # pragma: no cover - optional package
        raise RuntimeError(
            "UEL backend requested but unifiedefficientloader is not installed; "
            "install RamTorch with its 'uel' extra."
        ) from exc
    return UnifiedSafetensorsLoader, IncrementalSafetensorsWriter


class AsyncSafetensorsSource:
    """Consume UEL's bounded async stream in RamTorch chunk order.

    ``key_map`` maps ``"{chunk_index}.{state_tensor_name}"`` to the source
    safetensors key.  Loading happens concurrently in UEL worker threads;
    this class only consumes the ordered stream on RamTorch's existing loader
    thread, retaining at most the current chunk's tensors.
    """

    def __init__(
        self,
        path: str,
        key_map: Mapping[str, str],
        *,
        prefetch_batches: int = 2,
    ):
        self._UnifiedSafetensorsLoader, _ = _require_uel()
        self._path = path
        self._prefetch_batches = prefetch_batches
        self._by_chunk: Dict[int, Dict[str, str]] = defaultdict(dict)
        for global_name, source_name in key_map.items():
            chunk, separator, tensor = global_name.partition(".")
            if not separator or not chunk.isdigit() or not tensor:
                raise ValueError(
                    "UEL key-map entries must use '{chunk_index}.{tensor_name}'"
                )
            self._by_chunk[int(chunk)][tensor] = source_name
        if not self._by_chunk:
            raise ValueError("UEL key_map must not be empty")

        self._keys = [
            source_name
            for chunk in sorted(self._by_chunk)
            for source_name in self._by_chunk[chunk].values()
        ]
        self._loader = None
        self._stream = None
        self._pending: Dict[str, object] = {}

    def start_pass(self) -> None:
        """Start a fresh ordered UEL prefetch pass for one model forward."""
        self.close()
        self._loader = self._UnifiedSafetensorsLoader(self._path, low_memory=True)
        self._stream = iter(self._loader.async_stream(
            self._keys, batch_size=1,
            prefetch_batches=self._prefetch_batches, pin_memory=True,
        ))
        self._pending = {}

    def take_chunk(self, index: int, expected: Mapping[str, object]) -> Dict[str, object]:
        if self._loader is None or self._stream is None:
            raise RuntimeError("start_pass() must be called before take_chunk()")
        try:
            names = self._by_chunk[index]
        except KeyError as exc:
            raise KeyError(f"UEL key_map has no chunk {index}") from exc
        out = {}
        for tensor_name, source_name in names.items():
            while source_name not in self._pending:
                batch = next(self._stream)
                for loaded_name, tensor in batch:
                    self._pending[loaded_name] = tensor
            tensor = self._pending.pop(source_name)
            reference = expected[tensor_name]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(
                    f"UEL tensor {source_name!r} does not match chunk {index}.{tensor_name}: "
                    f"got {tuple(tensor.shape)} {tensor.dtype}, expected "
                    f"{tuple(reference.shape)} {reference.dtype}"
                )
            out[tensor_name] = tensor
            self._loader.mark_processed(source_name)
        if set(out) != set(expected):
            missing = sorted(set(expected) - set(out))
            raise ValueError(f"UEL key_map misses chunk {index} tensors: {missing[:5]}")
        return out

    def close(self) -> None:
        if self._loader is not None:
            self._loader.close()
            self._loader = None
        self._stream = None
        self._pending = {}


def save_safetensors_incremental(path: str, state_dict, *, max_workers: int = 4) -> None:
    """Write a state dict through UEL's asynchronous incremental writer."""
    _, IncrementalSafetensorsWriter = _require_uel()
    with IncrementalSafetensorsWriter(path, max_workers=max_workers) as writer:
        writer.write_dict(state_dict)
