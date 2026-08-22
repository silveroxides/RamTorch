"""CPU-only smoke test for the direct NVMe-reader primitive.

Run with ``python examples/nvme_store_smoke.py``.  It intentionally avoids a
CUDA dependency so CI and developers can verify the portable I/O path.
"""

import tempfile
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramtorch.nvme_store import NvmeTensorStore


def main() -> int:
    tensors = {
        "weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "bias": torch.arange(7, dtype=torch.int16),
    }
    with tempfile.TemporaryDirectory() as directory:
        store = NvmeTensorStore(str(Path(directory) / "weights.bin"))
        mapped = store.write(tensors)
        for name, expected in tensors.items():
            _, nbytes = store.file_slice(name)
            raw = torch.empty(nbytes, dtype=torch.uint8, pin_memory=False)
            store.readinto(name, raw)
            actual = raw.view(expected.dtype).view(expected.shape)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(mapped[name], expected)
        store.close()
    print("NVMe direct-reader smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
