"""CPU smoke test for UEL-backed asynchronous safetensors inference."""

import tempfile
from pathlib import Path

import torch

from ramtorch import OffloadModel


def main() -> int:
    try:
        from unifiedefficientloader import IncrementalSafetensorsWriter
    except ImportError as exc:
        raise SystemExit("Install unifiedefficientloader to run this smoke test") from exc

    chunks = [torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)]
    key_map = {
        "0.weight": "first.weight",
        "0.bias": "first.bias",
        "2.weight": "last.weight",
        "2.bias": "last.bias",
    }
    payload = {
        "first.weight": chunks[0].weight.detach(),
        "first.bias": chunks[0].bias.detach(),
        "last.weight": chunks[2].weight.detach(),
        "last.bias": chunks[2].bias.detach(),
    }
    x = torch.randn(3, 4)
    expected = chunks[2](chunks[1](chunks[0](x)))

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "weights.safetensors")
        with IncrementalSafetensorsWriter(path) as writer:
            writer.write_dict(payload)
        model = OffloadModel(
            chunks, device="cpu", window=2,
            uel_path=path, uel_key_map=key_map,
        )
        torch.testing.assert_close(model(x), expected)
        model.close()
    print("UEL async source smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
