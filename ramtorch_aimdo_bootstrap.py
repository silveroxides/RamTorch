"""Console launcher that initializes AIMDO before the target imports Torch."""

from __future__ import annotations

import argparse
import runpy
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ramtorch-aimdo",
        description="Initialize AIMDO, then execute a Python script.",
    )
    parser.add_argument("--devices", required=True,
                        help="comma-separated CUDA/ROCm device indices")
    parser.add_argument("--headroom-mb", type=int, default=0,
                        help="extra VRAM headroom AIMDO must leave free")
    parser.add_argument("script", help="target Python script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    try:
        import comfy_aimdo.control as control
    except ImportError as exc:
        parser.error("comfy-aimdo is required for this launcher: %s" % exc)
    device_ids = [int(value) for value in args.devices.split(",") if value]
    headroom = args.headroom_mb * 1024 * 1024
    if not control.init() or not control.init_devices(
        [(device_id, headroom) for device_id in device_ids]
    ):
        parser.error("AIMDO initialization failed")

    sys.argv = [args.script, *args.script_args]
    runpy.run_path(args.script, run_name="__main__")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
