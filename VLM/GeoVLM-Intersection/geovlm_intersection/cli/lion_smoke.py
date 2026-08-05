from __future__ import annotations

import argparse

from geovlm_intersection.backbones import load_lion_runtime
from geovlm_intersection.config.common import DEFAULT_LION_QUALITY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the OpenPCDet LION runtime adapter.")
    parser.add_argument("--quality", choices=["low", "mid", "high"], default=DEFAULT_LION_QUALITY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_lion_runtime(args.quality)
    print(f"config_path={runtime.config_path}")
    print(f"checkpoint_path={runtime.checkpoint_path}")
    print(f"config_name={runtime.config_name}")
    print(f"backbone_class={runtime.backbone_class.__name__}")
    print(f"detector_class={runtime.detector_class.__name__}")


if __name__ == "__main__":
    main()
