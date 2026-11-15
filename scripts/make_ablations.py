"""Write E2 configurations; each changes only one stage of the supplied method."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from filament.pipeline import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, default=Path("configs/ablations"))
    args = parser.parse_args()
    config = load_config(args.config)
    variants = {
        "full": config,
        "no_background_correction": replace(config, background="none"),
        "single_threshold": replace(config, threshold_method="single"),
        "no_shape_filter": replace(config, min_axis_ratio=1),
        "connected_components": replace(config, instances="components"),
        "no_boundary_expansion": replace(config, dilation_radius=0),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    for name, variant in variants.items():
        path = args.out / f"{name}.yaml"
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(variant), handle, sort_keys=False)
    print(f"Wrote {len(variants)} configurations to {args.out}")


if __name__ == "__main__":
    main()
