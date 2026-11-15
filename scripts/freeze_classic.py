"""Freeze E1/E2/E3 configurations using tuning results, before accessing test."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from filament.experiment import write_json
from filament.pipeline import load_config


def parameter_ranges(trials, config):
    grid = {
        "disk_inset": [0.02, 0.03, 0.05],
        "background_kernel": [15, 31, 63],
        "gaussian_sigma": [0, 1, 2],
        "threshold": [0.85, 0.88, 0.90, 0.92, 0.93, 0.96],
        "grow_threshold": [0.91, 0.93, 0.94, 0.96, 0.98],
        "min_area": [300, 600, 1000, 2000],
        "min_axis_ratio": [1, 2, 3],
        "merge_distance": [10, 20, 40],
        "dilation_radius": [0, 1, 2, 3],
    }
    if config.threshold_method == "single":
        grid.pop("grow_threshold")
    else:
        grid.pop("threshold")
    if config.instances == "components":
        grid.pop("merge_distance")
    if config.background != "median":
        grid.pop("background_kernel")
    impacts = {}
    for parameter in grid:
        groups = {}
        for row in trials:
            setting = row["config"]
            if parameter == "threshold" and setting["threshold_method"] != "single":
                continue
            if parameter == "grow_threshold" and setting["threshold_method"] == "single":
                continue
            if parameter == "merge_distance" and setting["instances"] == "components":
                continue
            key = json.dumps({k: v for k, v in setting.items() if k != parameter}, sort_keys=True)
            groups.setdefault(key, []).append(row["metrics"]["pq"])
        impacts[parameter] = max(
            (max(v) - min(v) for v in groups.values() if len(v) > 1), default=0
        )
    chosen = sorted(impacts, key=lambda k: (-impacts[k], k))[:3]
    return {key: sorted(set(grid[key] + [getattr(config, key)])) for key in chosen}, impacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, default=Path("results/full_tuning"))
    parser.add_argument("--out", type=Path, default=Path("configs/frozen_classic"))
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Frozen configurations cannot be overwritten")
    progress = json.loads((args.tuning / "progress.json").read_text(encoding="utf-8"))
    if progress["status"] != "complete":
        parser.error("Finish full tuning first")
    full = load_config(args.tuning / "best.yaml")
    trials = json.loads((args.tuning / "trials.json").read_text(encoding="utf-8"))
    grid, impacts = parameter_ranges(trials, full)
    configs = {
        "baseline": load_config("configs/baseline.yaml"),
        "full": full,
        "no_background_correction": replace(full, background="none"),
        "single_threshold": replace(full, threshold_method="single"),
        "no_shape_filter": replace(full, min_axis_ratio=1),
        "connected_components": replace(full, instances="components"),
        "no_boundary_expansion": replace(full, dilation_radius=0),
    }
    sensitivity = {}
    for parameter, values in grid.items():
        sensitivity[parameter] = []
        for value in values:
            name = f"{parameter}_{value}"
            configs[name] = replace(full, **{parameter: value})
            sensitivity[parameter].append({"value": value, "name": name})
    args.out.mkdir(parents=True)
    hashes, aliases, digests = {}, {}, {}
    for name, config in configs.items():
        key = json.dumps(asdict(config), sort_keys=True)
        if key in hashes:
            aliases[name] = hashes[key]
            continue
        hashes[key] = name
        path = args.out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        args.out / "manifest.json",
        {
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "selection_partition": "tune",
            "source_tuning": str(args.tuning),
            "tuning_provenance_sha256": hashlib.sha256(
                (args.tuning / "provenance.json").read_bytes()
            ).hexdigest(),
            "config_sha256": digests,
            "aliases": aliases,
            "sensitivity": sensitivity,
            "sample_tune_pq_ranges": impacts,
            "rule": "Largest observed single-parameter tune PQ range; test curves cannot change the selected full model.",
        },
    )
    print(
        json.dumps(
            {"configurations": len(digests), "aliases": aliases, "sensitivity": sensitivity},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
