"""Coordinate search on tune only, then re-rank top configurations on all tune images."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from filament.data import Dataset
from filament.evaluate import aggregate_counts
from filament.experiment import provenance, run_predictions, write_json
from filament.pipeline import load_config
from filament.splits import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--search", default="configs/search.yaml")
    parser.add_argument("--out", type=Path, default=Path("results/tuning"))
    parser.add_argument("--cache", default="cache/preprocess")
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=543)
    args = parser.parse_args()
    if min(args.sample_size, args.rounds, args.top_k, args.workers) < 1:
        parser.error("Search sizes, rounds, and workers must be positive")
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    current = load_config(args.config)
    stages = yaml.safe_load(Path(args.search).read_text(encoding="utf-8"))["stages"]
    ids = sorted(
        np.random.default_rng(args.seed).choice(
            split["tune"], min(args.sample_size, len(split["tune"])), replace=False
        )
    )
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "provenance.json",
        {
            **provenance(dataset, args.split, current),
            "sample_ids": ids,
            "stages": stages,
            "rounds": args.rounds,
            "seed": args.seed,
        },
    )
    trials = {}

    def score(config):
        key = json.dumps(asdict(config), sort_keys=True)
        if key not in trials:
            predictions = run_predictions(dataset, ids, config, args.workers, args.cache)
            metrics = aggregate_counts(predictions).metrics()
            trials[key] = {"config": asdict(config), "metrics": metrics}
            write_json(args.out / "trials.json", list(trials.values()))
            print(f"Trial {len(trials)}: PQ={metrics['pq']:.6f}", flush=True)
        return trials[key]["metrics"]["pq"]

    for _ in range(args.rounds):
        for stage in stages:
            options = [current, *(replace(current, **changes) for changes in stage["options"])]
            current = max(options, key=score)
    finalists = sorted(trials.values(), key=lambda trial: trial["metrics"]["pq"], reverse=True)[
        : args.top_k
    ]
    from filament.pipeline import PipelineConfig

    for trial in finalists:
        full = run_predictions(
            dataset, split["tune"], PipelineConfig(**trial["config"]), args.workers, args.cache
        )
        trial["full_tune_metrics"] = aggregate_counts(full).metrics()
        write_json(args.out / "finalists.json", finalists)
    winner = max(finalists, key=lambda trial: trial["full_tune_metrics"]["pq"])
    (args.out / "best.yaml").write_text(
        yaml.safe_dump(winner["config"], sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
