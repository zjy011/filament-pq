"""Coordinate search on tune only, then re-rank top configurations on all tune images."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

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
    parser.add_argument("--resume", action="store_true")
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
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error("Choose a new output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        **provenance(dataset, args.split, current),
        "sample_ids": ids,
        "stages": stages,
        "rounds": args.rounds,
        "seed": args.seed,
        "top_k": args.top_k,
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = args.out / "provenance.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "dataset_sha256",
            "split_sha256",
            "config",
            "sample_ids",
            "stages",
            "rounds",
            "seed",
            "top_k",
        ):
            if previous[key] != manifest[key]:
                parser.error(f"Resume differs in {key}")
    else:
        write_json(manifest_path, manifest)
    trials = {}
    if args.resume and (args.out / "trials.json").exists():
        for trial in json.loads((args.out / "trials.json").read_text(encoding="utf-8")):
            trials[json.dumps(trial["config"], sort_keys=True)] = trial
    pool = ProcessPoolExecutor(max_workers=args.workers)

    def score(config):
        key = json.dumps(asdict(config), sort_keys=True)
        if key not in trials:
            started = perf_counter()
            predictions = run_predictions(dataset, ids, config, args.workers, args.cache, pool)
            metrics = aggregate_counts(predictions).metrics()
            trials[key] = {
                "config": asdict(config),
                "metrics": metrics,
                "seconds": perf_counter() - started,
            }
            write_json(args.out / "trials.json", list(trials.values()))
            print(f"Trial {len(trials)}: PQ={metrics['pq']:.6f}", flush=True)
        return trials[key]["metrics"]["pq"]

    for round_number in range(args.rounds):
        for stage in stages:
            write_json(
                args.out / "progress.json",
                {
                    "round": round_number + 1,
                    "stage": stage["name"],
                    "trials": len(trials),
                    "current": asdict(current),
                    "status": "searching",
                },
            )
            print(f"Round {round_number + 1}: {stage['name']}", flush=True)
            options = [current, *(replace(current, **changes) for changes in stage["options"])]
            current = max(options, key=score)
    finalists = sorted(trials.values(), key=lambda trial: trial["metrics"]["pq"], reverse=True)[
        : args.top_k
    ]
    from filament.pipeline import PipelineConfig

    for index, trial in enumerate(finalists):
        write_json(
            args.out / "progress.json",
            {
                "status": "full_tune",
                "finalist": index + 1,
                "finalists": len(finalists),
                "trials": len(trials),
            },
        )
        result_path = args.out / f"finalist_{index + 1}.json"
        if args.resume and result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved["config"] != trial["config"]:
                parser.error("Finalist ordering changed")
            trial["full_tune_metrics"] = saved["metrics"]
            continue
        full = run_predictions(
            dataset,
            split["tune"],
            PipelineConfig(**trial["config"]),
            args.workers,
            args.cache,
            pool,
        )
        trial["full_tune_metrics"] = aggregate_counts(full).metrics()
        write_json(
            result_path,
            {"config": trial["config"], "metrics": trial["full_tune_metrics"], "per_image": full},
        )
        write_json(args.out / "finalists.json", finalists)
        print(f"Finalist {index + 1}: PQ={trial['full_tune_metrics']['pq']:.6f}", flush=True)
    pool.shutdown()
    winner = max(finalists, key=lambda trial: trial["full_tune_metrics"]["pq"])
    (args.out / "best.yaml").write_text(
        yaml.safe_dump(winner["config"], sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(winner, indent=2))
    write_json(
        args.out / "progress.json",
        {"status": "complete", "trials": len(trials), "best_pq": winner["full_tune_metrics"]["pq"]},
    )


if __name__ == "__main__":
    main()
