"""Pair annotators once per observation; no image download is needed."""

import argparse
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

from filament.data import Dataset
from filament.evaluate import aggregate_counts, bootstrap_pq, evaluate_pair
from filament.experiment import write_json
from filament.splits import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--partition", choices=["tune", "all", "test"], default="tune")
    parser.add_argument("--out", type=Path, default=Path("results/inter_annotator.json"))
    args = parser.parse_args()
    dataset = Dataset(args.annotations)
    split = load_split(dataset, args.split)
    ids = list(dataset.observations) if args.partition == "all" else split[args.partition]
    per_image, pairs = {}, 0
    for key in ids:
        observation = dataset.observations[key]
        rles = [a.rles(observation.shape) for a in observation.annotations]
        if len(rles) >= 2:
            comparisons = [evaluate_pair(a, b) for a, b in combinations(rles, 2)]
            pairs += len(comparisons)
            per_image[key] = aggregate_counts(comparisons)
    if not per_image:
        parser.error("No multiply annotated observations selected")
    result = {
        "dataset_sha256": dataset.sha256,
        "partition": args.partition,
        "pairs": pairs,
        "metrics": aggregate_counts([{"counts": asdict(c)} for c in per_image.values()]).metrics(),
        "pq_ci": bootstrap_pq(per_image),
        "per_image": {key: asdict(c) for key, c in per_image.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(
        json.dumps(
            {"pairs": pairs, "metrics": result["metrics"], "pq_ci": result["pq_ci"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
