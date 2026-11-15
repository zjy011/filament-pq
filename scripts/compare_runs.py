"""Paired image-level bootstrap of PQ(first) - PQ(second)."""

import argparse
import json
from pathlib import Path

from filament.evaluate import Counts, bootstrap_pq
from filament.experiment import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=543)
    args = parser.parse_args()
    manifests = [
        json.loads((path / "provenance.json").read_text()) for path in (args.first, args.second)
    ]
    for key in ("dataset_sha256", "split_sha256", "partition"):
        if manifests[0][key] != manifests[1][key]:
            parser.error(f"Runs differ in {key}")
    counts = [
        {
            row["id"]: Counts(**row["counts"])
            for row in json.loads((path / "per_image.json").read_text())
        }
        for path in (args.first, args.second)
    ]
    result = {
        "first": str(args.first),
        "second": str(args.second),
        "pq_difference": bootstrap_pq(*counts, samples=args.samples, seed=args.seed),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
