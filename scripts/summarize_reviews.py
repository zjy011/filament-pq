"""Validate two completed blind reviews and report agreement and FP sensitivity."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from filament.experiment import write_json

LABELS = {"possible_filament", "sunspot", "limb_artifact", "other"}


def load_reviews(path, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected) or {r["sample_id"] for r in rows} != set(expected):
        raise ValueError("Review must contain exactly the expected IDs, without duplicates")
    if any(r["label"] not in LABELS for r in rows):
        raise ValueError("Complete every label using the four permitted categories")
    return {r["sample_id"]: r["label"] for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = json.loads((args.package / "curator_only/key.json").read_text(encoding="utf-8"))
    samples = {row["sample_id"]: row for row in key["samples"]}
    a, b = load_reviews(args.reviewer_a, samples), load_reviews(args.reviewer_b, samples)
    n = len(samples)
    counts_a, counts_b = Counter(a.values()), Counter(b.values())
    agreement = sum(a[i] == b[i] for i in samples) / n
    expected = sum(counts_a[c] * counts_b[c] for c in LABELS) / n**2
    metrics = json.loads((Path(key["run"]) / "metrics.json").read_text(encoding="utf-8"))
    denominator = metrics["tp"] + 0.5 * (metrics["fp"] + metrics["fn"])
    sensitivity = {}
    for name, rule in (("both_reviewers", all), ("either_reviewer", any)):
        # Horvitz-Thompson expansion for a simple random sample of UNIQUE instances.
        sampled = sum(
            row["isolated_fp_occurrences"]
            for i, row in samples.items()
            if rule([a[i] == "possible_filament", b[i] == "possible_filament"])
        )
        estimated = min(
            key["population_isolated_fp_occurrences"], sampled * key["population_instances"] / n
        )
        reduced_denominator = denominator - 0.5 * estimated
        sensitivity[name] = {
            "estimated_ignored_fp_occurrences": estimated,
            "hypothetical_pq": metrics["sum_iou"] / reduced_denominator
            if reduced_denominator
            else 0,
        }
    result = {
        "samples": n,
        "reviewer_a_counts": dict(counts_a),
        "reviewer_b_counts": dict(counts_b),
        "agreement": agreement,
        "cohen_kappa": (agreement - expected) / (1 - expected) if expected < 1 else None,
        "original_pq": metrics["pq"],
        "sensitivity": sensitivity,
        "interpretation": "Non-expert FP-exclusion scenarios, not confidence bounds or corrected ground truth. "
        "TP/FN and IoU sums stay fixed. Sampling uncertainty is not included.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
