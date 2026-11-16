"""Summarize completed tuning-only filter-order experiments and paired contrasts."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from filament.evaluate import aggregate_counts, bootstrap_pq, summarize
from filament.experiment import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_records(records, names):
    counts = {name: {r["id"]: aggregate_counts(r["variants"][name]["comparisons"])
                     for r in records} for name in names}
    result = {"observations": len(records), "variants": {}, "contrasts": {}}
    for name in names:
        gained, lost = Counter(), Counter()
        for row in records:
            old_comparisons = row["variants"]["original"]["comparisons"]
            new_comparisons = row["variants"][name]["comparisons"]
            for old, new in zip(old_comparisons, new_comparisons, strict=True):
                if old["annotation_id"] != new["annotation_id"]:
                    raise ValueError("Annotation comparison order differs")
                old_fn = {e["index"]: e["category"] for e in old["errors"] if e["side"] == "fn"}
                new_fn = {e["index"]: e["category"] for e in new["errors"] if e["side"] == "fn"}
                gained.update(old_fn[i] for i in old_fn.keys() - new_fn.keys())
                lost.update(new_fn[i] for i in new_fn.keys() - old_fn.keys())
        result["variants"][name] = {
            **summarize([c for r in records for c in r["variants"][name]["comparisons"]]),
            "delta_vs_original": bootstrap_pq(counts[name], counts["original"]),
            "matched_instance_transitions": {
                "gained_from_original_fn": dict(gained), "lost_to_new_fn": dict(lost),
                "gained_total": sum(gained.values()), "lost_total": sum(lost.values()),
            },
        }
    contrasts = [("original_no_shape", "original")]
    for floor in (150, 300):
        contrasts.extend([
            (f"late_area_{floor}", "original"),
            (f"late_area_{floor}", f"relax_{floor}"),
            (f"late_both_{floor}", f"late_area_{floor}"),
            (f"late_area_no_shape_{floor}", f"late_area_{floor}"),
            (f"late_both_{floor}", f"late_area_no_shape_{floor}"),
            (f"late_area_no_shape_{floor}", "original_no_shape"),
        ])
    for first, second in contrasts:
        if first in counts and second in counts:
            result["contrasts"][f"{first} minus {second}"] = bootstrap_pq(counts[first], counts[second])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/classic_order_analysis"))
    args = parser.parse_args()
    cohorts, all_records, manifests = {}, [], []
    seen = set()
    for run in args.runs:
        if read(run / "progress.json")["status"] != "complete":
            parser.error(f"Run is not complete: {run}")
        manifest = read(run / "provenance.json")
        if manifests:
            for key in ("dataset_sha256", "split_sha256", "evaluation_sha256", "config",
                        "experiment_script_sha256"):
                if manifest[key] != manifests[0][key]:
                    parser.error(f"Cohorts differ in {key}")
        if seen.intersection(manifest["ids"]):
            parser.error("Cohorts overlap")
        if manifest["cohort"] in cohorts:
            parser.error("Duplicate cohort name")
        seen.update(manifest["ids"])
        records = [read(run / "raw_images" / f"{key}.json") for key in manifest["ids"]]
        if [r["id"] for r in records] != manifest["ids"]:
            parser.error("Per-image IDs differ from manifest")
        cohorts[manifest["cohort"]] = summarize_records(records, manifest["variants"])
        manifests.append(manifest)
        all_records.extend(records)
    if len(cohorts) > 1:
        common = set.intersection(*(set(m["variants"]) for m in manifests))
        for name in common:
            if any(m["variants"][name] != manifests[0]["variants"][name] for m in manifests):
                parser.error(f"Recipe changed between cohorts: {name}")
        cohorts["combined_tune"] = summarize_records(all_records, sorted(common))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", {
        "runs": [str(r) for r in args.runs], "cohorts": cohorts,
        "interpretation": "Exploratory tuning-only scores. Pilot-guided selection; pointwise "
                          "paired bootstrap intervals do not adjust for selection or multiplicity. "
                          "No new held-out test evaluation; no U-Net equivalence claim.",
    })
    fig, axes = plt.subplots(1, len(cohorts), figsize=(6 * len(cohorts), 6), squeeze=False)
    for ax, (cohort, data) in zip(axes[0], cohorts.items()):
        items = sorted(data["variants"].items(), key=lambda x: x[1]["pq"])
        y = np.arange(len(items))
        ax.barh(y, [m["pq"] for _, m in items],
                color=["#667785" if n == "original" else "#157a8a" for n, _ in items])
        ax.set_yticks(y, [n for n, _ in items], fontsize=8)
        ax.set(xlabel="Panoptic quality (PQ)", title=f"{cohort}: {data['observations']} tuning images")
        ax.set_xlim(0, max(m["pq"] for _, m in items) * 1.2)
        for i, (_, metrics) in enumerate(items):
            ax.text(metrics["pq"] + 0.002, i, f"{metrics['pq']:.4f}", va="center", fontsize=8)
    fig.suptitle("Classical filtering order: exploratory tuning results", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.text(0.02, 0.015, "Fixed candidate extraction and merging rules.\n"
             "Source: saved per-image experiment records; no test images.", fontsize=8)
    fig.savefig(args.out / "comparison.png", dpi=150)
    plt.close(fig)
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
