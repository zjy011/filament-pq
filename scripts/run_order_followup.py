"""One frozen exploratory test follow-up after the documented tuning gate passes.

The main test has already been inspected in the original study. This output is
explicitly exploratory and never overwrites the original held-out experiment.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from filament.data import Dataset
from filament.evaluate import Counts, aggregate_counts, bootstrap_pq, summarize
from filament.experiment import provenance, write_json
from filament.pipeline import load_config
from filament.splits import load_split
from scripts.classic_order_experiment import image_job

NAME = "late_area_300"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/followup/classic_late_area_300"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    analysis_path = Path("results/classic_order_analysis/summary.json")
    analysis = read(analysis_path)
    remainder = analysis["cohorts"]["remainder"]["variants"][NAME]["delta_vs_original"]
    combined = analysis["cohorts"]["combined_tune"]["variants"][NAME]["delta_vs_original"]
    if remainder["estimate"] <= 0 or combined["low"] <= 0:
        parser.error("The prospectively recorded tuning gate did not pass")
    dataset = Dataset("data/magfilo_2024_v1.0.json", "data/images")
    split_path = "splits/main.json"
    split = load_split(dataset, split_path)
    config = load_config("configs/frozen_classic/full.yaml")
    pilot = read("results/classic_order_pilot/provenance.json")
    rest = read("results/classic_order_remainder/provenance.json")
    current = provenance(dataset, split_path, config)
    impl_digest = digest("scripts/classic_order_experiment.py")
    if set(pilot["ids"]) & set(rest["ids"]) or set(pilot["ids"] + rest["ids"]) != set(split["tune"]):
        parser.error("Tuning cohort identity mismatch")
    for manifest in (pilot, rest):
        for key in ("dataset_sha256", "split_sha256", "evaluation_sha256", "config"):
            if manifest[key] != current[key]:
                parser.error(f"Frozen tuning implementation differs in {key}")
        if manifest["experiment_script_sha256"] != impl_digest:
            parser.error("Filtering-order implementation changed after tuning")
    recipe = pilot["variants"][NAME]
    if recipe != rest["variants"][NAME]:
        parser.error("Selected recipe differs between tuning cohorts")
    identity = {
        "dataset_sha256": current["dataset_sha256"], "split_sha256": current["split_sha256"],
        "evaluation_sha256": current["evaluation_sha256"], "config": asdict(config),
        "recipe": recipe, "experiment_script_sha256": impl_digest,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    registry = Path("results/followup_registry")
    registry.mkdir(exist_ok=True)
    receipt = registry / f"{run_id}.json"
    output = str(args.out.resolve())
    if receipt.exists():
        registered = read(receipt)
        if registered["output"] != output:
            parser.error("This frozen recipe is already registered at another output")
        if registered["status"] == "complete":
            print(f"Already complete; using saved result: {args.out / 'metrics.json'}")
            return
        if not args.resume:
            parser.error("An interrupted evaluation exists; use --resume")
    manifest = {
        **current, "identity": identity, "variant": NAME, "partition": "test", "ids": split["test"],
        "analysis_sha256": digest(analysis_path), "followup_script_sha256": digest(__file__),
        "selection_gate": {"remainder": remainder, "combined": combined},
        "interpretation": "Exploratory follow-up on the previously inspected main test; "
                          "not a prospectively blind confirmation or an equivalence test.",
    }
    if args.resume:
        previous = read(args.out / "provenance.json")
        for key in ("identity", "ids", "analysis_sha256", "followup_script_sha256"):
            if previous[key] != manifest[key]:
                parser.error(f"Resume differs in {key}")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose an empty output directory")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "provenance.json", {
            **manifest, "frozen_before_test_utc": datetime.now(timezone.utc).isoformat()
        })
    write_json(receipt, {"status": "running", "output": output, "identity": identity})
    raw = args.out / "raw_images"
    raw.mkdir(exist_ok=True)
    pending = [key for key in split["test"] if not (raw / f"{key}.json").exists()]
    jobs = [(str(dataset.images / dataset.observations[key].file_name), dataset.observations[key],
             config, {NAME: recipe}, "cache/full_tuning") for key in pending]
    done = len(split["test"]) - len(pending)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(image_job, jobs):
            write_json(raw / f"{row['id']}.json", row)
            done += 1
            write_json(args.out / "progress.json", {"status": "running", "images": done,
                                                    "total": len(split["test"])})
            if done % 25 == 0:
                print(f"Exploratory test: {done}/{len(split['test'])}", flush=True)
    rows = [read(raw / f"{key}.json") for key in split["test"]]
    counts = {r["id"]: aggregate_counts(r["variants"][NAME]["comparisons"]) for r in rows}
    metrics = summarize([c for r in rows for c in r["variants"][NAME]["comparisons"]])
    metrics.update({"observations": len(rows), "pq_ci": bootstrap_pq(counts),
                    "interpretation": manifest["interpretation"], "comparisons_to_saved": {}})
    for name, path in (("original", "results/final/classic/full/test/per_image.json"),
                       ("unet_512", "results/final/unet_e7a/per_image.json")):
        other = {r["id"]: Counts(**r["counts"]) for r in read(path)}
        metrics["comparisons_to_saved"][name] = {
            **bootstrap_pq(counts, other), "source": path, "source_sha256": digest(path)
        }
    write_json(args.out / "metrics.json", metrics)
    write_json(args.out / "progress.json", {"status": "complete", "images": len(rows)})
    write_json(receipt, {"status": "complete", "output": output, "identity": identity})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
