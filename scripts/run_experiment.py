"""Evaluate a frozen config; tuning partition is the safe default."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from filament.data import Dataset
from filament.disk import detect_disk
from filament.evaluate import (
    Counts,
    aggregate_counts,
    bootstrap_pq,
    corroborated_false_positives,
    evaluate_pair,
    human_comparison,
    summarize,
)
from filament.experiment import provenance, run_predictions, write_json
from filament.masks import export_csv
from filament.pipeline import load_config
from filament.splits import load_split, split_fingerprint
from filament.viz import save_overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--partition", choices=["tune", "test", "train", "val"], default="tune")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default=None, help="Optional M0/M1 disk cache (affects timings)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--overlays", type=int, default=6)
    parser.add_argument("--finalize-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Optional trained U-Net checkpoint")
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    config = load_config(args.config)
    if not 0 < args.probability_threshold < 1:
        parser.error("--probability-threshold must be in (0, 1)")
    checkpoint_hash = (
        hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None
    )
    if args.checkpoint:
        import torch

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint["split_sha256"] != split_fingerprint(args.split):
            parser.error("Checkpoint belongs to a different split")
        if not set(checkpoint["train_ids"]) <= set(split["train"]):
            parser.error("Checkpoint has trained outside the training partition")
        if not set(checkpoint["val_ids"]) <= set(split["val"]):
            parser.error("Checkpoint used validation observations outside val")
    ids = split[args.partition]
    if args.partition == "test" and (not args.finalize_test or args.limit):
        parser.error("Test evaluation requires --finalize-test and cannot use --limit")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        ids = sorted(
            np.random.default_rng(args.seed).choice(ids, min(args.limit, len(ids)), replace=False)
        )
    if not ids:
        parser.error("Selected partition is empty")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty; previous experiments are immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {
        **provenance(dataset, args.split, config),
        "partition": args.partition,
        "image_ids": ids,
        "cache": args.cache,
        "workers": args.workers,
        "checkpoint_sha256": checkpoint_hash,
        "probability_threshold": args.probability_threshold if args.checkpoint else None,
        "timing_scope": "segmentation + RLE; excludes image IO and evaluation",
    }
    receipt = None
    if args.partition == "test":
        key = hashlib.sha256(
            json.dumps(
                [
                    dataset.sha256,
                    metadata["split_sha256"],
                    asdict(config),
                    checkpoint_hash,
                    metadata["probability_threshold"],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        receipt = Path("results/test_registry") / f"{key}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental repeated test-set evaluation.
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump({"status": "started", "output": str(args.out.resolve())}, handle)
    write_json(args.out / "provenance.json", metadata)
    if args.checkpoint:
        from filament.unet.inference import predict_dataset

        results = predict_dataset(
            dataset, ids, args.checkpoint, config, args.probability_threshold, args.device
        )
    else:
        results = run_predictions(dataset, ids, config, args.workers, args.cache)
    comparisons = [c for r in results for c in r["comparisons"]]
    image_counts = {r["id"]: Counts(**r["counts"]) for r in results}
    human_counts, method_counts, relaxed = {}, {}, []
    noise = {"false_positives": 0, "corroborated": 0}
    predictions = {}
    for result in results:
        observation = dataset.observations[result["id"]]
        truth = [a.rles(observation.shape) for a in observation.annotations]
        prediction = result.pop("predictions")
        predictions[observation.id] = prediction
        relaxed.extend(evaluate_pair(a, prediction, threshold=0.3) for a in truth)
        if len(truth) >= 2:
            comparison = human_comparison(truth, prediction)
            human_counts[observation.id] = Counts(**comparison["human"])
            method_counts[observation.id] = Counts(**comparison["method"])
            result["human_comparison"] = comparison
            corroboration = corroborated_false_positives(truth, prediction)
            for key in noise:
                noise[key] += corroboration[key]
        if len(predictions) <= args.overlays:
            image = dataset.read_image(observation.id)
            save_overlay(
                args.out / "overlays" / f"{observation.id}.png",
                image,
                truth[0],
                prediction,
                detect_disk(image, config.disk_inset).mask,
            )
    metrics = {
        **summarize(comparisons),
        "observations": len(ids),
        "pq_ci": bootstrap_pq(image_counts, samples=args.bootstrap, seed=args.seed),
        "iou_0_3": summarize(relaxed),
        "annotation_noise": noise,
        "timing_seconds": {
            "median": float(np.median([r["seconds"] for r in results])),
            "max": max(r["seconds"] for r in results),
        },
        "by_site": {
            site: summarize([c for r in results if r["site"] == site for c in r["comparisons"]])
            for site in sorted({r["site"] for r in results})
        },
    }
    if human_counts:
        metrics["same_reference"] = {
            "human": aggregate_counts(
                [{"counts": asdict(c)} for c in human_counts.values()]
            ).metrics(),
            "method": aggregate_counts(
                [{"counts": asdict(c)} for c in method_counts.values()]
            ).metrics(),
            "method_minus_human": bootstrap_pq(
                method_counts, human_counts, samples=args.bootstrap, seed=args.seed
            ),
        }
    write_json(args.out / "metrics.json", metrics)
    write_json(args.out / "per_image.json", results)
    write_json(args.out / "predictions.json", predictions)
    export_csv(args.out / "predictions.csv", predictions)
    if receipt:
        write_json(receipt, {"status": "complete", "output": str(args.out.resolve())})
    print(
        json.dumps(
            {key: metrics[key] for key in ("observations", "pq", "sq", "rq", "dice", "pq_ci")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
