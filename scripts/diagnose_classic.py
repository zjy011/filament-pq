"""Trace classical misses through pipeline stages on tuning observations only.

This is exploratory diagnosis, not a new held-out performance estimate.
Stage coverage means any overlapping pixel; it does not imply a PQ match.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from filament.data import Dataset
from filament.evaluate import area_bucket
from filament.experiment import provenance, write_json
from filament.masks import decode, iou_matrix, label_rles
from filament.pipeline import (
    PipelineConfig,
    candidates,
    clean_mask,
    load_config,
    postprocess,
    prepare,
)
from filament.splits import load_split

STAGES = ("disk", "candidate", "morphology", "area", "shape", "final")


def miss_stage(overlaps):
    """Locate the last stage with overlap, allowing morphology to restore pixels."""
    if overlaps["final"]:
        return None
    for name in reversed(STAGES[:-1]):
        if overlaps[name]:
            return {"disk": "candidate_extraction", "candidate": "morphology",
                    "morphology": "area_filter", "area": "shape_filter",
                    "shape": "instance_or_boundary"}[name]
    return "outside_disk"


def diagnose_job(job):
    image_path, observation, settings, cache = job
    cv2.setNumThreads(1)
    path = Path(image_path)
    if not path.exists():
        path = path.with_suffix(".jpg")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != observation.shape:
        raise ValueError(f"Invalid image: {path}")
    config = PipelineConfig(**settings)
    ratio, disk = prepare(image, config, cache)
    candidate = candidates(ratio, disk, config)
    masks = {
        "disk": disk,
        "candidate": candidate,
        "morphology": clean_mask(candidate, disk, replace(config, min_area=1, min_axis_ratio=1)),
        "area": clean_mask(candidate, disk, replace(config, min_axis_ratio=1)),
        "shape": clean_mask(candidate, disk, config),
    }
    labels = postprocess(candidate, disk, config)
    masks["final"] = labels > 0
    predictions = label_rles(labels)
    rows = []
    for annotation in observation.annotations:
        gt = annotation.rles(observation.shape)
        matrix = iou_matrix(gt, predictions)
        for index, rle in enumerate(gt):
            mask = decode(rle)
            area = int(mask.sum())
            overlap = {name: int(np.count_nonzero(mask & value)) for name, value in masks.items()}
            best = float(matrix[index].max()) if predictions else 0.0
            degree = int(np.count_nonzero(matrix[index] > 0))
            category = ("matched" if best > 0.5 else "miss" if degree == 0
                        else "fragmentation" if degree >= 2 else "localization")
            rows.append({
                "annotation_id": annotation.id,
                "instance_id": str(annotation.instances[index]["id"]),
                "area": area, "bucket": area_bucket(area),
                "category": category, "best_iou": best,
                "overlap_pixels": overlap,
                "miss_stage": miss_stage(overlap),
            })
    return {"id": observation.id, "site": observation.site, "instances": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--config", default="configs/frozen_classic/full.yaml")
    parser.add_argument("--cache", default="cache/full_tuning")
    parser.add_argument("--out", type=Path, default=Path("results/classic_diagnosis"))
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if min(args.sample_size, args.workers) < 1:
        parser.error("Sample size and workers must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose an empty output directory")
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    ids = sorted(np.random.default_rng(args.seed).choice(
        split["tune"], min(args.sample_size, len(split["tune"])), replace=False
    ).tolist())
    assert not set(ids).intersection(split["test"])
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "provenance.json", {
        **provenance(dataset, args.split, config), "partition": "tune", "ids": ids,
        "seed": args.seed, "purpose": "Exploratory stage diagnosis; no test evaluation",
    })
    jobs = [(str(dataset.images / dataset.observations[key].file_name),
             dataset.observations[key], asdict(config), args.cache) for key in ids]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(diagnose_job, jobs):
            results.append(row)
            write_json(args.out / "progress.json", {
                "status": "running", "images": len(results), "total": len(ids)
            })
    instances = [row for image in results for row in image["instances"]]
    summary = {
        "partition": "tune", "observations": len(results), "annotation_instances": len(instances),
        "categories": dict(Counter(row["category"] for row in instances)),
        "miss_stages": dict(Counter(row["miss_stage"] for row in instances if row["category"] == "miss")),
        "miss_stages_by_area": dict(Counter(
            f"{row['miss_stage']}/{row['bucket']}" for row in instances if row["category"] == "miss"
        )),
        "stage_any_overlap": {name: sum(row["overlap_pixels"][name] > 0 for row in instances)
                              for name in STAGES},
        "notes": ["Counts are annotation-instance occurrences, not unique physical filaments.",
                  "Any overlap is not an IoU > 0.5 match.",
                  "Stage attribution is descriptive, not a counterfactual ablation.",
                  "Tuning observations only; this is not a held-out result."],
    }
    write_json(args.out / "per_image.json", results)
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "progress.json", {"status": "complete", "images": len(results)})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
