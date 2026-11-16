"""Bounded, tuning-only experiment on area/shape filtering before or after merging."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import expand_labels

from filament.data import Dataset
from filament.evaluate import aggregate_counts, bootstrap_pq, evaluate_pair, summarize
from filament.experiment import provenance, write_json
from filament.masks import label_rles
from filament.pipeline import candidates, clean_mask, load_config, merge_fragments, prepare
from filament.splits import load_split


def recipes(config):
    result = {
        "original": {"pre_area": config.min_area, "pre_shape": config.min_axis_ratio,
                     "post_area": 1, "post_shape": 1},
        "original_no_shape": {"pre_area": config.min_area, "pre_shape": 1,
                              "post_area": 1, "post_shape": 1},
    }
    for floor in (150, 300):
        result.update({
            f"relax_{floor}": {"pre_area": floor, "pre_shape": config.min_axis_ratio,
                               "post_area": 1, "post_shape": 1},
            f"relax_no_shape_{floor}": {"pre_area": floor, "pre_shape": 1,
                                        "post_area": 1, "post_shape": 1},
            f"late_area_{floor}": {"pre_area": floor, "pre_shape": config.min_axis_ratio,
                                   "post_area": config.min_area, "post_shape": 1},
            f"late_both_{floor}": {"pre_area": floor, "pre_shape": 1,
                                   "post_area": config.min_area,
                                   "post_shape": config.min_axis_ratio},
            f"late_area_no_shape_{floor}": {"pre_area": floor, "pre_shape": 1,
                                            "post_area": config.min_area, "post_shape": 1},
        })
    return result


def filter_groups(labels, min_area, min_shape):
    keep = np.bincount(labels.ravel()) >= min_area
    keep[0] = False
    if min_shape > 1:
        for region in regionprops(labels):
            if region.axis_major_length / max(region.axis_minor_length, 1e-6) < min_shape:
                keep[region.label] = False
    return np.where(keep[labels], labels, 0).astype(np.int32)


def variant_masks(candidate, disk, config, selected):
    """Reuse identical pre-filter/merge calculations; no truth enters segmentation."""
    if config.instances != "merge":
        raise ValueError("This controlled experiment requires the merge recipe")
    grouped = {}
    for name, spec in selected.items():
        key = (spec["pre_area"], spec["pre_shape"])
        if key not in grouped:
            clean = clean_mask(candidate, disk, replace(
                config, min_area=spec["pre_area"], min_axis_ratio=spec["pre_shape"]
            ))
            labels, _ = ndi.label(clean, structure=np.ones((3, 3)))
            grouped[key] = (merge_fragments(labels, config.merge_distance, config.merge_angle)
                            if labels.max() else labels)
        labels = filter_groups(grouped[key], spec["post_area"], spec["post_shape"])
        if config.dilation_radius:
            labels = expand_labels(labels, distance=config.dilation_radius)
        labels[~disk] = 0
        yield name, labels.astype(np.int32)


def image_job(job):
    image_path, observation, config, selected, cache = job
    cv2.setNumThreads(1)
    path = Path(image_path)
    if not path.exists():
        path = path.with_suffix(".jpg")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != observation.shape:
        raise ValueError(f"Invalid image: {path}")
    ratio, disk = prepare(image, config, cache)
    candidate = candidates(ratio, disk, config)
    truths = [(a.id, a.rles(observation.shape)) for a in observation.annotations]
    result = {"id": observation.id, "site": observation.site, "variants": {}}
    for name, labels in variant_masks(candidate, disk, config, selected):
        pred = label_rles(labels)
        comparisons = [{"annotation_id": aid, **evaluate_pair(gt, pred)} for aid, gt in truths]
        result["variants"][name] = {"comparisons": comparisons,
                                      "counts": asdict(aggregate_counts(comparisons))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--config", default="configs/frozen_classic/full.yaml")
    parser.add_argument("--cache", default="cache/full_tuning")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", choices=["pilot", "remainder"], default="pilot")
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.sample_size, args.workers) < 1:
        parser.error("Sample size and workers must be positive")
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    pilot = set(np.random.default_rng(args.seed).choice(
        split["tune"], min(args.sample_size, len(split["tune"])), replace=False
    ).tolist())
    ids = sorted(pilot if args.cohort == "pilot" else set(split["tune"]) - pilot)
    if not ids:
        parser.error("Empty evaluation cohort")
    assert not set(ids).intersection(split["test"])
    config = load_config(args.config)
    selected = recipes(config)
    if args.variants:
        if set(args.variants) - selected.keys():
            parser.error("Unknown variant")
        selected = {name: spec for name, spec in selected.items()
                    if name in {*args.variants, "original"}}
    manifest = {
        **provenance(dataset, args.split, config), "partition": "tune", "cohort": args.cohort,
        "ids": ids, "seed": args.seed, "sample_size": args.sample_size, "variants": selected,
        "experiment_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Exploratory tuning-only comparison; remainder is not an untouched test set.",
    }
    if args.resume:
        previous = json.loads((args.out / "provenance.json").read_text(encoding="utf-8"))
        for key in ("dataset_sha256", "split_sha256", "evaluation_sha256", "config", "ids",
                    "variants", "experiment_script_sha256"):
            if previous[key] != manifest[key]:
                parser.error(f"Resume differs in {key}")
    elif args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose an empty output directory or --resume")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "provenance.json", {
            **manifest, "frozen_before_evaluation_utc": datetime.now(timezone.utc).isoformat()
        })
    raw = args.out / "raw_images"
    raw.mkdir(exist_ok=True)
    pending = [key for key in ids if not (raw / f"{key}.json").exists()]
    jobs = [(str(dataset.images / dataset.observations[key].file_name), dataset.observations[key],
             config, selected, args.cache) for key in pending]
    done = len(ids) - len(pending)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(image_job, jobs):
            write_json(raw / f"{row['id']}.json", row)
            done += 1
            write_json(args.out / "progress.json", {
                "status": "running", "images": done, "total": len(ids), "variants": len(selected)
            })
            if done % 10 == 0:
                print(f"{done}/{len(ids)} images; {len(selected)} variants", flush=True)
    records = [json.loads((raw / f"{key}.json").read_text(encoding="utf-8")) for key in ids]
    counts = {name: {r["id"]: aggregate_counts(r["variants"][name]["comparisons"])
                     for r in records} for name in selected}
    summary = {"partition": "tune", "cohort": args.cohort, "observations": len(ids), "variants": {}}
    for name in selected:
        metrics = summarize([c for r in records for c in r["variants"][name]["comparisons"]])
        summary["variants"][name] = {
            **metrics, "delta_vs_original": bootstrap_pq(counts[name], counts["original"])
        }
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "progress.json", {"status": "complete", "images": len(ids)})
    print(json.dumps({name: {k: m[k] for k in ("pq", "tp", "fp", "fn", "delta_vs_original")}
                      for name, m in summary["variants"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
