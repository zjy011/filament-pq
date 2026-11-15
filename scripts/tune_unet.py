"""Tune probability threshold and shared M4-M6 using validation observations only."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from filament.data import Dataset
from filament.evaluate import aggregate_counts, evaluate_pair
from filament.experiment import provenance, write_json
from filament.masks import label_rles
from filament.pipeline import PipelineConfig, load_config, postprocess
from filament.splits import load_split, split_fingerprint


def score_job(job):
    path, observation, spec = job
    cv2.setNumThreads(1)
    with np.load(path) as saved:
        disk = saved["disk"]
        probability = cv2.resize(
            saved["probability"], disk.shape[::-1], interpolation=cv2.INTER_LINEAR
        )
    pred = label_rles(
        postprocess(probability >= spec["threshold"], disk, PipelineConfig(**spec["config"]))
    )
    comparisons = [
        {"annotation_id": a.id, **evaluate_pair(a.rles(observation.shape), pred)}
        for a in observation.annotations
    ]
    return {
        "id": observation.id,
        "comparisons": comparisons,
        "counts": asdict(aggregate_counts(comparisons)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.rounds) < 1:
        parser.error("Workers and rounds must be positive")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error("Choose a new output directory")
    import torch

    from filament.unet.data import prepare_input
    from filament.unet.model import build_model, select_device

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["dataset_sha256"] != dataset.sha256 or checkpoint[
        "split_sha256"
    ] != split_fingerprint(args.split):
        parser.error("Checkpoint belongs to another dataset/split")
    if not set(checkpoint["train_ids"]) <= set(split["train"]) or not set(
        checkpoint["val_ids"]
    ) <= set(split["val"]):
        parser.error("Checkpoint used data outside train/val")
    config = load_config(args.config)
    if config.disk_inset != checkpoint["postprocess"]["disk_inset"]:
        parser.error("Keep M0 identical to training")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "probabilities"
    cache.mkdir(exist_ok=True)
    manifest = {
        **provenance(dataset, args.split, config),
        "image_ids": split["val"],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "rounds": args.rounds,
        "partition": "val",
    }
    manifest = json.loads(json.dumps(manifest))
    if args.resume:
        previous = json.loads((args.out / "provenance.json").read_text(encoding="utf-8"))
        for key in (
            "dataset_sha256",
            "split_sha256",
            "config",
            "image_ids",
            "checkpoint_sha256",
            "rounds",
        ):
            if previous[key] != manifest[key]:
                parser.error(f"Resume differs in {key}")
    else:
        write_json(args.out / "provenance.json", manifest)
    device = select_device(args.device)
    model = build_model(checkpoint["encoder"], weights=None)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    for index, key in enumerate(split["val"], 1):
        path = cache / f"{key}.npz"
        if path.exists():
            continue
        small, disk = prepare_input(dataset.read_image(key), checkpoint["size"], config.disk_inset)
        with torch.inference_mode():
            probability = (
                model(torch.from_numpy(small[None, None]).to(device)).sigmoid()[0, 0].cpu().numpy()
            )
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, probability=probability, disk=disk)
        temporary.replace(path)
        write_json(
            args.out / "progress.json",
            {"status": "probabilities", "images": index, "total": len(split["val"])},
        )
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    current = {"threshold": 0.5, "config": asdict(config)}
    stages = [
        [{"threshold": t} for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)],
        [{"min_area": a} for a in (300, 600, 1000, 2000)],
        [{"min_axis_ratio": a} for a in (1, 2, 3)],
        [{"opening_radius": o, "closing_radius": c} for o in (0, 1, 2) for c in (0, 2, 4)],
        [{"instances": "components"}]
        + [
            {"instances": "merge", "merge_distance": d, "merge_angle": a}
            for d in (10, 20, 40)
            for a in (20, 40)
        ]
        + [{"instances": "watershed", "merge_distance": 20, "merge_angle": 20}],
        [{"dilation_radius": r} for r in (0, 1, 2, 3)],
    ]
    trials = {}
    if args.resume and (args.out / "trials.json").exists():
        for trial in json.loads((args.out / "trials.json").read_text(encoding="utf-8")):
            trials[json.dumps(trial["spec"], sort_keys=True)] = trial
    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        def score(spec):
            key = json.dumps(spec, sort_keys=True)
            if key not in trials:
                rows = list(
                    pool.map(
                        score_job,
                        [(cache / f"{i}.npz", dataset.observations[i], spec) for i in split["val"]],
                    )
                )
                trials[key] = {"spec": spec, "metrics": aggregate_counts(rows).metrics()}
                write_json(args.out / "trials.json", list(trials.values()))
                print(
                    f"Validation trial {len(trials)}: PQ={trials[key]['metrics']['pq']:.6f}",
                    flush=True,
                )
            return trials[key]["metrics"]["pq"]

        for round_number in range(args.rounds):
            for stage_number, stage in enumerate(stages):
                options = [current]
                for change in stage:
                    if "threshold" in change:
                        options.append({**current, "threshold": change["threshold"]})
                    else:
                        options.append({**current, "config": {**current["config"], **change}})
                current = max(options, key=score)
                write_json(
                    args.out / "progress.json",
                    {
                        "status": "searching",
                        "round": round_number + 1,
                        "stage": stage_number + 1,
                        "trials": len(trials),
                        "best_pq": score(current),
                    },
                )
        # M5a-only control: no morphology, area/shape filtering or boundary expansion.
        control = {
            **current,
            "config": {
                **current["config"],
                "opening_radius": 0,
                "closing_radius": 0,
                "min_area": 1,
                "min_axis_ratio": 1,
                "instances": "components",
                "dilation_radius": 0,
            },
        }
        score(control)
    for name, spec in (("best", current), ("components_only", control)):
        (args.out / f"{name}.yaml").write_text(
            yaml.safe_dump(spec["config"], sort_keys=False), encoding="utf-8"
        )
        write_json(args.out / f"{name}.json", trials[json.dumps(spec, sort_keys=True)])
    write_json(
        args.out / "progress.json",
        {
            "status": "complete",
            "trials": len(trials),
            "best_pq": trials[json.dumps(current, sort_keys=True)]["metrics"]["pq"],
        },
    )


if __name__ == "__main__":
    main()
