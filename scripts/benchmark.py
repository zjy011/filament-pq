"""CPU latency on a fixed tune sample; no scoring or parameter selection."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np

from filament.data import Dataset
from filament.experiment import provenance, write_json
from filament.masks import label_rles
from filament.pipeline import load_config, segment
from filament.splits import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    dataset = Dataset("data/magfilo_2024_v1.0.json", "data/images")
    split = load_split(dataset, "splits/main.json")
    ids = sorted(
        np.random.default_rng(args.seed).choice(
            split["tune"], min(args.count, len(split["tune"])), replace=False
        )
    )
    config = load_config(args.config)
    if args.checkpoint:
        import torch

        from filament.unet.inference import predict_image
        from filament.unet.model import build_model

        torch.set_num_threads(1)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = build_model(checkpoint["encoder"], weights=None)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        def predict(image):
            return predict_image(
                model,
                image,
                checkpoint["size"],
                config,
                args.probability_threshold,
                torch.device("cpu"),
            )
    else:

        def predict(image):
            return label_rles(segment(image, config, cache=None))

    for key in ids[:3]:
        predict(dataset.read_image(key))
    timing = []
    for key in ids:
        image = dataset.read_image(key)
        start = perf_counter()
        predict(image)
        timing.append({"id": key, "seconds": perf_counter() - start})
    result = {
        **provenance(dataset, "splits/main.json", config),
        "checkpoint": str(args.checkpoint),
        "processor": platform.processor(),
        "threads": 1,
        "cache": None,
        "scope": "segmentation including M0 and RLE; image IO, evaluation and warmup excluded",
        "per_image": timing,
        "median": float(np.median([r["seconds"] for r in timing])),
        "max": max(r["seconds"] for r in timing),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ("median", "max", "threads")}, indent=2))


if __name__ == "__main__":
    main()
