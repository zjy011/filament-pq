"""Create blinded crops and two blank reviewer CSVs for isolated false positives."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from filament.data import Dataset
from filament.experiment import write_json
from filament.masks import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--padding", type=int, default=60)
    args = parser.parse_args()
    if args.count < 1 or args.padding < 0:
        parser.error("Count must be positive and padding nonnegative")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new output directory")
    dataset = Dataset(args.annotations, args.images)
    provenance = json.loads((args.run / "provenance.json").read_text())
    if provenance["dataset_sha256"] != dataset.sha256:
        parser.error("Experiment belongs to another dataset")
    results = json.loads((args.run / "per_image.json").read_text())
    predictions = json.loads((args.run / "predictions.json").read_text())
    # An instance may be an FP against several annotators; sample it only once.
    candidates = sorted(
        {
            (r["id"], e["index"])
            for r in results
            for c in r["comparisons"]
            for e in c["errors"]
            if e["side"] == "fp" and e["category"] == "isolated"
        }
    )
    if not candidates:
        parser.error("No isolated false positives found")
    selected = np.random.default_rng(args.seed).choice(
        len(candidates), min(args.count, len(candidates)), replace=False
    )
    (args.out / "crops").mkdir(parents=True)
    key, rows = [], []
    for i, candidate in enumerate(selected):
        observation_id, instance = candidates[candidate]
        sample = f"sample_{i + 1:03d}"
        mask = decode(predictions[observation_id][instance])
        yy, xx = np.where(mask)
        image = dataset.read_image(observation_id)
        y1, y2 = max(0, yy.min() - args.padding), min(image.shape[0], yy.max() + args.padding + 1)
        x1, x2 = max(0, xx.min() - args.padding), min(image.shape[1], xx.max() + args.padding + 1)
        crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(
            mask[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(crop, contours, -1, (0, 255, 255), 1)
        success, encoded = cv2.imencode(".png", crop)
        if not success:
            raise OSError("Failed to encode crop")
        encoded.tofile(args.out / "crops" / f"{sample}.png")
        rows.append([sample, f"crops/{sample}.png", "", ""])
        key.append(
            {
                "sample_id": sample,
                "observation_id": observation_id,
                "prediction_index": int(instance),
            }
        )
    for reviewer in ("a", "b"):
        with (args.out / f"reviewer_{reviewer}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "crop", "label", "notes"])
            writer.writerows(rows)
    (args.out / "curator_only").mkdir()
    write_json(
        args.out / "curator_only" / "key.json",
        {"run": str(args.run), "seed": args.seed, "samples": key},
    )
    (args.out / "INSTRUCTIONS.txt").write_text(
        "Review independently. Do not inspect curator_only or the source experiment.\n"
        "Labels: possible_filament, sunspot, limb_artifact, other.\n"
        "Yellow outlines identify the candidate; no other annotator labels are shown.\n"
        "These non-expert labels support sensitivity analysis, not corrected ground truth.\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(rows)} unique candidates in {args.out}")


if __name__ == "__main__":
    main()
