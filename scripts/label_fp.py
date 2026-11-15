"""Create blinded crops and two blank reviewer CSVs for isolated false positives."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

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
    occurrences = Counter(
        (r["id"], e["index"])
        for r in results
        for c in r["comparisons"]
        for e in c["errors"]
        if e["side"] == "fp" and e["category"] == "isolated"
    )
    candidates = sorted(occurrences)
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
        raw_crop = crop.copy()
        contours, _ = cv2.findContours(
            mask[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(crop, contours, -1, (0, 255, 255), 1)
        success, encoded = cv2.imencode(".png", np.concatenate([raw_crop, crop], axis=1))
        if not success:
            raise OSError("Failed to encode crop")
        encoded.tofile(args.out / "crops" / f"{sample}.png")
        rows.append([sample, f"crops/{sample}.png", "", ""])
        key.append(
            {
                "sample_id": sample,
                "observation_id": observation_id,
                "prediction_index": int(instance),
                "isolated_fp_occurrences": occurrences[(observation_id, instance)],
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
        {
            "run": str(args.run),
            "seed": args.seed,
            "samples": key,
            "population_instances": len(candidates),
            "population_isolated_fp_occurrences": sum(occurrences.values()),
        },
    )
    (args.out / "INSTRUCTIONS.txt").write_text(
        "Review independently. Do not inspect curator_only or the source experiment.\n"
        "Labels: possible_filament, sunspot, limb_artifact, other.\n"
        "Left: original crop. Right: yellow outline identifies the candidate.\n"
        "No other annotator labels are shown. Fill every row; leave notes when uncertain.\n"
        "Unzip your reviewer package, open index.html, label all samples, then export CSV.\n"
        "Alternatively, edit your blank CSV directly. Do not discuss labels before both exports.\n"
        "possible_filament: elongated dark solar structure plausibly missed by an annotator.\n"
        "sunspot: compact dark spot; limb_artifact: boundary/ring artifact; other: other/noise.\n"
        "These non-expert labels support sensitivity analysis, not corrected ground truth.\n",
        encoding="utf-8",
    )
    from scripts.review_page import make_review_page

    for reviewer in ("a", "b"):
        with ZipFile(args.out / f"reviewer_{reviewer}.zip", "w", ZIP_DEFLATED) as bundle:
            bundle.write(args.out / f"reviewer_{reviewer}.csv", f"reviewer_{reviewer}.csv")
            bundle.write(args.out / "INSTRUCTIONS.txt", "INSTRUCTIONS.txt")
            bundle.writestr("index.html", make_review_page(rows, reviewer))
            bundle.writestr(
                "DATA_CREDIT.txt",
                "Crops from MAGFiLO v1.0 / NSO GONG H-alpha.\n"
                "CC BY-NC 4.0; https://doi.org/10.7910/DVN/J6JNVK\n",
            )
            for row in rows:
                bundle.write(args.out / row[1], row[1])
    print(f"Prepared {len(rows)} unique candidates in {args.out}")


if __name__ == "__main__":
    main()
