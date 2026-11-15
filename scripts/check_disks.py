"""Sample tuning images per observatory and render M0 circles for visual QA."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from filament.data import Dataset
from filament.disk import detect_disk
from filament.experiment import write_json
from filament.pipeline import load_config
from filament.splits import load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--per-site", type=int, default=10)
    parser.add_argument("--seed", type=int, default=543)
    parser.add_argument("--out", type=Path, default=Path("results/disk_qa"))
    args = parser.parse_args()
    if args.per_site < 1:
        parser.error("--per-site must be positive")
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for site in sorted({o.site for o in dataset.observations.values()}):
        ids = [key for key in split["tune"] if dataset.observations[key].site == site]
        selected = rng.choice(ids, min(len(ids), args.per_site), replace=False)
        tiles = []
        for key in selected:
            image = dataset.read_image(key)
            disk = detect_disk(image, config.disk_inset)
            tile = cv2.resize(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), (400, 400))
            scale = 400 / image.shape[1]
            center = tuple(round(c * scale) for c in disk.center)
            cv2.circle(
                tile, center, round(disk.radius * (1 - config.disk_inset) * scale), (255, 255, 0), 1
            )
            cv2.rectangle(tile, (0, 0), (400, 23), (0, 0, 0), -1)
            cv2.putText(tile, key, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
            records.append({"id": key, "site": site, "center": disk.center, "radius": disk.radius})
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i : i + 5]) for i in range(0, len(tiles), 5)])
        success, encoded = cv2.imencode(".png", sheet)
        if not success:
            raise OSError("Failed to encode disk QA sheet")
        encoded.tofile(args.out / f"{site}.png")
    write_json(
        args.out / "circles.json",
        {"seed": args.seed, "disk_inset": config.disk_inset, "images": records},
    )
    print(f"Rendered {len(records)} tuning images from six observatories")


if __name__ == "__main__":
    main()
