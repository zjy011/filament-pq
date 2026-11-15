"""Download public annotations/images, verify images, and freeze grouped splits."""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from filament.data import Dataset
from filament.splits import make_split

ANNOTATION_URL = "https://dataverse.harvard.edu/api/access/datafile/10407987"
ANNOTATION_SHA256 = "6f0cbe45e12d8e7a49847664fe9c9ab9d8ddd4efd9019af907af8e01db585bb6"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "filament-pq/0.1"})
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            temporary.replace(destination)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fetch_image(observation, root: Path) -> dict:
    path = root / observation.file_name
    if not path.exists():
        download(observation.url, path)
    content = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != observation.shape:
        raise ValueError(f"Invalid image: {path}; remove it and retry")
    return {
        "id": observation.id,
        "file_name": observation.file_name,
        "url": observation.url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/magfilo_2024_v1.0.json"))
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=543)
    args = parser.parse_args()
    if not args.annotations.exists():
        download(ANNOTATION_URL, args.annotations)
    dataset = Dataset(args.annotations, args.images)
    if args.annotations.name == "magfilo_2024_v1.0.json" and dataset.sha256 != ANNOTATION_SHA256:
        raise ValueError("MAGFiLO annotation checksum differs from the verified v1.0 release")
    print(json.dumps(dataset.summary(), indent=2), flush=True)
    args.splits.mkdir(parents=True, exist_ok=True)
    for site in [None, *sorted({o.site for o in dataset.observations.values()})]:
        path = args.splits / (f"site_{site}.json" if site else "main.json")
        split = make_split(dataset, args.seed, site)
        if path.exists() and json.loads(path.read_text()) != split:
            raise ValueError(f"Refusing to replace a different frozen split: {path}")
        path.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    if args.download_images:
        expected_path = Path(__file__).resolve().parent.parent / "data_manifest.json"
        expected = {}
        if expected_path.exists():
            expected_manifest = json.loads(expected_path.read_text(encoding="utf-8"))
            if expected_manifest["dataset"]["sha256"] == dataset.sha256:
                expected = {row["id"]: row["sha256"] for row in expected_manifest["images"]}
        records, failures = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_image, o, args.images): o.id
                for o in dataset.observations.values()
            }
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    record = future.result()
                    if record["id"] in expected and record["sha256"] != expected[record["id"]]:
                        raise ValueError("Image checksum differs from data_manifest.json")
                    records.append(record)
                except Exception as exc:
                    failures.append({"id": futures[future], "error": str(exc)})
                if done % 50 == 0 or done == len(futures):
                    print(f"Images: {done}/{len(futures)}; failed: {len(failures)}", flush=True)
        manifest = {
            "annotation_url": ANNOTATION_URL,
            "dataset": dataset.summary(),
            "images": sorted(records, key=lambda r: r["id"]),
            "failures": failures,
        }
        args.images.mkdir(parents=True, exist_ok=True)
        (args.images.parent / "download_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if failures:
            raise SystemExit("Some downloads failed; rerun to resume (see download_manifest.json)")


if __name__ == "__main__":
    main()
