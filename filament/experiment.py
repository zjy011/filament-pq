"""Reusable image jobs and provenance for experiment and tuning CLIs."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cv2

from filament.evaluate import aggregate_counts, evaluate_pair
from filament.masks import label_rles
from filament.pipeline import PipelineConfig, segment
from filament.splits import split_fingerprint


def predict_job(job):
    image_path, observation, config, cache = job
    import numpy as np

    cv2.setNumThreads(1)
    path = Path(image_path)
    if not path.exists():
        path = path.with_suffix(".jpg")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != observation.shape:
        raise ValueError(f"Invalid image: {path}")
    start = perf_counter()
    pred = label_rles(segment(image, PipelineConfig(**config), cache))
    seconds = perf_counter() - start
    comparisons = [
        {"annotation_id": a.id, **evaluate_pair(a.rles(observation.shape), pred)}
        for a in observation.annotations
    ]
    return {
        "id": observation.id,
        "site": observation.site,
        "seconds": seconds,
        "predictions": pred,
        "comparisons": comparisons,
        "counts": asdict(aggregate_counts(comparisons)),
    }


def run_predictions(dataset, ids, config, workers=1, cache=None):
    jobs = [
        (
            str(dataset.images / dataset.observations[key].file_name),
            dataset.observations[key],
            asdict(config),
            str(cache) if cache else None,
        )
        for key in ids
    ]
    if workers == 1:
        return [predict_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(predict_job, jobs))


def provenance(dataset, split_path, config):
    source_root = Path(__file__).resolve().parent.parent
    source_hash = hashlib.sha256()
    for directory in ("filament", "scripts"):
        for path in sorted((source_root / directory).rglob("*.py")):
            source_hash.update(path.relative_to(source_root).as_posix().encode())
            source_hash.update(path.read_bytes())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    return {
        "dataset_sha256": dataset.sha256,
        "source_sha256": source_hash.hexdigest(),
        "split_sha256": split_fingerprint(split_path),
        "config": asdict(config),
        "git_revision": revision,
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "opencv-python-headless", "scikit-image", "pycocotools")
        },
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
