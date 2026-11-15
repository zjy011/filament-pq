"""Stable, observation-grouped splits stratified by site and annotator count."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from filament.data import Dataset


def _partition(dataset: Dataset, ids: list[str], fraction: float, seed: int):
    strata = defaultdict(list)
    for key in sorted(ids):
        observation = dataset.observations[key]
        strata[(observation.site, len(observation.annotations))].append(key)
    rng = np.random.default_rng(seed)
    left, right = [], []
    for _, group in sorted(strata.items()):
        group = list(rng.permutation(group))
        n = int(np.floor(len(group) * fraction + 0.5))
        if len(group) > 1:
            n = min(len(group) - 1, max(1, n))
        right.extend(group[:n])
        left.extend(group[n:])
    return sorted(left), sorted(right)


def make_split(dataset: Dataset, seed: int = 543, held_out_site: str | None = None) -> dict:
    ids = list(dataset.observations)
    if held_out_site is None:
        tune, test = _partition(dataset, ids, 0.30, seed)
    else:
        if held_out_site not in {o.site for o in dataset.observations.values()}:
            raise ValueError(f"Unknown site: {held_out_site}")
        tune = [i for i in ids if dataset.observations[i].site != held_out_site]
        test = [i for i in ids if dataset.observations[i].site == held_out_site]
    train, val = _partition(dataset, tune, 0.10, seed + 1)
    split = {
        "schema_version": 1,
        "seed": seed,
        "dataset_sha256": dataset.sha256,
        "held_out_site": held_out_site,
        "tune": tune,
        "test": test,
        "train": train,
        "val": val,
    }
    validate_split(dataset, split)
    return split


def validate_split(dataset: Dataset, split: dict) -> None:
    if split["dataset_sha256"] != dataset.sha256:
        raise ValueError("Split belongs to a different annotation file")
    groups = {}
    for key in ("tune", "test", "train", "val"):
        values = split[key]
        groups[key] = set(values)
        if len(values) != len(groups[key]):
            raise ValueError(f"Duplicate IDs in {key}")
    if groups["tune"] & groups["test"] or groups["train"] & groups["val"]:
        raise ValueError("Observation leakage across partitions")
    if groups["train"] | groups["val"] != groups["tune"]:
        raise ValueError("train + val must equal tune")
    if groups["tune"] | groups["test"] != set(dataset.observations):
        raise ValueError("Split must cover every observation exactly once")


def load_split(dataset: Dataset, path: str | Path) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split(dataset, split)
    return split


def split_fingerprint(path: str | Path) -> str:
    """Canonical JSON hash so Windows/Colab line endings do not change identity."""
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(split, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
