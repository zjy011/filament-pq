"""Compare against inspected functions from the organizer's notebook on tune images.

Download source with the Kaggle API and pass its saved response as --notebook.
Only the four named scoring definitions are loaded, never notebook top-level cells.
Review the source before using a new notebook version.
"""

import argparse
import ast
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from filament.data import Dataset
from filament.evaluate import Counts, evaluate_pair
from filament.experiment import write_json
from filament.masks import decode, iou_matrix, label_rles
from filament.pipeline import load_config, segment
from filament.splits import load_split


def load_reference(path):
    response = json.loads(Path(path).read_text(encoding="utf-8"))
    if "blob" in response:
        blob = response["blob"]
        notebook = json.loads(blob.get("source") or blob["sourceNullable"])
    else:
        notebook = response
    source = "\n\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    names = {"get_overlap_matrices", "fp_count_hit", "fn_count_hit", "get_pq_score"}
    definitions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise ValueError("Notebook is missing the expected organizer functions")
    namespace = {"torch": torch, "pd": SimpleNamespace(DataFrame=object)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def itertuples(self, index=False):
        return iter(self.rows)


def layers(rles, shape):
    array = np.stack([decode(rle) for rle in rles]) if rles else np.zeros((0, *shape))
    return torch.from_numpy(array.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--annotations", default="data/magfilo_2024_v1.0.json")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--split", default="splits/main.json")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", default="results/official_validation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference, source_hash = load_reference(args.notebook)
    dataset = Dataset(args.annotations, args.images)
    split = load_split(dataset, args.split)
    config = load_config(args.config)
    rng = np.random.default_rng(543)
    ids = sorted(rng.choice(split["tune"], min(args.count, len(split["tune"])), replace=False))
    rows, counts, differences = [], Counts(), []
    for key in ids:
        observation = dataset.observations[key]
        pred = label_rles(segment(dataset.read_image(key), config))
        for annotation in observation.annotations:
            gt = annotation.rles(observation.shape)
            reference_iou, _ = reference["get_overlap_matrices"](
                layers(gt, observation.shape), layers(pred, observation.shape)
            )
            ours = iou_matrix(gt, pred)
            np.testing.assert_array_equal(ours, reference_iou.numpy())
            rows.append(SimpleNamespace(iou_matrix=reference_iou, n_gt=len(gt), n_pred=len(pred)))
            counts += Counts(**evaluate_pair(gt, pred)["counts"])
            differences.append(
                float(np.max(np.abs(ours - reference_iou.numpy()))) if ours.size else 0
            )
        print(f"Checked {key}", flush=True)
    # Explicitly include an empty annotation with a prediction: a CSV-only GT
    # iterator would omit this case, although the organizer's scorer supports it.
    rows.append(SimpleNamespace(iou_matrix=torch.zeros((0, 1)), n_gt=0, n_pred=1))
    counts += Counts(fp=1, comparisons=1)
    expected = reference["get_pq_score"](Rows(rows))
    actual = counts.metrics()["pq"]
    if actual != expected:
        raise AssertionError(f"PQ mismatch: {actual} != {expected}")
    result = {
        "reference": "https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook",
        "source_sha256": source_hash,
        "dataset_sha256": dataset.sha256,
        "config": asdict(config),
        "image_ids": ids,
        "observations": len(ids),
        "comparisons_including_empty_fixture": len(rows),
        "max_iou_difference": max(differences),
        "official_pq": expected,
        "local_pq": actual,
        "exact_match": True,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
