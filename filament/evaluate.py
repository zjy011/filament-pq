"""Pooled PQ, exhaustive unmatched-instance errors, and image-cluster bootstrap."""

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from pycocotools import mask as coco
from scipy.optimize import linear_sum_assignment

from filament.masks import areas, iou_matrix


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sum_iou: float = 0.0
    pixel_intersection: int = 0
    pixel_gt: int = 0
    pixel_pred: int = 0
    comparisons: int = 0

    def __add__(self, other):
        return Counts(**{key: value + getattr(other, key) for key, value in asdict(self).items()})

    def metrics(self) -> dict:
        denominator = self.tp + 0.5 * (self.fp + self.fn)
        pixels = self.pixel_gt + self.pixel_pred
        return {
            **asdict(self),
            "pq": self.sum_iou / denominator if denominator else 0.0,
            "sq": self.sum_iou / self.tp if self.tp else 0.0,
            "rq": self.tp / denominator if denominator else 0.0,
            "dice": 2 * self.pixel_intersection / pixels if pixels else 1.0,
        }


def matching(iou: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if not 0 < threshold < 1:
        raise ValueError("IoU threshold must be in (0, 1)")
    eligible = iou > threshold
    if threshold >= 0.5 or not iou.size:
        # This is deliberately the organizer's hit-matrix rule, including any
        # ambiguous hits in overlapping GT polygons. Do not silently alter it.
        return eligible
    # Below .5, choose a one-to-one matching maximizing total eligible IoU.
    rows, cols = linear_sum_assignment(np.where(eligible, iou, 0), maximize=True)
    result = np.zeros_like(eligible)
    result[rows, cols] = eligible[rows, cols]
    return result


def area_bucket(area: float) -> str:
    return "small" if area < 600 else "medium" if area <= 2400 else "large"


def evaluate_pair(gt: list[dict], pred: list[dict], threshold: float = 0.5) -> dict:
    shapes = {tuple(rle["size"]) for rle in [*gt, *pred]}
    if len(shapes) > 1:
        raise ValueError("Ground truth and prediction dimensions differ")
    matrix = iou_matrix(gt, pred)
    hits = matching(matrix, threshold)
    gt_hit, pred_hit = hits.any(axis=1), hits.any(axis=0)
    gt_union = coco.merge(gt) if gt else None
    pred_union = coco.merge(pred) if pred else None
    intersection = (
        int(coco.area(coco.merge([gt_union, pred_union], intersect=True))) if gt and pred else 0
    )
    counts = Counts(
        int(hits.sum()),
        int((~pred_hit).sum()),
        int((~gt_hit).sum()),
        sum(float(x) for x in matrix[hits]),
        intersection,
        int(coco.area(gt_union)) if gt else 0,
        int(coco.area(pred_union)) if pred else 0,
        1,
    )
    gt_degree, pred_degree = (matrix > 0).sum(axis=1), (matrix > 0).sum(axis=0)
    gt_areas, pred_areas = areas(gt), areas(pred)
    errors = []
    for i in np.flatnonzero(~gt_hit):
        category = (
            "miss"
            if gt_degree[i] == 0
            else "fragmentation"
            if gt_degree[i] >= 2
            else "localization"
        )
        errors.append(
            {
                "side": "fn",
                "index": int(i),
                "category": category,
                "area": float(gt_areas[i]),
                "bucket": area_bucket(gt_areas[i]),
            }
        )
    for i in np.flatnonzero(~pred_hit):
        if pred_degree[i] == 0:
            category = "isolated"
        elif pred_degree[i] >= 2:
            category = "overmerge"
        else:
            gt_index = int(np.flatnonzero(matrix[:, i] > 0)[0])
            category = "fragment_related" if gt_degree[gt_index] >= 2 else "localization"
        errors.append(
            {
                "side": "fp",
                "index": int(i),
                "category": category,
                "area": float(pred_areas[i]),
                "bucket": area_bucket(pred_areas[i]),
            }
        )
    recall = {bucket: {"matched": 0, "total": 0} for bucket in ("small", "medium", "large")}
    for size, hit in zip(gt_areas, gt_hit):
        bucket = area_bucket(size)
        recall[bucket]["total"] += 1
        recall[bucket]["matched"] += int(hit)
    return {
        "counts": asdict(counts),
        "metrics": counts.metrics(),
        "errors": errors,
        "recall_by_area": recall,
        "matched_ious": [float(x) for x in matrix[hits]],
        "overlap_degrees": {"gt": gt_degree.tolist(), "pred": pred_degree.tolist()},
        "ambiguous_matches": int((hits.sum(axis=1) > 1).sum() + (hits.sum(axis=0) > 1).sum()),
    }


def aggregate_counts(records: list[dict]) -> Counts:
    total = Counts()
    for record in records:
        total += Counts(**record["counts"])
    return total


def summarize(records: list[dict]) -> dict:
    errors = Counter()
    recall = {bucket: {"matched": 0, "total": 0} for bucket in ("small", "medium", "large")}
    for record in records:
        for error in record["errors"]:
            errors[f"{error['side']}/{error['category']}/{error['bucket']}"] += 1
        for bucket, counts in record["recall_by_area"].items():
            for key in counts:
                recall[bucket][key] += counts[key]
    for bucket in recall.values():
        bucket["recall"] = bucket["matched"] / bucket["total"] if bucket["total"] else None
    return {
        **aggregate_counts(records).metrics(),
        "errors": dict(sorted(errors.items())),
        "recall_by_area": recall,
        "ambiguous_matches": sum(r["ambiguous_matches"] for r in records),
    }


def bootstrap_pq(
    by_image: dict[str, Counts],
    other: dict[str, Counts] | None = None,
    samples: int = 1000,
    seed: int = 543,
) -> dict:
    """Resample observations, pooling all their annotation comparisons together.

    If other is provided, return PQ(first) - PQ(other) with paired resampling.
    """
    if not by_image or samples < 1:
        raise ValueError("Bootstrap needs observations and a positive sample count")
    keys = sorted(by_image)
    if other is not None and set(other) != set(keys):
        raise ValueError("Paired bootstrap requires exactly the same observation IDs")

    def values(source):
        return np.array(
            [[source[k].sum_iou, source[k].tp + 0.5 * (source[k].fp + source[k].fn)] for k in keys]
        )

    def pq(total):
        return float(total[0] / total[1]) if total[1] else 0.0

    first = values(by_image)
    second = values(other) if other is not None else None
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        selected = rng.integers(0, len(keys), len(keys))
        score = pq(first[selected].sum(axis=0))
        if second is not None:
            score -= pq(second[selected].sum(axis=0))
        scores.append(score)
    estimate = pq(first.sum(axis=0)) - (pq(second.sum(axis=0)) if second is not None else 0)
    return {
        "estimate": estimate,
        "low": float(np.quantile(scores, 0.025)),
        "high": float(np.quantile(scores, 0.975)),
        "samples": samples,
        "seed": seed,
        "observations": len(keys),
        "paired": other is not None,
    }


def human_comparison(annotation_sets: list[list[dict]], prediction: list[dict]) -> dict:
    """Rotate held-out annotators; identical references and multiplicities for both sides."""
    if len(annotation_sets) < 2:
        raise ValueError("Human comparison requires at least two annotators")
    human, method = Counts(), Counts()
    for held_out, annotator in enumerate(annotation_sets):
        for reference, truth in enumerate(annotation_sets):
            if reference != held_out:
                human += Counts(**evaluate_pair(truth, annotator)["counts"])
                method += Counts(**evaluate_pair(truth, prediction)["counts"])
    return {"human": asdict(human), "method": asdict(method)}


def corroborated_false_positives(
    annotation_sets: list[list[dict]], prediction: list[dict], threshold: float = 0.5
) -> dict:
    """E5: unmatched relative to one annotator, overlapping another at IoU > threshold."""
    total, confirmed = 0, 0
    if len(annotation_sets) < 2:
        return {"false_positives": 0, "corroborated": 0}
    matrices = [iou_matrix(truth, prediction) for truth in annotation_sets]
    for i, matrix in enumerate(matrices):
        fp = ~matching(matrix, threshold).any(axis=0)
        other_hits = np.zeros(len(prediction), dtype=bool)
        for j, other in enumerate(matrices):
            if j != i:
                other_hits |= (other > threshold).any(axis=0)
        total += int(fp.sum())
        confirmed += int((fp & other_hits).sum())
    return {"false_positives": total, "corroborated": confirmed}
