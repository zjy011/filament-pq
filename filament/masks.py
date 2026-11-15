"""COCO polygon rasterization and compressed RLE, without dense mask stacks."""

import csv
from pathlib import Path

import numpy as np
from pycocotools import mask as coco


def encode(mask: np.ndarray) -> dict:
    if mask.ndim != 2:
        raise ValueError("Expected a 2-D mask")
    rle = coco.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def decode(rle: dict) -> np.ndarray:
    return coco.decode(rle).astype(bool)


def polygon_rle(segmentation: list, height: int, width: int) -> dict:
    """Coordinates are COCO (x, y), with COCO's subpixel rasterization rules."""
    if not segmentation:
        raise ValueError("An instance must contain a polygon")
    for polygon in segmentation:
        if len(polygon) < 6 or len(polygon) % 2 or not np.isfinite(polygon).all():
            raise ValueError("Invalid COCO polygon")
    rle = coco.merge(coco.frPyObjects(segmentation, height, width))
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def areas(rles: list[dict]) -> np.ndarray:
    return np.asarray(coco.area(rles), dtype=np.float64) if rles else np.zeros(0)


def iou_matrix(gt: list[dict], pred: list[dict]) -> np.ndarray:
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float32)
    # Cast to float32 to match the organizer's torch mask-matmul implementation.
    return np.asarray(coco.iou(gt, pred, [0] * len(pred)), dtype=np.float32)


def union_mask(rles: list[dict], shape: tuple[int, int]) -> np.ndarray:
    return decode(coco.merge(rles)) if rles else np.zeros(shape, dtype=bool)


def label_rles(labels: np.ndarray) -> list[dict]:
    return [encode(labels == i) for i in np.unique(labels) if i != 0]


def export_csv(path: str | Path, entries: dict[str, list[dict]]) -> None:
    """Organizer columns; keys are image IDs or annotator-image IDs.

    Empty entries have no filament row. Keep the observation/annotation manifest
    alongside this CSV: the organizer's CSV-only iterator cannot discover them.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for entry_id, rles in sorted(entries.items()):
            for i, rle in enumerate(rles):
                writer.writerow([f"{entry_id}_{i}", rle["counts"]])
