"""Write disk/instance overlays without a GUI or plotting dependency."""

from pathlib import Path

import cv2
import numpy as np

from filament.evaluate import matching
from filament.masks import decode, iou_matrix


def save_overlay(
    path: str | Path,
    image: np.ndarray,
    gt: list[dict],
    pred: list[dict],
    disk_mask: np.ndarray | None = None,
):
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hits = matching(iou_matrix(gt, pred))
    for rles, matched, is_gt in ((gt, hits.any(axis=1), True), (pred, hits.any(axis=0), False)):
        for rle, hit in zip(rles, matched):
            color = (0, 220, 0) if hit else (0, 0, 255) if is_gt else (0, 180, 255)
            contours, _ = cv2.findContours(
                decode(rle).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, color, 2)
    if disk_mask is not None:
        contours, _ = cv2.findContours(
            disk_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (255, 255, 0), 2)
    cv2.rectangle(canvas, (10, 8), (810, 52), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        "Green: TP | Red: FN | Orange: FP | Cyan: disk",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", canvas)
    if not success:
        raise OSError(f"Could not encode overlay: {path}")
    encoded.tofile(destination)
