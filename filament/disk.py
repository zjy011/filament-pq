"""M0: isolate the largest bright object and fit an inset solar disk."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SolarDisk:
    center: tuple[float, float]  # x, y
    radius: float  # before inset
    mask: np.ndarray


def detect_disk(image: np.ndarray, inset: float = 0.03) -> SolarDisk:
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected an 8-bit grayscale image")
    if not 0 <= inset < 0.5 or image.max() == image.min():
        raise ValueError("Invalid disk inset or constant image")
    _, foreground = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Solar disk not found")
    contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if cv2.contourArea(contour) < image.size * 0.1 or radius > max(image.shape) * 0.6:
        raise ValueError("Implausible solar disk; inspect this image before evaluation")
    # GONG's standardized full-disk frames have a limb near 0.44 * image width.
    # Refine with a Hough circle at <=512 px to reject the gray outer annulus
    # and prominences that inflate Otsu's minimum enclosing circle. The 0.35-0.48
    # bounds are a full-disk acquisition prior, not fitted to filament labels.
    scale = min(1.0, 512 / max(image.shape))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    side = min(small.shape)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(small, (5, 5), 1),
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=side / 2,
        param1=100,
        param2=30,
        minRadius=int(side * 0.35),
        maxRadius=int(side * 0.48),
    )
    if circles is not None:
        for x, y, r in circles[0]:
            if np.hypot(x - small.shape[1] / 2, y - small.shape[0] / 2) < side * 0.1:
                # Keep the precise contour fit when the two agree to within
                # Hough's several-pixel localization uncertainty at this scale.
                if abs(radius * scale - r) > 3 or np.hypot(cx * scale - x, cy * scale - y) > 3:
                    cx, cy, radius = float(x / scale), float(y / scale), float(r / scale)
                break
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius * (1 - inset)) ** 2
    return SolarDisk((cx, cy), float(radius), mask)
