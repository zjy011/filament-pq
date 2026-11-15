"""Configurable M0-M6 classical pipeline; dimensions are native image pixels."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import frangi
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import expand_labels, watershed

from filament.disk import SolarDisk, detect_disk


@dataclass(frozen=True)
class PipelineConfig:
    disk_inset: float = 0.03
    background: str = "median"  # none, radial, median
    background_size: int = 512
    background_kernel: int = 31
    gaussian_sigma: float = 0.0
    threshold_method: str = "single"  # single, hysteresis, frangi
    threshold: float = 0.90
    seed_threshold: float = 0.88  # darker seeds: smaller intensity ratio
    grow_threshold: float = 0.94
    frangi_scales: tuple[float, ...] = (2, 4, 8)
    frangi_threshold: float = 0.01
    opening_radius: int = 1
    closing_radius: int = 0
    min_area: int = 600
    min_axis_ratio: float = 1.0
    instances: str = "components"  # components, merge, watershed
    merge_distance: float = 20.0
    merge_angle: float = 20.0
    split_min_area: int = 10000
    split_peak_distance: int = 20
    dilation_radius: int = 0

    def __post_init__(self):
        if self.background not in {"none", "radial", "median"}:
            raise ValueError("Unknown background method")
        if self.threshold_method not in {"single", "hysteresis", "frangi"}:
            raise ValueError("Unknown threshold method")
        if self.instances not in {"components", "merge", "watershed"}:
            raise ValueError("Unknown instance method")
        if not 0 <= self.disk_inset < 0.5:
            raise ValueError("disk_inset must be in [0, 0.5)")
        if not 0 < self.seed_threshold <= self.grow_threshold <= 2:
            raise ValueError("Require 0 < seed_threshold <= grow_threshold <= 2")
        if not 0 < self.threshold <= 2 or self.gaussian_sigma < 0:
            raise ValueError("Invalid threshold or Gaussian sigma")
        if self.background_kernel < 3 or self.background_kernel % 2 != 1:
            raise ValueError("background_kernel must be odd and >= 3")
        if self.background_size < self.background_kernel:
            raise ValueError("background_size must be >= background_kernel")
        if min(self.opening_radius, self.closing_radius, self.dilation_radius) < 0:
            raise ValueError("Morphology radii must be nonnegative")
        if self.min_area < 1 or self.min_axis_ratio < 1:
            raise ValueError("Invalid shape filter")
        if self.merge_distance < 0 or not 0 <= self.merge_angle <= 90:
            raise ValueError("Invalid merge parameters")
        if self.split_min_area < 1 or self.split_peak_distance < 1:
            raise ValueError("Invalid watershed parameters")
        if not self.frangi_scales or min(self.frangi_scales) <= 0 or self.frangi_threshold < 0:
            raise ValueError("Invalid Frangi parameters")


def load_config(path: str | Path) -> PipelineConfig:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PipelineConfig(**values)


def correct_background(image: np.ndarray, disk: SolarDisk, config: PipelineConfig) -> np.ndarray:
    values = image.astype(np.float32)
    typical = float(np.median(values[disk.mask]))
    if config.background == "none":
        background = np.full(image.shape, typical, dtype=np.float32)
    elif config.background == "median":
        size = config.background_size
        small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(
            disk.mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        # Extend local limb intensities. Filling with a global median makes an
        # artificial dark annulus in I/B next to the masked-out solar boundary.
        nearest = ndi.distance_transform_edt(
            ~small_mask, return_distances=False, return_indices=True
        )
        small[~small_mask] = small[tuple(nearest[:, ~small_mask])]
        background = cv2.resize(
            cv2.medianBlur(small, config.background_kernel),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
    else:
        yy, xx = np.indices(image.shape)
        radial = np.hypot(xx - disk.center[0], yy - disk.center[1]).astype(np.int32)
        bins = radial[disk.mask]
        pixels = values[disk.mask]
        order = np.argsort(bins, kind="stable")
        unique, starts, counts = np.unique(bins[order], return_index=True, return_counts=True)
        sorted_pixels = pixels[order]
        medians = np.array([np.median(sorted_pixels[s : s + n]) for s, n in zip(starts, counts)])
        profile = np.interp(np.arange(radial.max() + 1), unique, medians)
        background = profile[radial].astype(np.float32)
    ratio = values / np.maximum(background, 1.0)
    ratio[~disk.mask] = 1.0
    return ratio


def prepare(image: np.ndarray, config: PipelineConfig, cache: str | Path | None = None):
    """Cache M0/M1 by image content, relevant parameters and implementation version."""
    path = None
    if cache is not None:
        settings = {
            key: asdict(config)[key]
            for key in ("disk_inset", "background", "background_size", "background_kernel")
        }
        digest = hashlib.sha256(
            b"m0m1-v4"
            + str(image.shape).encode()
            + image.tobytes()
            + json.dumps(settings, sort_keys=True).encode()
        ).hexdigest()
        path = Path(cache) / f"{digest}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                return data["ratio"], data["mask"]
    disk = detect_disk(image, config.disk_inset)
    ratio = correct_background(image, disk, config)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(suffix=".npz", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                # Repeated coordinate-search reads are CPU-bound with DEFLATE.
                # Store exact arrays uncompressed; cached and uncached values agree.
                np.savez(handle, ratio=ratio, mask=disk.mask)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return ratio, disk.mask


def candidates(ratio: np.ndarray, disk_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    smooth = ndi.gaussian_filter(ratio, config.gaussian_sigma) if config.gaussian_sigma else ratio
    if config.threshold_method == "single":
        return (smooth < config.threshold) & disk_mask
    seeds = (smooth < config.seed_threshold) & disk_mask
    allowed = (smooth < config.grow_threshold) & disk_mask
    if config.threshold_method == "frangi":
        response = frangi(smooth, sigmas=config.frangi_scales, black_ridges=True)
        allowed &= response >= config.frangi_threshold
        seeds &= allowed
    return ndi.binary_propagation(seeds, structure=np.ones((3, 3)), mask=allowed)


def _kernel(radius: int):
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy <= radius * radius).astype(np.uint8)


def clean_mask(mask: np.ndarray, disk_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    cleaned = mask.astype(np.uint8)
    for radius, operation in (
        (config.opening_radius, cv2.MORPH_OPEN),
        (config.closing_radius, cv2.MORPH_CLOSE),
    ):
        if radius:
            cleaned = cv2.morphologyEx(cleaned, operation, _kernel(radius))
    labels, _ = ndi.label(cleaned.astype(bool) & disk_mask, structure=np.ones((3, 3)))
    keep = np.bincount(labels.ravel()) >= config.min_area
    keep[0] = False
    if config.min_axis_ratio > 1:
        filtered, _ = ndi.label(keep[labels], structure=np.ones((3, 3)))
        keep = np.zeros(int(filtered.max()) + 1, dtype=bool)
        for region in regionprops(filtered):
            ratio = region.axis_major_length / max(region.axis_minor_length, 1e-6)
            keep[region.label] = ratio >= config.min_axis_ratio
        labels = filtered
    return keep[labels]


def merge_fragments(labels: np.ndarray, distance: float, angle: float) -> np.ndarray:
    regions = regionprops(labels)
    parent = np.arange(len(regions) + 1)
    features = []
    for region in regions:
        skeleton = np.argwhere(skeletonize(region.image))
        if len(skeleton) < 2:
            skeleton = np.argwhere(region.image)
        centered = skeleton - skeleton.mean(axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        direction = vectors[:, -1]
        boundary = region.image & ~ndi.binary_erosion(region.image)
        points = np.argwhere(boundary) + np.array(region.bbox[:2])
        features.append((direction, points, cKDTree(points)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, first in enumerate(regions):
        for j in range(i + 1, len(regions)):
            second = regions[j]
            a, b = first.bbox, second.bbox
            dy = max(a[0] - b[2] + 1, b[0] - a[2] + 1, 0)
            dx = max(a[1] - b[3] + 1, b[1] - a[3] + 1, 0)
            if np.hypot(dy, dx) > distance:
                continue
            alignment = np.clip(abs(features[i][0] @ features[j][0]), 0, 1)
            if np.degrees(np.arccos(alignment)) > angle:
                continue
            nearest = features[i][2].query(features[j][1], k=1)[0].min()
            if nearest <= distance:
                parent[root(second.label)] = root(first.label)
    mapping = np.zeros(len(regions) + 1, dtype=np.int32)
    roots = {}
    for region in regions:
        key = root(region.label)
        mapping[region.label] = roots.setdefault(key, len(roots) + 1)
    return mapping[labels]


def split_instances(labels: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Conservative optional watershed; disconnected merged fragments stay grouped."""
    result = np.zeros_like(labels)
    next_label = 1
    for region in regionprops(labels):
        patch = region.image
        pieces = patch.astype(np.int32)
        _, components = ndi.label(patch, structure=np.ones((3, 3)))
        if region.area >= config.split_min_area and components == 1:
            distances = ndi.distance_transform_edt(np.pad(patch, 1))[1:-1, 1:-1]
            peaks = peak_local_max(
                distances,
                min_distance=config.split_peak_distance,
                labels=patch.astype(np.uint8),
                exclude_border=False,
            )
            if len(peaks) > 1:
                markers = np.zeros_like(pieces)
                markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
                proposed = watershed(-distances, markers, mask=patch)
                sizes = np.bincount(proposed.ravel())[1:]
                if sizes.size and sizes.min() >= config.min_area:
                    pieces = proposed
        target = result[region.slice]
        for key in np.unique(pieces):
            if key:
                target[pieces == key] = next_label
                next_label += 1
    return result


def postprocess(mask: np.ndarray, disk_mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Shared M4-M6 entry point for classical and learned foreground masks."""
    labels, _ = ndi.label(clean_mask(mask, disk_mask, config), structure=np.ones((3, 3)))
    if config.instances != "components" and labels.max():
        labels = merge_fragments(labels, config.merge_distance, config.merge_angle)
    if config.instances == "watershed":
        labels = split_instances(labels, config)
    if config.dilation_radius:
        # Nearest-label expansion prevents overlaps; does not merge instance IDs.
        labels = expand_labels(labels, distance=config.dilation_radius)
    labels[~disk_mask] = 0
    return labels.astype(np.int32)


def segment(image: np.ndarray, config: PipelineConfig, cache: str | Path | None = None):
    ratio, disk_mask = prepare(image, config, cache)
    return postprocess(candidates(ratio, disk_mask, config), disk_mask, config)
