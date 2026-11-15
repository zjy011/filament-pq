from dataclasses import replace

import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi

from filament.disk import detect_disk
from filament.pipeline import (
    PipelineConfig,
    candidates,
    merge_fragments,
    postprocess,
    prepare,
    segment,
    split_instances,
)


def solar_image():
    image = np.zeros((128, 128), np.uint8)
    cv2.circle(image, (64, 64), 55, 200, -1)
    cv2.line(image, (40, 50), (85, 70), 90, 5)
    cv2.putText(image, "A", (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    return image


def config(**kwargs):
    return PipelineConfig(background="none", min_area=10, opening_radius=0, **kwargs)


def test_disk_and_pipeline_ignore_corner_text():
    image = solar_image()
    disk = detect_disk(image)
    np.testing.assert_allclose(disk.center, (64, 64), atol=1)
    assert abs(disk.radius - 55) < 1
    labels = segment(image, config())
    assert labels.max() == 1 and labels[60, 62] == 1
    assert not labels[:15].any()


@pytest.mark.parametrize("background", ["none", "radial", "median"])
def test_all_backgrounds_produce_finite_ratio(background):
    cfg = replace(config(), background=background, background_size=128, background_kernel=15)
    ratio, mask = prepare(solar_image(), cfg)
    assert np.isfinite(ratio).all() and (ratio[~mask] == 1).all()
    assert ratio[52, 45] < 0.8


def test_hysteresis_grows_only_from_dark_seeds():
    ratio = np.ones((20, 20), np.float32)
    ratio[3, 3:8] = 0.92
    ratio[3, 3] = 0.8
    ratio[12, 12:16] = 0.92
    candidate = candidates(ratio, np.ones_like(ratio, bool), config(threshold_method="hysteresis"))
    assert candidate[3, 3:8].all() and not candidate[12].any()
    with pytest.raises(ValueError, match="seed_threshold"):
        config(seed_threshold=0.96, grow_threshold=0.88)


def test_frangi_path_runs():
    image = solar_image()
    assert segment(image, config(threshold_method="frangi")).shape == image.shape


def test_fragment_merge_preserves_masks_and_angle():
    mask = np.zeros((60, 80), bool)
    mask[10:14, 5:25] = True
    mask[10:14, 29:49] = True
    mask[19:40, 50:54] = True
    labels, _ = ndi.label(mask)
    merged = merge_fragments(labels, distance=6, angle=20)
    assert merged[11, 10] == merged[11, 35]
    assert merged[25, 51] != merged[11, 35]
    np.testing.assert_array_equal(merged > 0, mask)


def test_dilation_keeps_ids_disjoint_and_inside_disk():
    mask = np.zeros((30, 30), bool)
    mask[10:13, 8:11] = True
    mask[10:13, 15:18] = True
    disk = np.ones_like(mask)
    disk[:, :7] = False
    labels = postprocess(mask, disk, replace(config(), min_area=1, dilation_radius=5))
    assert set(np.unique(labels)) == {0, 1, 2}
    assert not labels[~disk].any()
    assert (labels > 0).sum() > mask.sum()


def test_watershed_keeps_merged_disconnected_instance():
    labels = np.zeros((30, 30), np.int32)
    labels[3:9, 2:10] = 1
    labels[3:9, 14:22] = 1
    result = split_instances(labels, config(split_min_area=20, split_peak_distance=2))
    assert result.max() == 1


def test_cache_depends_on_pixels_and_preprocessing_only(tmp_path):
    image = solar_image()
    first, _ = prepare(image, config(), tmp_path)
    cached, _ = prepare(image, config(threshold=0.85), tmp_path)
    np.testing.assert_array_equal(first, cached)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    image[60, 64] = 120
    prepare(image, config(), tmp_path)
    prepare(image, replace(config(), background="radial"), tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_blank_input_fails_loudly():
    with pytest.raises(ValueError):
        detect_disk(np.zeros((30, 30), np.uint8))


def test_median_background_does_not_create_a_dark_limb_annulus():
    yy, xx = np.indices((256, 256))
    radius = np.hypot(yy - 128, xx - 128)
    image = np.where(radius < 110, 220 - radius * 0.9, 0).astype(np.uint8)
    cfg = replace(config(), background="median", background_size=128, background_kernel=15)
    ratio, mask = prepare(image, cfg)
    edge = mask & (radius > 95)
    assert np.quantile(ratio[edge], 0.05) > 0.90


def test_disk_finds_limb_inside_bright_gray_annulus():
    image = np.zeros((256, 256), np.uint8)
    cv2.circle(image, (128, 128), 123, 130, -1)
    cv2.circle(image, (128, 128), 105, 200, -1)
    cv2.circle(image, (128, 128), 105, 255, 2)
    disk = detect_disk(image)
    assert abs(disk.radius - 106) < 3
    assert not disk.mask[128, 240]
