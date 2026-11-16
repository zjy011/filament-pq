from dataclasses import replace

import numpy as np

from filament.pipeline import PipelineConfig, postprocess
from scripts.classic_order_experiment import filter_groups, recipes, variant_masks


def test_original_matches_existing_pipeline_and_late_area_rescues_fragments():
    mask = np.zeros((60, 100), bool)
    mask[20:24, 10:30] = True
    mask[20:24, 35:55] = True
    disk = np.ones_like(mask)
    config = PipelineConfig(min_area=120, min_axis_ratio=2, opening_radius=0,
                            instances="merge", merge_distance=10, dilation_radius=1)
    selected = {"original": recipes(config)["original"],
                "late": {"pre_area": 50, "pre_shape": 2, "post_area": 120, "post_shape": 1}}
    results = dict(variant_masks(mask, disk, config, selected))
    np.testing.assert_array_equal(results["original"], postprocess(mask, disk, config))
    assert results["original"].max() == 0
    assert results["late"].max() == 1
    assert np.count_nonzero(results["late"]) > 160  # expansion occurs after area acceptance
    # Also exercise baseline equality when objects survive.
    config = replace(config, min_area=50)
    result = dict(variant_masks(mask, disk, config, {"original": recipes(config)["original"]}))
    np.testing.assert_array_equal(result["original"], postprocess(mask, disk, config))


def test_shape_filter_after_grouping_preserves_elongated_group_of_compact_pieces():
    mask = np.zeros((40, 60), bool)
    mask[10:20, 5:15] = True
    mask[10:20, 20:30] = True
    config = PipelineConfig(min_area=150, min_axis_ratio=2, opening_radius=0,
                            instances="merge", merge_distance=10, dilation_radius=0)
    results = dict(variant_masks(mask, np.ones_like(mask), config, {
        "pre_shape": {"pre_area": 50, "pre_shape": 2, "post_area": 150, "post_shape": 1},
        "post_shape": {"pre_area": 50, "pre_shape": 1, "post_area": 150, "post_shape": 2},
    }))
    assert not results["pre_shape"].any()
    assert np.count_nonzero(results["post_shape"]) == 200


def test_group_area_uses_all_fragments_and_handles_noncontiguous_ids():
    labels = np.array([[0, 3, 3, 0, 3, 3], [0, 0, 0, 0, 7, 0]], dtype=np.int32)
    filtered = filter_groups(labels, 4, 1)
    assert np.count_nonzero(filtered == 3) == 4
    assert not np.any(filtered == 7)
    assert not filter_groups(np.zeros((5, 5), dtype=np.int32), 1, 2).any()
