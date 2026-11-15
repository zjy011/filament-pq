import numpy as np
import pytest

from filament.evaluate import (
    Counts,
    aggregate_counts,
    area_bucket,
    bootstrap_pq,
    corroborated_false_positives,
    evaluate_pair,
    human_comparison,
    matching,
)
from filament.masks import decode, encode, export_csv, iou_matrix, polygon_rle


def rect(x1, y1, x2, y2, size=20):
    mask = np.zeros((size, size), bool)
    mask[y1:y2, x1:x2] = True
    return encode(mask)


def test_rle_roundtrip_and_coco_xy_coordinates(tmp_path):
    rle = polygon_rle([[2, 3, 7, 3, 7, 5, 2, 5]], 12, 17)
    expected = np.zeros((12, 17), bool)
    expected[3:5, 2:7] = True
    np.testing.assert_array_equal(decode(rle), expected)
    np.testing.assert_array_equal(decode(encode(expected)), expected)
    export_csv(tmp_path / "rle.csv", {"abc": [rle], "empty": []})
    assert (tmp_path / "rle.csv").read_text().splitlines()[0] == "filament_id,segmentation_rle"


def test_perfect_and_empty_cases():
    rle = rect(1, 2, 7, 8)
    assert evaluate_pair([rle], [rle])["metrics"]["pq"] == 1
    assert evaluate_pair([], [rle])["counts"]["fp"] == 1
    assert evaluate_pair([rle], [])["counts"]["fn"] == 1
    empty = evaluate_pair([], [])["metrics"]
    assert empty["pq"] == 0 and empty["dice"] == 1


def test_threshold_is_strict_and_lower_threshold_is_one_to_one():
    truth, half = rect(0, 0, 10, 10), rect(0, 0, 5, 10)
    assert iou_matrix([truth], [half])[0, 0] == 0.5
    assert evaluate_pair([truth], [half])["counts"]["tp"] == 0
    assert evaluate_pair([truth], [half], 0.3)["counts"]["tp"] == 1
    hits = matching(np.array([[0.4, 0.39], [0.38, 0.0]]), 0.3)
    assert hits.sum() == 2 and hits[0, 1] and hits[1, 0]


def test_official_ambiguous_hit_behavior_is_preserved():
    mask = rect(1, 1, 5, 5)
    result = evaluate_pair([mask, mask], [mask])
    assert result["counts"]["tp"] == 2
    assert result["ambiguous_matches"] == 1


def test_pool_counts_instead_of_averaging_pq():
    correct = evaluate_pair([rect(0, 0, 4, 4)], [rect(0, 0, 4, 4)])
    empty = evaluate_pair([], [rect(0, 0, 2, 2), rect(4, 4, 6, 6), rect(8, 8, 10, 10)])
    pooled = aggregate_counts([correct, empty]).metrics()
    assert pooled["pq"] == 0.4
    assert pooled["pq"] == pooled["sq"] * pooled["rq"]


def test_error_taxonomy_covers_each_unmatched_instance_once():
    truth = [rect(0, 0, 10, 4), rect(12, 0, 15, 3), rect(12, 4, 15, 7), rect(0, 15, 2, 17)]
    pred = [rect(0, 0, 3, 4), rect(7, 0, 10, 4), rect(11, 0, 16, 8), rect(15, 15, 18, 18)]
    result = evaluate_pair(truth, pred)
    counts = result["counts"]
    assert sum(e["side"] == "fn" for e in result["errors"]) == counts["fn"]
    assert sum(e["side"] == "fp" for e in result["errors"]) == counts["fp"]
    assert {e["category"] for e in result["errors"]} == {
        "fragmentation",
        "fragment_related",
        "miss",
        "overmerge",
        "isolated",
        "localization",
    }


def test_same_reference_multiplicities_and_corroboration():
    first, second = rect(0, 0, 4, 4), rect(10, 10, 14, 14)
    annotations = [[first], [first, second], []]
    comparison = human_comparison(annotations, [first, second])
    assert comparison["human"]["comparisons"] == 6
    assert comparison["method"]["comparisons"] == 6
    noise = corroborated_false_positives(annotations, [first, second])
    assert noise == {"false_positives": 3, "corroborated": 3}


def test_bootstrap_pairs_observations_and_is_deterministic():
    counts = {"a": Counts(tp=2, sum_iou=1.5), "b": Counts(fp=3)}
    delta = bootstrap_pq(counts, counts, samples=100)
    assert delta["estimate"] == delta["low"] == delta["high"] == 0
    assert bootstrap_pq(counts, samples=30) == bootstrap_pq(counts, samples=30)
    with pytest.raises(ValueError, match="same observation"):
        bootstrap_pq(counts, {"a": counts["a"]})


def test_area_boundaries_and_shape_validation():
    assert [area_bucket(n) for n in [599, 600, 2400, 2401]] == [
        "small",
        "medium",
        "medium",
        "large",
    ]
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_pair([rect(0, 0, 2, 2)], [rect(0, 0, 2, 2, size=30)])
