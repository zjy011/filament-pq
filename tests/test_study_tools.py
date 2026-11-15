import csv

import pytest

from filament.pipeline import PipelineConfig
from scripts.freeze_classic import parameter_ranges
from scripts.review_page import make_review_page
from scripts.summarize_reviews import load_reviews


def test_review_page_contains_only_anonymous_samples():
    page = make_review_page([["sample_001", "crops/sample_001.png", "", ""]], "a")
    assert "curator_only" not in page
    assert "reviewer_" in page
    assert "possible_filament" in page
    assert "sample_001" in page
    assert "fetch(" not in page


def test_reviews_reject_missing_invalid_or_duplicate_labels(tmp_path):
    path = tmp_path / "review.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label"])
        writer.writerow(["sample_001", ""])
    with pytest.raises(ValueError, match="Complete every label"):
        load_reviews(path, ["sample_001"])
    path.write_text("sample_id,label\nsample_001,sunspot\nsample_001,other\n")
    with pytest.raises(ValueError, match="without duplicates"):
        load_reviews(path, ["sample_001", "sample_002"])
    path.write_text("sample_id,label\nsample_001,sunspot\n")
    assert load_reviews(path, ["sample_001"]) == {"sample_001": "sunspot"}


def test_sensitivity_excludes_inactive_parameters_and_uses_tune_only():
    from dataclasses import asdict

    config = PipelineConfig()
    trials = [
        {"config": {**asdict(config), "threshold": t}, "metrics": {"pq": score}}
        for t, score in ((0.85, 0.1), (0.9, 0.2), (0.96, 0.05))
    ]
    grid, impacts = parameter_ranges(trials, config)
    assert "threshold" in grid
    assert impacts["threshold"] == pytest.approx(0.15)
    assert "merge_distance" not in impacts
    assert "grow_threshold" not in impacts
