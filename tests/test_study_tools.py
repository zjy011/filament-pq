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


def test_atomic_replace_retries_sharing_errors_and_keeps_failures(monkeypatch, tmp_path):
    from scripts import resilient_io

    attempts = []
    real_replace = resilient_io._replace
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_text("complete JSON")

    def temporarily_locked(src, dst, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("transient sharing violation")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(resilient_io, "_replace", temporarily_locked)
    monkeypatch.setattr(resilient_io.time, "sleep", lambda _: None)
    resilient_io.replace_with_retry(source, target)
    assert len(attempts) == 3 and target.read_text() == "complete JSON"

    def permanently_locked(*args, **kwargs):
        raise PermissionError("permanent denial")

    monkeypatch.setattr(resilient_io, "_replace", permanently_locked)
    with pytest.raises(PermissionError, match="permanent denial"):
        resilient_io.replace_with_retry(source, target)


def test_resilient_wrapper_preserves_windows_spawn_pickling(tmp_path):
    import shutil
    import subprocess
    import sys

    from scripts import resilient_io

    package = tmp_path / "scripts"
    package.mkdir()
    (package / "__init__.py").write_text("")
    shutil.copyfile(resilient_io.__file__, package / "resilient_io.py")
    (package / "probe.py").write_text(
        "from concurrent.futures import ProcessPoolExecutor\n"
        "from multiprocessing import get_context\n"
        "def square(value):\n    return value * value\n"
        "if __name__ == '__main__':\n"
        "    with ProcessPoolExecutor(2, mp_context=get_context('spawn')) as pool:\n"
        "        assert list(pool.map(square, [2, 3])) == [4, 9]\n"
        "    print('spawn passed')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.resilient_io", "scripts.probe"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert "spawn passed" in completed.stdout
