import json

import pytest

from filament.data import Dataset
from filament.splits import make_split, split_fingerprint, validate_split


@pytest.fixture
def dataset(tmp_path):
    images, annotations = [], []
    for site in ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]:
        for i in range(12):
            filename = f"202101{(i + 1):02d}120000{site}.jpeg"
            for annotator in range(1 + i % 3):
                key = f"{annotator}-{filename[:-5]}"
                images.append({"id": key, "file_name": filename, "height": 32, "width": 32})
                if i % 2:
                    annotations.append(
                        {
                            "id": key + "_a",
                            "image_id": key,
                            "segmentation": [[1, 1, 4, 1, 4, 4, 1, 4]],
                        }
                    )
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"images": images, "annotations": annotations}))
    return Dataset(path)


def test_groups_by_photo_and_keeps_empty_sets(dataset):
    assert len(dataset.observations) == 72
    assert dataset.summary()["annotation_sets"] == 144
    assert dataset.summary()["empty_annotation_sets"] > 0


def test_main_split_is_reproducible_and_disjoint(dataset):
    split = make_split(dataset)
    assert split == make_split(dataset)
    assert split != make_split(dataset, seed=12)
    assert not set(split["tune"]) & set(split["test"])
    assert set(split["train"]) | set(split["val"]) == set(split["tune"])
    assert {dataset.observations[key].site for key in split["test"]} == {
        "Bh",
        "Ch",
        "Lh",
        "Mh",
        "Th",
        "Uh",
    }
    split["test"].append(split["tune"][0])
    with pytest.raises(ValueError, match="leakage"):
        validate_split(dataset, split)


def test_site_holdout_and_fingerprint(dataset):
    split = make_split(dataset, held_out_site="Bh")
    assert all(dataset.observations[key].site == "Bh" for key in split["test"])
    assert all(dataset.observations[key].site != "Bh" for key in split["tune"])
    split["dataset_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different annotation"):
        validate_split(dataset, split)


def test_split_fingerprint_is_portable_across_line_endings_and_formatting(dataset, tmp_path):
    split = make_split(dataset)
    windows, linux = tmp_path / "windows.json", tmp_path / "linux.json"
    windows.write_bytes(json.dumps(split, indent=2).replace("\n", "\r\n").encode())
    linux.write_bytes(json.dumps(split, sort_keys=True).encode())
    assert split_fingerprint(windows) == split_fingerprint(linux)
