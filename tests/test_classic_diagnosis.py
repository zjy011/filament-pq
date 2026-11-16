from scripts.diagnose_classic import STAGES, miss_stage


def test_miss_attribution_handles_restoration_and_final_detection():
    overlap = dict.fromkeys(STAGES, 0)
    assert miss_stage(overlap) == "outside_disk"
    overlap["disk"] = 10
    assert miss_stage(overlap) == "candidate_extraction"
    overlap["candidate"] = 2
    assert miss_stage(overlap) == "morphology"
    overlap["morphology"] = 2
    assert miss_stage(overlap) == "area_filter"
    # Closing may restore pixels absent from the raw candidate mask.
    overlap["candidate"] = 0
    assert miss_stage(overlap) == "area_filter"
    overlap["area"] = 2
    assert miss_stage(overlap) == "shape_filter"
    overlap["shape"] = 2
    assert miss_stage(overlap) == "instance_or_boundary"
    overlap["final"] = 1
    assert miss_stage(overlap) is None
