import pytest

from scripts import run_order_followup


@pytest.mark.parametrize("remainder_gain,combined_low", [(-0.001, 0.001), (0.001, -0.001)])
def test_test_evaluation_is_blocked_when_either_tuning_condition_fails(
    monkeypatch, remainder_gain, combined_low
):
    monkeypatch.setattr("sys.argv", ["followup"])
    monkeypatch.setattr(run_order_followup, "read", lambda _: {
        "cohorts": {
            "remainder": {"variants": {"late_area_300": {
                "delta_vs_original": {"estimate": remainder_gain}
            }}},
            "combined_tune": {"variants": {"late_area_300": {
                "delta_vs_original": {"low": combined_low}
            }}},
        }
    })
    def forbidden_data_access(*args):
        raise AssertionError("Test data must not be opened before the selection gate passes")
    monkeypatch.setattr(run_order_followup, "Dataset", forbidden_data_access)
    with pytest.raises(SystemExit) as error:
        run_order_followup.main()
    assert error.value.code == 2
