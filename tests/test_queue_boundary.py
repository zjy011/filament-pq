import json
from pathlib import Path

import pytest

from scripts import complete_study


def test_requested_seed_finishes_evaluation_without_launching_next_jobs(monkeypatch, tmp_path):
    calls = []

    class FakeQueue:
        def __init__(self, name):
            self.progress = tmp_path / "queue.json"

        def train(self, out, **kwargs):
            calls.append(("train", Path(out).name))

        def learned_evaluation(self, name, checkpoint, **kwargs):
            calls.append(("evaluate", name))

    monkeypatch.setattr(complete_study, "Queue", FakeQueue)
    monkeypatch.setattr(
        "sys.argv",
        ["queue", "--phase", "learned", "--extensions", "--stop-after", "unet_1024_L1_545"],
    )
    complete_study.main()
    assert calls[-2:] == [("train", "unet_1024_L1_545"), ("evaluate", "unet_1024_L1_545")]
    assert len(calls) == 8
    assert not any("L2" in name or "site_" in name for _, name in calls)
    assert json.loads((tmp_path / "queue.json").read_text()) == {
        "status": "stopped_after", "experiment": "unet_1024_L1_545"
    }


@pytest.mark.parametrize("args", [
    ["--phase", "classic", "--extensions"],
    ["--phase", "learned"],
])
def test_unreachable_stop_target_is_rejected(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["queue", *args, "--stop-after", "unet_1024_L1_545"])
    with pytest.raises(SystemExit) as error:
        complete_study.main()
    assert error.value.code == 2
