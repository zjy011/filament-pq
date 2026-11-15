"""Export aggregate results and selected recipes for Git; keep raw runs local."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml

from filament.experiment import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/report/summary.json"))
    parser.add_argument("--report", type=Path, help="Include a visually checked final PDF")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = read(args.summary)
    pending = summary["extension_analysis"]["pending"] + summary["pending_curve_points"]
    pending += [
        f"CPU benchmark: {n}"
        for n in ("baseline", "classical", "unet")
        if n not in summary.get("cpu_benchmark", {})
    ]
    if (args.require_complete or args.report) and pending:
        parser.error(f"Automatic experiments are still pending: {pending}")

    evidence, figures = Path("docs/evidence"), Path("docs/figures")
    evidence.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    artifacts = []

    def record(path, source):
        text_file = path.suffix in {".json", ".yaml"}
        payload = (
            path.read_text(encoding="utf-8").encode("utf-8") if text_file else path.read_bytes()
        )
        artifacts.append(
            {
                "path": path.as_posix(),
                "source": str(source),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "hash_mode": "utf8_lf" if text_file else "raw",
            }
        )

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        record(target, source)

    copy(args.summary, evidence / "report_summary.json")
    for name in (
        "pipeline",
        "main_results",
        "site_results",
        "sensitivity",
        "errors",
        "iou_training",
        "error_examples_fn",
        "error_examples_fp",
        "resolution_topology",
        "site_transfer",
    ):
        source = args.summary.parent / f"{name}.png"
        if source.exists():
            copy(source, figures / source.name)

    for name in summary["extension_analysis"]["completed"]:
        run = Path("results/final") / name
        target = evidence / "extensions" / name
        for filename in ("metrics.json", "provenance.json"):
            copy(run / filename, target / filename)
        provenance = read(run / "provenance.json")
        recipe = Path("configs/measured_extensions") / f"{name}.yaml"
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text(yaml.safe_dump(provenance["config"], sort_keys=False), encoding="utf-8")
        record(recipe, run / "provenance.json")
        if provenance.get("checkpoint_sha256"):
            for source, filename in (
                (Path("results") / name / "history.json", "training_history.json"),
                (Path("results") / name / "provenance.json", "training_provenance.json"),
                (
                    Path("results") / f"{name}_postprocess" / "best.json",
                    "validation_selection.json",
                ),
            ):
                copy(source, target / filename)
    if args.report:
        copy(args.report, Path("docs/final_report.pdf"))
    write_json(
        evidence / "artifact_manifest.json",
        {
            "automatic_experiments_complete": not pending,
            "pending_automatic": pending,
            "human_review": summary["manual_review"].get("status", "complete"),
            "artifacts": artifacts,
            "scope": "Aggregate evidence, selected recipes and report figures only; no raw data, "
            "weights, blind-review packages or curator key. Recipes are copied from the "
            "recorded evaluation provenance, not reselected from test scores.",
        },
    )
    print(f"Exported {len(artifacts)} artifacts; {len(pending)} automatic items pending")


if __name__ == "__main__":
    main()
