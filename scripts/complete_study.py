"""Resumable experiment queues. Run classical and learned queues separately."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from filament.experiment import write_json
from filament.pipeline import load_config
from filament.splits import split_fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Queue:
    def __init__(self, name):
        self.name = name
        self.logs = Path("results/study_logs")
        self.logs.mkdir(parents=True, exist_ok=True)
        self.progress = self.logs / f"{name}.json"

    def run(self, module, arguments, marker, resumable=False):
        marker = Path(marker)
        if marker.exists():
            if marker.name != "progress.json" or read(marker).get("status") == "complete":
                return
        arguments = [str(value) for value in arguments]
        if resumable and "--out" in arguments:
            out = Path(arguments[arguments.index("--out") + 1])
            if (out / "provenance.json").exists():
                arguments.append("--resume")
        command = [sys.executable, "-m", f"scripts.{module}", *arguments]
        tag = str(marker.parent).replace("\\", "_").replace("/", "_")
        log = self.logs / f"{tag}_{module}.log"
        write_json(
            self.progress,
            {"status": "running", "command": command, "log": str(log), "marker": str(marker)},
        )
        print(f"[{self.name}] {module}: {marker.parent}", flush=True)
        with log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode:
            write_json(
                self.progress,
                {
                    "status": "failed",
                    "returncode": completed.returncode,
                    "command": command,
                    "log": str(log),
                },
            )
            raise RuntimeError(f"{module} failed; inspect {log}")
        if not marker.exists():
            raise RuntimeError(f"Expected output missing: {marker}")

    def wait_existing(self, progress):
        while not Path(progress).exists() or read(progress).get("status") != "complete":
            write_json(self.progress, {"status": "waiting", "for": str(progress)})
            time.sleep(10)

    def evaluate(
        self,
        config,
        out,
        partition="test",
        split="splits/main.json",
        checkpoint=None,
        threshold=0.5,
    ):
        out = Path(out)
        if partition == "test" and not (out / "metrics.json").exists():
            dataset_hash = read(split)["dataset_sha256"]
            model_hash = (
                hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None
            )
            identity = [
                dataset_hash,
                split_fingerprint(split),
                asdict(load_config(config)),
                model_hash,
                threshold if checkpoint else None,
            ]
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            receipt = Path("results/test_registry") / f"{digest}.json"
            if receipt.exists():
                previous = read(receipt)
                source = Path(previous["output"])
                if previous["status"] == "complete" and source.resolve() != out.resolve():
                    shutil.copytree(
                        source, out, ignore=shutil.ignore_patterns("raw_images", "overlays")
                    )
                    write_json(
                        out / "reused_run.json",
                        {
                            "source": str(source),
                            "reason": "Identical dataset, split, config, checkpoint and threshold; no repeated test evaluation.",
                        },
                    )
                    return
        args = [
            "--config",
            config,
            "--out",
            out,
            "--partition",
            partition,
            "--split",
            split,
            "--workers",
            "4",
            "--overlays",
            "6",
            "--cache",
            "cache/full_tuning",
        ]
        if partition == "test":
            args.append("--finalize-test")
        if checkpoint:
            args.extend(
                [
                    "--checkpoint",
                    checkpoint,
                    "--probability-threshold",
                    threshold,
                    "--device",
                    "cuda",
                ]
            )
        self.run("run_experiment", args, Path(out) / "metrics.json", resumable=True)

    def train(self, out, split="splits/main.json", size=512, seed=543, loss="L1"):
        args = [
            "--out",
            out,
            "--split",
            split,
            "--size",
            size,
            "--seed",
            seed,
            "--loss",
            loss,
            "--device",
            "cuda",
            "--epochs",
            50,
            "--patience",
            8,
        ]
        if size == 1024:
            args.extend(["--batch-size", 1, "--accumulate", 4])
        self.run("train_unet", args, Path(out) / "progress.json", resumable=True)

    def learned_evaluation(self, name, checkpoint, split="splits/main.json", control=False):
        tuning = Path("results") / f"{name}_postprocess"
        self.run(
            "tune_unet",
            ["--checkpoint", checkpoint, "--split", split, "--out", tuning],
            tuning / "progress.json",
            resumable=True,
        )
        best = read(tuning / "best.json")
        # The checkpoint and val-selected recipe are now fixed before this test call.
        self.evaluate(
            tuning / "best.yaml",
            Path("results/final") / name,
            split=split,
            checkpoint=checkpoint,
            threshold=best["spec"]["threshold"],
        )
        if control:
            self.evaluate(
                tuning / "components_only.yaml",
                Path("results/final") / f"{name}_components_only",
                split=split,
                checkpoint=checkpoint,
                threshold=best["spec"]["threshold"],
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["classic", "learned"], required=True)
    parser.add_argument("--extensions", action="store_true")
    parser.add_argument(
        "--wait-for", type=Path, help="Wait for an independently running job's progress.json"
    )
    parser.add_argument(
        "--wait-existing", action="store_true", help="Wait for already-running full tune/E7a jobs"
    )
    args = parser.parse_args()
    queue = Queue(args.phase)
    if args.wait_for:
        queue.wait_existing(args.wait_for)
    if args.phase == "classic":
        if args.wait_existing:
            queue.wait_existing("results/full_tuning/progress.json")
        else:
            queue.run(
                "tune",
                [
                    "--config",
                    "configs/quick_tuned.yaml",
                    "--search",
                    "configs/search.yaml",
                    "--out",
                    "results/full_tuning",
                    "--cache",
                    "cache/full_tuning",
                ],
                "results/full_tuning/progress.json",
                resumable=True,
            )
        frozen = Path("configs/frozen_classic")
        queue.run("freeze_classic", [], frozen / "manifest.json")
        manifest = read(frozen / "manifest.json")
        # Produce the main estimate and review material first.
        for name in ("baseline", "full"):
            queue.evaluate(frozen / f"{name}.yaml", Path("results/final/classic") / name / "test")
        queue.run(
            "label_fp",
            ["--run", "results/final/classic/full/test", "--out", "results/blind_review_100"],
            "results/blind_review_100/reviewer_b.zip",
        )
        for name in manifest["config_sha256"]:
            queue.evaluate(frozen / f"{name}.yaml", Path("results/final/classic") / name / "test")
        sensitivity_names = {"full"}
        for settings in manifest["sensitivity"].values():
            sensitivity_names.update(row["name"] for row in settings)
        for name in sorted(sensitivity_names):
            name = manifest["aliases"].get(name, name)
            queue.evaluate(
                frozen / f"{name}.yaml",
                Path("results/final/classic") / name / "tune",
                partition="tune",
            )
        if args.extensions:
            for site in ("main", "Bh", "Ch", "Lh", "Mh", "Th", "Uh"):
                split = "splits/main.json" if site == "main" else f"splits/site_{site}.json"
                out = Path("results") / f"site_classic_{site}_tuning"
                queue.run(
                    "tune",
                    [
                        "--config",
                        "configs/baseline.yaml",
                        "--search",
                        "configs/search_site.yaml",
                        "--split",
                        split,
                        "--out",
                        out,
                        "--cache",
                        "cache/full_tuning",
                        "--top-k",
                        3,
                    ],
                    out / "progress.json",
                    resumable=True,
                )
                queue.evaluate(
                    out / "best.yaml", Path("results/final") / f"site_classic_{site}", split=split
                )
    else:
        if args.wait_existing:
            queue.wait_existing("results/unet_e7a/progress.json")
        else:
            queue.train("results/unet_e7a")
        queue.learned_evaluation("unet_e7a", "results/unet_e7a/best.pt", control=True)
        if args.extensions:
            for loss in ("L1", "L2"):
                for seed in (543, 544, 545):
                    name = f"unet_1024_{loss}_{seed}"
                    out = Path("results") / name
                    queue.train(out, size=1024, seed=seed, loss=loss)
                    queue.learned_evaluation(name, out / "best.pt")
            for site in ("Bh", "Ch", "Lh", "Mh", "Th", "Uh"):
                split = f"splits/site_{site}.json"
                name = f"site_unet_{site}"
                out = Path("results") / name
                queue.train(out, split=split)
                queue.learned_evaluation(name, out / "best.pt", split=split)
    write_json(queue.progress, {"status": "complete", "extensions": args.extensions})


if __name__ == "__main__":
    main()
