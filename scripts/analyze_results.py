"""Build E1-E7 summaries and scientific figures from saved results, without inference."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2

os.environ.setdefault("MPLCONFIGDIR", str(Path("cache/matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from filament.data import Dataset
from filament.evaluate import Counts, bootstrap_pq
from filament.experiment import write_json
from filament.masks import decode

COLORS = ["#667785", "#157a8a", "#ce6b3b", "#7452a1"]
ERRORS = [
    ("fn", "miss"),
    ("fn", "fragmentation"),
    ("fn", "localization"),
    ("fp", "overmerge"),
    ("fp", "isolated"),
    ("fp", "fragment_related"),
    ("fp", "localization"),
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(path):
    path = Path(path)
    return {
        "path": str(path),
        "metrics": read(path / "metrics.json"),
        "images": read(path / "per_image.json"),
        "provenance": read(path / "provenance.json"),
    }


def paired(first, second):
    for key in ("dataset_sha256", "split_sha256", "partition"):
        if first["provenance"][key] != second["provenance"][key]:
            raise ValueError(f"Comparison differs in {key}")
    return bootstrap_pq(
        {r["id"]: Counts(**r["counts"]) for r in first["images"]},
        {r["id"]: Counts(**r["counts"]) for r in second["images"]},
    )


def errors(metrics):
    counts = Counter()
    for key, value in metrics["errors"].items():
        side, category, _ = key.split("/")
        counts[f"{side}/{category}"] += value
    return dict(counts)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize_extensions(extensions):
    """Only complete seed groups receive aggregate estimates."""
    losses, sites = ("L1", "L2"), ("Bh", "Ch", "Lh", "Mh", "Th", "Uh")
    expected = [f"unet_1024_{loss}_{seed}" for loss in losses for seed in (543, 544, 545)]
    expected += ["site_classic_main"] + [
        f"site_{method}_{site}" for method in ("classic", "unet") for site in sites
    ]
    result = {
        "completed": [name for name in expected if name in extensions],
        "pending": [name for name in expected if name not in extensions],
        "loss_groups": {},
        "seed_rows": [],
        "site_rows": [],
    }
    for loss in losses:
        rows = []
        for seed in (543, 544, 545):
            name = f"unet_1024_{loss}_{seed}"
            if name not in extensions:
                continue
            m = extensions[name]
            row = {
                "name": name,
                "loss": loss,
                "seed": seed,
                "pq": m["pq"],
                "sq": m["sq"],
                "rq": m["rq"],
                "dice": m["dice"],
                "fn_fragmentation": sum(
                    v for k, v in m["errors"].items() if k.startswith("fn/fragmentation/")
                ),
            }
            rows.append(row)
            result["seed_rows"].append(row)
        if len(rows) == 3:
            result["loss_groups"][loss] = {
                metric: {
                    "mean": float(np.mean([r[metric] for r in rows])),
                    "sample_sd": float(np.std([r[metric] for r in rows], ddof=1)),
                }
                for metric in ("pq", "fn_fragmentation")
            }
    main = extensions.get("site_classic_main")
    if main:
        unet = read("results/final/unet_e7a/metrics.json")
        for site in sites:
            for method, reference in (("classic", main), ("unet", unet)):
                held = extensions.get(f"site_{method}_{site}")
                if held:
                    before = reference["by_site"][site]
                    result["site_rows"].append(
                        {
                            "site": site,
                            "method": method,
                            "main_pq": before["pq"],
                            "held_out_pq": held["pq"],
                            "held_out_minus_main": held["pq"] - before["pq"],
                            "main_comparisons": before["comparisons"],
                            "held_out_comparisons": held["comparisons"],
                            "held_out_observations": held["observations"],
                        }
                    )
    return result


def extension_figures(summary, out):
    analysis = summary["extension_analysis"]
    rows = analysis["seed_rows"]
    if len(rows) == 6:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
        for ax, metric, title in zip(axes, ("pq", "fn_fragmentation"), ("PQ", "FN fragmentation")):
            for j, loss in enumerate(("L1", "L2")):
                group = analysis["loss_groups"][loss][metric]
                ax.bar(
                    j,
                    group["mean"],
                    yerr=group["sample_sd"],
                    color=COLORS[j + 1],
                    alpha=0.6,
                    capsize=4,
                )
                points = [r[metric] for r in rows if r["loss"] == loss]
                ax.scatter(np.arange(3) * 0.06 + j - 0.06, points, color="#202b35", zorder=3)
            reference = summary["methods"]["U-Net"]
            reference_value = (
                reference["metrics"]["pq"]
                if metric == "pq"
                else reference["errors"]["fn/fragmentation"]
            )
            ax.axhline(reference_value, ls="--", color=COLORS[2], label="512 L1, one seed")
            ax.set_xticks([0, 1], ["1024 L1", "1024 L2"])
            ax.set(ylabel=title, title=f"{title}: mean and sample SD")
            ax.legend(fontsize=8)
        save(fig, out, "resolution_topology")
    if len(analysis["site_rows"]) == 12:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
        for ax, method in zip(axes, ("classic", "unet")):
            rows = [r for r in analysis["site_rows"] if r["method"] == method]
            for key, label, color in (
                ("main_pq", "Mixed-site main subset", COLORS[0]),
                ("held_out_pq", "Entire excluded site", COLORS[1]),
            ):
                ax.plot(
                    [r["site"] for r in rows],
                    [r[key] for r in rows],
                    "o-",
                    label=label,
                    color=color,
                )
            ax.set(
                ylabel="PQ",
                xlabel="Site",
                title="Classical restricted recipe" if method == "classic" else "512-pixel U-Net",
            )
            ax.legend(fontsize=8)
        save(fig, out, "site_transfer")


def error_examples(full, out):
    dataset = Dataset("data/magfilo_2024_v1.0.json", "data/images")
    predictions = read(Path(full["path"]) / "predictions.json")
    examples = []
    figures = {
        "fn": plt.subplots(3, 2, figsize=(9, 7.5), layout="constrained"),
        "fp": plt.subplots(4, 2, figsize=(9, 10), layout="constrained"),
    }
    for row, (side, category) in enumerate(ERRORS):
        available = {}
        for record in full["images"]:
            for comparison in record["comparisons"]:
                for error in comparison["errors"]:
                    if error["side"] == side and error["category"] == category:
                        available.setdefault(
                            (record["id"], error["index"]), (record, comparison, error)
                        )
        keys = sorted(available)
        indices = np.random.default_rng(543 + row).choice(
            len(keys), min(2, len(keys)), replace=False
        )
        for col in range(2):
            ax = figures[side][1][row if side == "fn" else row - 3, col]
            ax.axis("off")
            if col >= len(indices):
                ax.text(0.5, 0.5, "No additional example", ha="center")
                continue
            record, comparison, error = available[keys[int(indices[col])]]
            observation = dataset.observations[record["id"]]
            annotation = next(
                a for a in observation.annotations if a.id == comparison["annotation_id"]
            )
            truth, pred = annotation.rles(observation.shape), predictions[observation.id]
            target = decode((truth if side == "fn" else pred)[error["index"]])
            yy, xx = np.where(target)
            y1, y2 = max(0, int(yy.min()) - 65), min(target.shape[0], int(yy.max()) + 66)
            x1, x2 = max(0, int(xx.min()) - 65), min(target.shape[1], int(xx.max()) + 66)
            crop = cv2.cvtColor(
                dataset.read_image(observation.id)[y1:y2, x1:x2], cv2.COLOR_GRAY2RGB
            )
            for instances, color in ((truth, (0, 240, 240)), (pred, (255, 140, 30))):
                for instance in instances:
                    patch = decode(instance)[y1:y2, x1:x2].astype(np.uint8)
                    if patch.any():
                        contours, _ = cv2.findContours(
                            patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        cv2.drawContours(crop, contours, -1, color, 1)
            focus, _ = cv2.findContours(
                target[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(crop, focus, -1, (0, 240, 240) if side == "fn" else (255, 140, 30), 2)
            ax.imshow(crop)
            ax.set_title(
                f"{side.upper()} / {category.replace('_', ' ')}\n{observation.id}", fontsize=12
            )
            examples.append(
                {
                    "side": side,
                    "category": category,
                    "id": observation.id,
                    "annotation_id": annotation.id,
                    "index": error["index"],
                }
            )
    for side, (fig, _) in figures.items():
        save(fig, out, f"error_examples_{side}")
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/report"))
    parser.add_argument("--skip-examples", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Draft only: mark unfinished E3 curve points",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    manifest = read("configs/frozen_classic/manifest.json")
    baseline = run("results/final/classic/baseline/test")
    full = run("results/final/classic/full/test")
    learned = run("results/final/unet_e7a")
    control = run("results/final/unet_e7a_components_only")
    methods = {"Baseline": baseline, "Classical": full, "U-Net": learned, "U-Net CC only": control}
    summary = {
        "methods": {
            name: {"path": r["path"], "metrics": r["metrics"], "errors": errors(r["metrics"])}
            for name, r in methods.items()
        },
        "paired": {
            "classical_minus_baseline": paired(full, baseline),
            "unet_minus_classical": paired(learned, full),
            "unet_minus_components_only": paired(learned, control),
        },
        "frozen_classic": manifest,
        "classic_config": full["provenance"]["config"],
        "unet_config": learned["provenance"]["config"],
        "unet_threshold": learned["provenance"]["probability_threshold"],
        "ablations": {},
        "sensitivity": {},
        "extensions": {},
        "pending_curve_points": [],
    }
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 2.7))
    ax.set(xlim=(0, 10), ylim=(0, 2.8))
    ax.axis("off")
    nodes = [
        (0.1, 1.0, 1.3, "2048 px image\nM0: disk"),
        (1.9, 1.9, 2.6, "Classical M1-M3\nBackground / smoothing\nThreshold"),
        (1.9, 0.2, 2.6, "U-Net (512 / 1024 px)\nUpsample probability\nThreshold"),
        (5.0, 1.0, 2.0, "Shared M4-M6\nClean / instances\nBoundary"),
        (7.6, 1.0, 2.2, "Per-annotator scores\nPooled PQ / errors\nBootstrap"),
    ]
    for x, y, w, label in nodes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, 0.65, boxstyle="round,pad=0.06", facecolor="#eaf2f5", edgecolor="#397287"
            )
        )
        ax.text(x + w / 2, y + 0.325, label, ha="center", va="center", fontsize=10)
    for start, end in [
        ((1.4, 1.35), (1.9, 2.15)),
        ((1.4, 1.35), (1.9, 0.55)),
        ((4.5, 2.15), (5.0, 1.4)),
        ((4.5, 0.55), (5.0, 1.2)),
        ((7.0, 1.3), (7.6, 1.3)),
    ]:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, color="#397287")
        )
    save(fig, args.out, "pipeline")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained")
    for index, (name, r) in enumerate(methods.items()):
        m = r["metrics"]
        ci = m["pq_ci"]
        axes[0].bar(
            index,
            m["pq"],
            color=COLORS[index],
            yerr=[[m["pq"] - ci["low"]], [ci["high"] - m["pq"]]],
            capsize=3,
        )
        axes[0].text(index, m["pq"] + 0.03, f"{m['pq']:.3f}", ha="center", fontsize=9)
        axes[1].plot(
            ["Small", "Medium", "Large"],
            [m["recall_by_area"][b]["recall"] for b in ("small", "medium", "large")],
            "o-",
            color=COLORS[index],
            label=name,
        )
    axes[0].set_xticks(range(4), methods, rotation=15)
    axes[0].set(
        ylabel="PQ",
        title="Held-out observations (95% bootstrap CI)",
        ylim=(0, min(1, max(r["metrics"]["pq"] for r in methods.values()) + 0.12)),
    )
    axes[1].set(ylabel="Instance recall", title="Recall by ground-truth area", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    save(fig, args.out, "main_results")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    sites = sorted(full["metrics"]["by_site"])
    for index, (name, r) in enumerate(list(methods.items())[:3]):
        site_metrics = r["metrics"]["by_site"]
        axes[0].plot(
            sites, [site_metrics[s]["pq"] for s in sites], "o-", color=COLORS[index], label=name
        )
        axes[1].plot(
            sites,
            [site_metrics[s]["recall_by_area"]["large"]["recall"] for s in sites],
            "o-",
            color=COLORS[index],
            label=name,
        )
    axes[0].set(ylabel="PQ", xlabel="Observatory", title="Main-test performance by site")
    axes[1].set(
        ylabel="Recall", xlabel="Observatory", title="Large-instance recall by site", ylim=(0, 1)
    )
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, args.out, "site_results")
    variants = [
        "no_background_correction",
        "single_threshold",
        "no_shape_filter",
        "connected_components",
        "no_boundary_expansion",
    ]
    for name in variants:
        actual = manifest["aliases"].get(name, name)
        ablated = full if actual == "full" else run(f"results/final/classic/{actual}/test")
        summary["ablations"][name] = {
            "alias": actual,
            "metrics": ablated["metrics"],
            "full_minus_ablated": paired(full, ablated),
            "delta_sq": full["metrics"]["sq"] - ablated["metrics"]["sq"],
            "delta_rq": full["metrics"]["rq"] - ablated["metrics"]["rq"],
            "error_changes": {
                key: errors(full["metrics"]).get(key, 0) - errors(ablated["metrics"]).get(key, 0)
                for key in set(errors(full["metrics"])) | set(errors(ablated["metrics"]))
            },
        }
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, (parameter, settings) in zip(axes, manifest["sensitivity"].items()):
        rows = []
        for setting in settings:
            name = manifest["aliases"].get(setting["name"], setting["name"])
            row = {"value": setting["value"]}
            for partition in ("tune", "test"):
                path = Path(f"results/final/classic/{name}/{partition}/metrics.json")
                if path.exists() or not args.allow_incomplete:
                    row[partition] = read(path)["pq"]
                else:
                    row[partition] = None
                    summary["pending_curve_points"].append(str(path))
            rows.append(row)
        summary["sensitivity"][parameter] = rows
        for partition, color in (("tune", COLORS[0]), ("test", COLORS[1])):
            ax.plot(
                [r["value"] for r in rows],
                [r[partition] if r[partition] is not None else np.nan for r in rows],
                "o-",
                label=partition,
                color=color,
            )
        ax.axvline(
            summary["classic_config"][parameter],
            color="#b44b40",
            ls="--",
            lw=1,
            label="frozen choice",
        )
        ax.set(xlabel=parameter.replace("_", " "), ylabel="PQ")
        ax.legend(fontsize=8)
    if summary["pending_curve_points"]:
        fig.suptitle(f"Draft: {len(summary['pending_curve_points'])} curve points pending")
    save(fig, args.out, "sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (side, categories) in zip(
        axes,
        (
            ("fn", ["miss", "fragmentation", "localization"]),
            ("fp", ["isolated", "overmerge", "fragment_related", "localization"]),
        ),
    ):
        x = np.arange(len(categories))
        for j, (name, r) in enumerate(list(methods.items())[:3]):
            bottom = np.zeros(len(categories))
            for bucket, alpha in (("small", 0.4), ("medium", 0.7), ("large", 1)):
                values = np.array(
                    [r["metrics"]["errors"].get(f"{side}/{c}/{bucket}", 0) for c in categories]
                )
                ax.bar(
                    x + (j - 1) * 0.25,
                    values,
                    0.23,
                    bottom=bottom,
                    color=COLORS[j],
                    alpha=alpha,
                    label=f"{name}" if bucket == "large" else None,
                )
                bottom += values
        ax.set_xticks(x, [c.replace("_", "\n") for c in categories], rotation=15)
        ax.set(title=f"{side.upper()} by category", ylabel="Annotation-comparison occurrences")
        ax.legend(fontsize=8)
    save(fig, args.out, "errors")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    for j, (name, r) in enumerate(list(methods.items())[:3]):
        ious = [iou for row in r["images"] for c in row["comparisons"] for iou in c["matched_ious"]]
        axes[0].hist(
            ious, bins=np.linspace(0.5, 1, 16), histtype="step", color=COLORS[j], label=name
        )
    axes[0].set(xlabel="Matched IoU", ylabel="Count", title="Strict IoU > 0.5 matches")
    axes[0].legend(fontsize=8)
    history = read("results/unet_e7a/history.json")
    axes[1].plot(
        [r["epoch"] for r in history], [r["validation"]["pq"] for r in history], color=COLORS[2]
    )
    axes[1].set(xlabel="Epoch", ylabel="Validation PQ", title="E7a checkpoint selection")
    summary["training"] = {
        "epochs": len(history),
        "best_epoch": max(history, key=lambda r: r["validation"]["pq"])["epoch"],
        "seconds": sum(r["seconds"] for r in history),
    }
    save(fig, args.out, "iou_training")
    for path in sorted(Path("results/final").iterdir()):
        if path.name.startswith(("site_", "unet_1024_")) and (path / "metrics.json").exists():
            summary["extensions"][path.name] = read(path / "metrics.json")
    summary["extension_analysis"] = summarize_extensions(summary["extensions"])
    if "unet_1024_L1_543" in summary["extensions"]:
        summary["extension_analysis"]["resolution_paired_seed_543"] = paired(
            run("results/final/unet_1024_L1_543"), learned
        )
    for seed in (543, 544, 545):
        if all(f"unet_1024_{loss}_{seed}" in summary["extensions"] for loss in ("L1", "L2")):
            summary["extension_analysis"].setdefault("loss_paired_by_seed", {})[str(seed)] = paired(
                run(f"results/final/unet_1024_L2_{seed}"), run(f"results/final/unet_1024_L1_{seed}")
            )
    extension_figures(summary, args.out)
    for name in ("baseline", "classical", "unet"):
        path = Path(f"results/benchmark_{name}.json")
        if path.exists():
            summary.setdefault("cpu_benchmark", {})[name] = read(path)
    manual = Path("results/manual_review.json")
    summary["manual_review"] = (
        read(manual) if manual.exists() else {"status": "Awaiting two independent human reviews"}
    )
    if not args.skip_examples:
        summary["examples"] = error_examples(full, args.out)
    write_json(args.out / "summary.json", summary)
    print(f"Wrote reproducible summaries and figures to {args.out}")


if __name__ == "__main__":
    main()
