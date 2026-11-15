"""Render a course report from measured E1-E7 summaries; never invent missing results."""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/report"))
    parser.add_argument("--out", type=Path, default=Path("output/pdf/final_report.pdf"))
    parser.add_argument(
        "--contributors", type=Path, help="Optional JSON list of {name, contribution}"
    )
    args = parser.parse_args()
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    methods = summary["methods"]
    base, full, unet = [methods[k]["metrics"] for k in ("Baseline", "Classical", "U-Net")]
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(name="Body10", fontName="Times-Roman", fontSize=10, leading=12, spaceAfter=7)
    )
    styles.add(
        ParagraphStyle(
            name="Caption9",
            fontName="Helvetica",
            fontSize=8.5,
            leading=10.5,
            spaceAfter=10,
            textColor=colors.HexColor("#354b59"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Title18",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=21,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles["Heading1"].fontSize, styles["Heading1"].leading = 14, 17
    styles["Heading1"].textColor = colors.HexColor("#154d62")
    styles["Heading2"].fontSize, styles["Heading2"].leading = 11, 14
    story = []

    def p(text, style="Body10"):
        story.append(Paragraph(text, styles[style]))

    def heading(text):
        p(text, "Heading1")

    def table(headers, rows, widths=None):
        content = [
            [Paragraph(escape(str(c)), styles["Caption9"]) for c in row] for row in [headers, *rows]
        ]
        t = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eff3")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#397287")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f7f8")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        story.extend([t, Spacer(1, 8)])

    def figure(name, caption, max_height=230):
        path = args.results / f"{name}.png"
        if not path.exists():
            return
        width, height = ImageReader(str(path)).getSize()
        scale = min(468 / width, max_height / height)
        story.append(Image(str(path), width=width * scale, height=height * scale))
        p(caption, "Caption9")

    def ci(value):
        return f"{value['estimate']:.4f} [{value['low']:.4f}, {value['high']:.4f}]"

    p(
        "How Far Can Classical Image Processing Go?<br/>Solar Filament Instance Segmentation",
        "Title18",
    )
    p(
        "CS543 / ECE549 Computer Vision - Fall 2026<br/>Reproducible evaluation on MAGFiLO v1.0",
        "Caption9",
    )
    heading("Abstract")
    p(
        f"We evaluate a modular classical image-processing pipeline for solar filament instance segmentation "
        f"on a frozen, observation-grouped split of MAGFiLO. On {full['observations']} held-out observations, "
        f"the untuned baseline obtains panoptic quality (PQ) {base['pq']:.4f}, the tuned classical pipeline "
        f"{full['pq']:.4f}, and a 512-pixel ResNet34 U-Net with validation-selected shared post-processing "
        f"{unet['pq']:.4f}. The classical improvement over baseline is "
        f"{ci(summary['paired']['classical_minus_baseline'])}; brackets denote an observation-level, "
        f"paired 95% bootstrap interval. We compare methods with held-out annotators under identical "
        "references, decompose unmatched instances by error type and size, and quantify false positives "
        "supported by another annotator. The contribution is a controlled instance-level evaluation, "
        "rather than a new detector. Non-expert human review remains separate from the principal scores."
    )
    heading("1. Problem and related work")
    p(
        "Solar filaments appear as dark, elongated structures in full-disk H-alpha images. Their fine "
        "branches, varying contrast and proximity to other dark structures make pixel classification "
        "and instance grouping distinct problems. A mask can cover approximately the correct pixels "
        "while splitting one annotated filament into several objects or merging neighboring filaments. "
        "We therefore measure PQ alongside foreground Dice, rather than treating a pixel score as a "
        "complete measure of instance segmentation quality."
    )
    p(
        "Classical extraction predates this project. Shih and Kowalski [3] combined intensity "
        "normalization, thresholding and morphological operations. Qu et al. [4] use image enhancement, "
        "threshold selection, SVM discrimination and morphological analysis. Fuller et al. [5] clean "
        "instrument artifacts before region growing and skeleton characterization. Bernasconi et al. [6] "
        "extend automated detection to chirality characterization and temporal tracking. These studies "
        "motivate the problem but are not reimplementations evaluated here. Our components use standard image-processing "
        "operations and are not claimed as algorithmic inventions. Modern learned alternatives include "
        "U-Net [7], semi-supervised filament detection [8], Flat U-Net [9] and EdgeAttNet [10]. We do not "
        "rank their published scores against ours because datasets, splits and evaluation targets differ."
    )
    p(
        "We study the effects of tuning and individual modules, local parameter stability and residual "
        "errors. U-Net with shared instance post-processing provides a learned reference; clDice [11] "
        "and leave-one-site experiments extend the analysis."
    )
    heading("2. Data and units of independence")
    p(
        "The downloaded MAGFiLO JSON contains 1,593 annotation sets, 10,244 polygons and 958 distinct "
        "2048-by-2048 GONG observations from six sites. There are 548 singly, 185 doubly and 225 triply "
        "annotated images; 154 annotation sets are empty. These counts are computed from the local "
        "index. COCO image IDs describe annotator passes, so splitting these IDs directly would leak "
        "the same photograph across partitions. We group by the physical image filename before "
        "stratifying jointly by site and annotation multiplicity."
    )
    p(
        "Seed 543 produces 671 tune and 287 test observations. Within tune, the learned model uses "
        "606 train and 65 validation observations. Every annotation of one photograph follows that "
        "photograph. These are local held-out results, not Kaggle leaderboard results. Images and "
        "annotations are bound to checksums; the original dataset is available from Harvard Dataverse [1,2]."
    )
    story.append(PageBreak())
    heading("3. Methods and selection protocol")
    table(
        ["Stage", "Operation and purpose"],
        [
            [
                "M0: solar disk",
                "Otsu contour/circle with 512-pixel Hough correction; inward radius margin removes rings and text.",
            ],
            [
                "M1: background",
                "Divide intensity by a constant, radial profile or upsampled median-filter background.",
            ],
            [
                "M2-M3: foreground",
                "Optional Gaussian smoothing followed by single threshold, hysteresis or Frangi-constrained hysteresis.",
            ],
            [
                "M4: cleaning",
                "Opening/closing, minimum component area and major/minor-axis ratio filtering.",
            ],
            [
                "M5: instances",
                "8-connected components; optional distance/orientation fragment merging; conservative watershed.",
            ],
            [
                "M6: boundary",
                "Nearest-label dilation without overlapping instance IDs, restricted to the solar disk.",
            ],
        ],
        [105, 363],
    )
    p(
        "Hough correction is used only when the centered circle prior disagrees materially with the "
        "Otsu fit; the prior is specific to standardized GONG full-disk images. Median-background "
        "estimation extends the nearest disk-edge intensities outside the mask to avoid introducing "
        "an artificial dark annulus. Fragment orientation is estimated by PCA of a skeleton and "
        "distance by nearest component boundaries. Merged fragments can retain disconnected pixels "
        "under one instance ID. The conservative watershed is triggered by area, requires connected "
        "input, and rejects splits producing pieces below the minimum area. The plan's additional "
        "branch-point trigger is not included in these measured configurations."
    )
    p(
        "Classical tuning uses two coordinate-search rounds on 150 randomly sampled tune observations. "
        "It re-evaluates the top five distinct configurations on all 671 tune observations and freezes "
        "the highest-PQ one. M0/M1 arrays are cached exactly, and four CPU workers process images. "
        "The sensitivity parameters are the three largest observed one-parameter PQ ranges in the "
        "sample-tune trials. Their grids and ablation configurations are frozen before test evaluation; "
        "test curves are descriptive and cannot choose a replacement model."
    )
    cfg = summary["classic_config"]
    settings = [
        ["Disk inset", cfg["disk_inset"]],
        ["Background / kernel", f"{cfg['background']} / {cfg['background_kernel']}"],
        ["Gaussian sigma", cfg["gaussian_sigma"]],
        ["Threshold mode", cfg["threshold_method"]],
        ["Seed / grow", f"{cfg['seed_threshold']} / {cfg['grow_threshold']}"],
        ["Single threshold", cfg["threshold"]],
        ["Area / axis ratio", f"{cfg['min_area']} / {cfg['min_axis_ratio']}"],
        ["Instances", cfg["instances"]],
        ["Merge distance / angle", f"{cfg['merge_distance']} / {cfg['merge_angle']}"],
        ["Dilation radius", cfg["dilation_radius"]],
        ["Opening / closing", f"{cfg['opening_radius']} / {cfg['closing_radius']}"],
        ["Search sample / rounds", "150 / 2"],
    ]
    table(
        ["Classical setting", "Value", "Classical setting", "Value"],
        [settings[i] + settings[i + 1] for i in range(0, len(settings), 2)],
        [136, 98, 136, 98],
    )
    p(
        "Inactive values are retained in the serialized configuration for reproducibility; for example, "
        "a single-threshold value has no effect when hysteresis is selected. An ablation changes only "
        "its named module and does not retune the remaining settings. Identical ablations explicitly "
        "reuse the full-model result."
    )
    story.append(PageBreak())
    heading("Learned reference and evaluation protocol")
    figure(
        "pipeline",
        "Figure 1. Foreground generation differs; instance conversion and evaluation share the same implementation.",
        125,
    )
    p(
        "E7a is an ImageNet-initialized ResNet34 U-Net with one input and one output channel. Each epoch "
        "selects one annotator per training image, then applies flips, right-angle rotations and modest "
        "brightness/contrast changes. Adam uses learning rate 0.0003, batch four, mixed precision, "
        "at most 50 epochs and patience eight on native-resolution validation PQ. Probabilities are "
        "upsampled to 2048 pixels before shared M4-M6 processing. Probability threshold and M4-M6 "
        f"parameters are selected only on validation; the selected threshold is {summary['unet_threshold']:.2f}. "
        "A connected-components-only control removes morphology, area/shape filtering and boundary expansion."
    )
    p(
        "Each prediction is compared separately with every annotation set of its observation. We "
        "pool TP, FP, FN and matched IoU sums before computing PQ = sum(IoU)/(TP + 0.5 FP + 0.5 FN), "
        "SQ = sum(IoU)/TP and RQ = TP/(TP + 0.5 FP + 0.5 FN). Empty annotation sets are retained, "
        "so all predictions against them are false positives. Foreground Dice pools pixel intersection "
        "and union counts across the same comparisons. Annotation sets are not treated as independent "
        "bootstrap units: 1,000 resamples draw observations and retain all their comparisons together."
    )
    p(
        "The strict IoU &gt; 0.5 hit-matrix rule matches the organizer's original scorer [12], including "
        "rare ambiguous matches from overlapping ground-truth polygons. The implementation was checked "
        "on 20 tune observations and an empty-set fixture: IoU arrays and aggregate PQ agree exactly. "
        "For the IoU &gt; 0.3 robustness check we use a one-to-one assignment maximizing eligible IoU. "
        "This lower-threshold score is auxiliary, not the selection objective."
    )
    p(
        "The principal comparisons use the frozen main split. Bootstrap intervals resample observations "
        "rather than individual annotator comparisons; paired differences use the same sampled IDs for "
        "both methods. Module ablations and sensitivity scans were frozen before test evaluation. "
        "Completed test evaluations are recorded in a registry and reused for identical configurations. "
        "Neither a test-score peak nor an unexpected ablation result changes the selected pipeline."
    )
    story.append(PageBreak())
    heading("4. Main results (E1, E7a)")
    table(
        ["Method", "PQ [95% CI]", "SQ", "RQ", "Dice", "PQ @ .3"],
        [
            [
                name,
                f"{m['pq']:.4f} [{m['pq_ci']['low']:.3f}, {m['pq_ci']['high']:.3f}]",
                f"{m['sq']:.3f}",
                f"{m['rq']:.3f}",
                f"{m['dice']:.3f}",
                f"{m['iou_0_3']['pq']:.3f}",
            ]
            for name, entry in methods.items()
            for m in [entry["metrics"]]
        ],
        [90, 140, 53, 53, 53, 79],
    )
    figure(
        "main_results",
        "Figure 2. Test PQ with observation-bootstrap uncertainty, and ground-truth instance recall. "
        "Area bins are &lt;600, 600-2,400 and &gt;2,400 native pixels.",
        205,
    )
    p(
        f"Relative to the tuned classical method, U-Net changes SQ by {unet['sq'] - full['sq']:+.4f} "
        f"and RQ by {unet['rq'] - full['rq']:+.4f}; the corresponding PQ difference is "
        f"{summary['paired']['unet_minus_classical']['estimate']:+.4f}. These components separate "
        "successful-boundary agreement from instance recognition, although SQ conditions on a "
        "different set of matched instances for each method and is not a controlled boundary-only comparison."
    )
    table(
        ["Paired difference", "Estimate and 95% CI"],
        [[name.replace("_", " "), ci(value)] for name, value in summary["paired"].items()],
        [250, 218],
    )
    p("Human context", "Heading2")
    p(
        "Human-machine comparison uses only multiply annotated test observations. Each annotator is "
        "held out in turn; that annotator and the method are compared against every remaining annotator, "
        "with identical image/reference multiplicities. This avoids interpreting an overall inter-annotator "
        "PQ near 0.33 as a mathematical ceiling or using dissimilar reference sets to claim human-level performance."
    )
    table(
        ["Method", "Same-reference human PQ", "Method PQ", "Method - human [95% CI]"],
        [
            [
                name,
                f"{m['same_reference']['human']['pq']:.4f}",
                f"{m['same_reference']['method']['pq']:.4f}",
                ci(m["same_reference"]["method_minus_human"]),
            ]
            for name, m in (("Classical", full), ("U-Net", unet))
        ],
        [80, 117, 80, 191],
    )
    story.append(PageBreak())
    heading("Site breakdown and training")
    figure(
        "site_results",
        "Figure 3. Main-test site breakdown. These observations belong to the ordinary mixed-site split; "
        "they are not the leave-one-site-out experiments in Section 9.",
        165,
    )
    figure(
        "iou_training",
        "Figure 4. Matched-instance IoU distributions and the E7a validation-PQ trajectory. "
        "The histogram conditions on successful matches and cannot characterize missed instances.",
        165,
    )
    p(
        f"E7a completed {summary['training']['epochs']} epochs; its selected checkpoint is epoch "
        f"{summary['training']['best_epoch']}. Logged epoch computation totals approximately "
        f"{summary['training']['seconds'] / 60:.1f} minutes on the RTX 4060 Laptop. This excludes process "
        "startup and some checkpoint IO and is not an isolated throughput benchmark. Concurrent classical "
        "workers can affect wall-clock times. The validation set is reused for checkpoint and post-processing "
        "selection, so validation PQ is optimistic and the untouched test score remains the main estimate."
    )
    table(
        ["Site", "Comparisons", "Classical FN", "U-Net FN", "Classical FP", "U-Net FP"],
        [
            [
                site,
                full["by_site"][site]["comparisons"],
                *[
                    f"{m['by_site'][site][side] / m['by_site'][site]['comparisons']:.2f}"
                    for side in ("fn", "fp")
                    for m in (full, unet)
                ],
            ]
            for site in sorted(full["by_site"])
        ],
        [48, 84, 84, 84, 84, 84],
    )
    p(
        "FN and FP columns give instances per annotation comparison, accounting for unequal site sample "
        "sizes and annotation multiplicities. Category-by-area-by-site counts are retained in the "
        "machine-readable metrics. These descriptive site differences do not isolate instrument effects.",
        "Caption9",
    )

    story.append(PageBreak())
    heading("5. Ablations and sensitivity (E2-E3)")
    p(
        "Positive differences below mean that the full classical method is better than the indicated "
        "ablated variant. All differences pool counts on the same test observations; confidence intervals "
        "use paired observation bootstrap. These exploratory module comparisons have pointwise intervals "
        "without multiple-comparison correction. They describe the frozen full configuration and do not "
        "establish a module's universal benefit under separately reoptimized pipelines. "
        "The differences are not additive because the modules interact."
    )
    table(
        ["Replaced/removed component", "Full - ablated PQ [95% CI]", "Delta SQ", "Delta RQ"],
        [
            [
                name.replace("_", " "),
                ci(v["full_minus_ablated"]),
                f"{v['delta_sq']:+.4f}",
                f"{v['delta_rq']:+.4f}",
            ]
            for name, v in summary["ablations"].items()
        ],
        [152, 188, 64, 64],
    )
    pending_points = summary.get("pending_curve_points", [])
    if pending_points:
        p(
            f"<b>Draft: {len(pending_points)} curve points are still running.</b> "
            "Missing values are left blank, not interpolated or replaced by estimates."
        )
    p(
        "The exact parameter-selection ranking and grids are saved in the frozen manifest. Completed "
        "curve points evaluate all 671 tune or all 287 test observations for each preregistered value. "
        "A test peak away from the frozen value is an observation about transfer of the tuning choice; "
        "it is not used to select another test-winning configuration. An unchanged curve can reflect "
        "discrete component decisions that are locally insensitive to a parameter."
    )
    strongest = max(
        summary["ablations"],
        key=lambda k: summary["ablations"][k]["full_minus_ablated"]["estimate"],
    )
    p(
        f"The largest observed full-minus-ablated PQ difference is for "
        f"{escape(strongest.replace('_', ' '))}: {ci(summary['ablations'][strongest]['full_minus_ablated'])}. "
        "The complete table includes negative and null effects as well as improvements; no result is removed "
        "because it contradicts the intended purpose of a component."
    )
    figure(
        "sensitivity",
        "Figure 5. Parameter sensitivity. Dashed lines indicate the choices made using tuning data only.",
        210,
    )
    error_keys = [
        "fn/miss",
        "fn/fragmentation",
        "fn/localization",
        "fp/isolated",
        "fp/overmerge",
        "fp/fragment_related",
        "fp/localization",
    ]
    table(
        [
            "Full minus variant",
            "FN miss",
            "FN frag.",
            "FN loc.",
            "FP isol.",
            "FP merge",
            "FP frag.",
            "FP loc.",
        ],
        [
            [name.replace("_", " "), *[f"{v['error_changes'].get(k, 0):+d}" for k in error_keys]]
            for name, v in summary["ablations"].items()
        ],
        [132, 48, 48, 48, 48, 48, 48, 48],
    )
    p(
        "Error-count changes use annotation-comparison occurrences. Negative values mean the full "
        "pipeline has fewer errors in that category; these changes are not independent causal effects.",
        "Caption9",
    )

    story.append(PageBreak())
    heading("6. Error decomposition (E4)")
    p(
        "Error categories cover every unmatched ground-truth instance and prediction exactly once. "
        "On the FN side, zero overlap is a miss, overlap with two or more predictions is fragmentation, "
        "and one insufficient-overlap prediction is localization error. On the FP side, zero overlap "
        "is isolated error and overlap with multiple ground truths is overmerging. A remaining FP is "
        "fragment-related if its sole ground truth overlaps multiple predictions, otherwise localization. "
        "Overlaps use IoU &gt; 0; categories are operational definitions, not independent physical causes."
    )
    figure(
        "errors",
        "Figure 6. Error counts by category. Bar shade encodes small (light), medium and large (dark) "
        "instance areas. FN and FP sides are separate; a fragmented object can contribute to both.",
        255,
    )
    table(
        ["Error category", "Baseline", "Classical", "U-Net"],
        [
            [
                key,
                *[
                    methods[name]["errors"].get(key, 0)
                    for name in ("Baseline", "Classical", "U-Net")
                ],
            ]
            for key in sorted(
                set().union(
                    *(methods[name]["errors"] for name in ("Baseline", "Classical", "U-Net"))
                )
            )
        ],
        [222, 82, 82, 82],
    )
    large = full["recall_by_area"]["large"]
    error_counts = methods["Classical"]["errors"]
    dominant_fn = max((k for k in error_counts if k.startswith("fn/")), key=error_counts.get)
    dominant_fp = max((k for k in error_counts if k.startswith("fp/")), key=error_counts.get)
    p(
        f"For the final classical method, the most frequent FN category is "
        f"{dominant_fn.split('/')[1].replace('_', ' ')} ({error_counts[dominant_fn]} occurrences), "
        f"and the most frequent FP category is {dominant_fp.split('/')[1].replace('_', ' ')} "
        f"({error_counts[dominant_fp]} occurrences). These counts identify the largest residual categories "
        "under the stated annotation-dependent taxonomy."
    )
    p(
        f"The classical method matches {large['matched']} of {large['total']} large ground-truth "
        f"instances across annotation comparisons (recall {large['recall']:.3f}). Repeated annotator "
        "comparisons count repeatedly in these totals, as in the principal metric. This measure "
        "quantifies detection of large structures but does not establish utility for forecasting solar "
        "events: no eruption outcomes or operational decision task are evaluated."
    )
    p(
        "Qualitative examples on the following pages are sampled with fixed seeds within each observed "
        "error category. They are illustrations rather than additional selection data. The manifest "
        "stores observation IDs, annotation IDs and instance indices so every crop can be reconstructed. "
        "Different annotators may assign the same prediction different error categories, which is one "
        "reason not to interpret a single overlay as complete ground truth."
    )

    story.append(PageBreak())
    heading("7. Representative error crops: missed instances")
    figure(
        "error_examples_fn",
        "Figure 7a. Two FN examples per category. Cyan outlines denote the chosen "
        "reference annotation and orange outlines the prediction; thicker contours mark the sampled instance. "
        "Rows separate misses, fragmentation and localization errors. These figures "
        "are excluded from the independent reviewer packages, which show no reference annotation.",
        540,
    )
    story.append(PageBreak())
    heading("Representative error crops: unmatched predictions")
    figure(
        "error_examples_fp",
        "Figure 7b. Two FP examples per category. Cyan denotes the reference and orange the prediction; "
        "thicker contours identify the sampled prediction. Rows separate overmerge, isolated, "
        "fragment-related and localization errors. The sample manifest preserves exact instance IDs.",
        550,
    )

    story.append(PageBreak())
    heading("8. Annotation noise and limitations (E5)")
    rows = []
    for name, m in (("Classical", full), ("U-Net", unet)):
        noise = m["annotation_noise"]
        rate = noise["corroborated"] / noise["false_positives"] if noise["false_positives"] else 0
        rows.append([name, noise["false_positives"], noise["corroborated"], f"{rate:.1%}"])
    table(
        ["Method", "Multi-annotator FP", "Supported by another", "Fraction"],
        rows,
        [95, 120, 155, 98],
    )
    p(
        "The objective statistic counts a prediction unmatched against one reference when it has IoU "
        "greater than 0.5 with at least one different annotator's instance. Its denominator is FP "
        "annotation-comparison occurrences on multiply annotated observations, not all test false "
        "positives or unique predictions. This is evidence of reference dependence; it is not proof "
        "that every corroborated prediction is physically correct. The strict overlap criterion can "
        "also miss cases in which annotators agree on a structure but disagree on its boundary."
    )
    manual = summary["manual_review"]
    if "status" in manual:
        p(
            "<b>Human auxiliary review is pending.</b> The software samples 100 unique isolated false "
            "positives and prepares two independent, anonymous reviewer packages. Each contains an "
            "original crop, candidate outline and four labels: possible filament, sunspot, limb artifact "
            "or other. No human ratings or agreement numbers have been generated by the program. "
            "The principal objective results above are complete independently of this auxiliary step."
        )
    else:
        p(
            f"Two independent reviews cover {manual['samples']} samples. Agreement is {manual['agreement']:.1%}; "
            f"Cohen's kappa is {manual['cohen_kappa']}. Category counts and FP-exclusion scenarios are "
            "stored in the accompanying machine-readable summary. These are non-expert sensitivity "
            "scenarios, not corrected ground truth or confidence bounds."
        )
    p(
        "Sampling is uniform over unique predicted instances that are isolated against at least one "
        "annotation. The sensitivity tool retains each sample's isolated-FP comparison multiplicity "
        "and expands possible-filament judgments to the sampling population. It considers separate "
        "scenarios where both reviewers or either reviewer mark a possible filament, removes only "
        "the estimated FP penalties, and leaves TP, FN and IoU sums fixed. Sampling uncertainty and "
        "expert correctness are not implied by these two scenarios."
    )
    heading("Discussion")
    p(
        "PQ combines recognition with the quality of successful boundaries. Changes in SQ, RQ and "
        "error counts should be interpreted together: improving foreground overlap does not guarantee "
        "better grouping of instances. A compact isolated dark region can represent an unannotated "
        "filament, a sunspot or noise, and visual plausibility alone cannot establish its label. "
        "The same-reference comparison provides a more defensible human context than a single "
        "dataset-wide agreement number, while still measuring consistency with available annotations."
    )
    p(
        "Several limits remain. Random observation splitting does not enforce temporal separation or "
        "prevent related solar structures from appearing on different dates/sites. The disk prior is "
        "specialized to GONG. The classical search is coordinate-based, not globally exhaustive. "
        "Only one primary U-Net seed is used in E7a; extension seeds cannot substitute for an "
        "independent dataset. Annotation sets are incomplete and potentially correlated. Pointwise "
        "bootstrap intervals quantify variation under this split and reference collection, not all "
        "sources of measurement and selection uncertainty. Nothing here establishes a causal "
        "relationship to space-weather prediction quality."
    )
    if summary.get("cpu_benchmark"):
        table(
            ["Isolated CPU latency", "Median seconds", "Maximum seconds"],
            [
                [name, f"{b['median']:.3f}", f"{b['max']:.3f}"]
                for name, b in summary["cpu_benchmark"].items()
            ],
            [230, 119, 119],
        )
        p(
            "CPU latency uses 50 fixed tune observations, one numerical thread, no preprocessing cache "
            "and three warmups. Image IO and scoring are excluded; M0 through output RLE are included."
        )
    else:
        p(
            "Isolated CPU latency has not yet been added to this report snapshot. Per-run timings "
            "include concurrent worker load and are not substituted for a controlled CPU benchmark."
        )

    story.append(PageBreak())
    heading("9. Extensions: resolution, topology and held-out sites")
    extensions = summary["extensions"]
    rows = []
    for loss in ("L1", "L2"):
        entries = [extensions.get(f"unet_1024_{loss}_{seed}") for seed in (543, 544, 545)]
        if all(entries):
            scores = [m["pq"] for m in entries]
            fragments = [
                sum(v for k, v in m["errors"].items() if k.startswith("fn/fragmentation/"))
                for m in entries
            ]
            rows.append(
                [
                    loss,
                    f"{np.mean(scores):.4f} +/- {np.std(scores, ddof=1):.4f}",
                    f"{np.mean(fragments):.1f} +/- {np.std(fragments, ddof=1):.1f}",
                ]
            )
    if rows:
        table(
            ["1024-pixel loss", "PQ mean +/- sample SD", "FN fragmentation mean +/- SD"],
            rows,
            [110, 175, 183],
        )
    else:
        p(
            "The three-seed 1024-pixel extension is not yet complete in this report snapshot. "
            "No missing seed is replaced by an assumed score or reported as a completed comparison."
        )
    p(
        "The fixed E7b recipe compares Dice+BCE (L1) with L1 plus 0.5 soft-clDice (L2), using seeds "
        "543, 544 and 545 and ten soft-skeleton iterations. Both losses use true batch four, "
        "matching E7a. A batch-four capacity test including L2 peaked at approximately 7.0 GiB "
        "reserved GPU memory. An earlier gradient-accumulation prototype was archived and excluded "
        "before any E7b test evaluation; its scores do not enter the formal comparisons. "
        "Every checkpoint and post-processing recipe is selected on its own validation split."
    )
    rows = []
    for site in ("Bh", "Ch", "Lh", "Mh", "Th", "Uh"):
        c, u = extensions.get(f"site_classic_{site}"), extensions.get(f"site_unet_{site}")
        main_c = extensions.get("site_classic_main")
        if c and u and main_c:
            rows.append(
                [
                    site,
                    f"{main_c['by_site'][site]['pq']:.4f}",
                    f"{c['pq']:.4f}",
                    f"{unet['by_site'][site]['pq']:.4f}",
                    f"{u['pq']:.4f}",
                ]
            )
    if rows:
        table(
            ["Site", "Classical main", "Classical held out", "U-Net main", "U-Net held out"],
            rows,
            [48, 105, 105, 105, 105],
        )
    else:
        p(
            "The six-site paired extension is not yet complete in this snapshot. The frozen queue "
            "records completed fold outputs separately and does not infer unrun folds."
        )
    p(
        "Each E6 fold excludes one site from all training and parameter selection. The classical "
        "recipe fixes the untuned baseline modules and retunes only threshold and area, avoiding "
        "transfer of main-split settings selected with the excluded site's data. Its main-split "
        "comparator uses the same restricted recipe. U-Net retrains from ImageNet for each fold. "
        "Site-held-out tests contain all observations from that site, whereas the main-split site "
        "subset is smaller. Their difference is descriptive and combines domain exclusion with "
        "different evaluation samples and training sizes; it is not a paired causal domain-shift estimate."
    )
    p(
        "Shared preprocessing was developed before these folds using main-tune images from all six "
        "sites. E6 therefore evaluates fitting and parameter-selection transfer within a fixed "
        "design, rather than a prospectively blinded study on an entirely unknown instrument."
    )
    heading("Reproducibility and contributions")
    p(
        "The repository stores code, configuration recipes, frozen split IDs and a checksum manifest. "
        "Per-experiment outputs include provenance, raw per-image records, predictions, aggregate "
        "metrics and observation bootstrap intervals. Resumable queues skip completed results, "
        "save training optimizer/scaler/RNG state and reuse identical test configurations. Raw data "
        "and model weights remain local; the public sources and download tools allow reconstruction. "
        "MAGFiLO is distributed under CC BY-NC 4.0; NSO/AURA/NSF/GONG image attribution is retained. "
        "The official clDice MIT notice is included in the source distribution."
    )
    if args.contributors:
        contributors = json.loads(args.contributors.read_text(encoding="utf-8"))
        for member in contributors:
            p(f"<b>{escape(member['name'])}</b>: {escape(member['contribution'])}")
    else:
        p(
            "<b>Member names and individual contributions await confirmation.</b> Automated assistance "
            "was used for implementation, experiment execution and report assembly. Team members must "
            "verify the results, complete independent human review and describe their actual contributions "
            "before course submission."
        )

    story.append(PageBreak())
    heading("References and artifact index")
    references = [
        (
            "Ahmadzadeh et al. (2024). A dataset of manually annotated filaments from H-alpha observations. Scientific Data 11, 1031.",
            "https://doi.org/10.1038/s41597-024-03876-y",
        ),
        (
            "MAGFiLO v1.0. Harvard Dataverse. Dataset and annotation download.",
            "https://doi.org/10.7910/DVN/J6JNVK",
        ),
        (
            "Shih and Kowalski (2003). Automatic extraction of filaments in H-alpha solar images. Solar Physics 218, 99-122.",
            "https://digitalcommons.njit.edu/fac_pubs/14319/",
        ),
        (
            "Qu et al. (2005). Automatic solar filament detection using image processing techniques. Solar Physics 228, 119-135.",
            "https://doi.org/10.1007/s11207-005-5780-1",
        ),
        (
            "Fuller, Aboudarham and Bentley (2005). Filament recognition and image cleaning on Meudon H-alpha spectroheliograms. Solar Physics 227, 61-73.",
            "https://doi.org/10.1007/s11207-005-8364-1",
        ),
        (
            "Bernasconi, Rust and Hakim (2005). Advanced Automated Solar Filament Detection and Characterization Code: Description, Performance, and Results. Solar Physics 228, 97-117.",
            "https://doi.org/10.1007/s11207-005-2766-y",
        ),
        (
            "Ronneberger, Fischer and Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.",
            "https://arxiv.org/abs/1505.04597",
        ),
        (
            "Diercke et al. (2024). A Universal Method for Solar Filament Detection from H-alpha Observations using Semi-supervised Deep Learning.",
            "https://arxiv.org/abs/2402.15407",
        ),
        (
            "Zhu et al. (2025). Flat U-Net: An Efficient Ultralightweight Model for Solar Filament Segmentation in Full-disk H-alpha Images.",
            "https://arxiv.org/abs/2502.07259",
        ),
        (
            "Solomon et al. (2025). EdgeAttNet: Towards Barb-Aware Filament Segmentation.",
            "https://arxiv.org/abs/2509.02964",
        ),
        (
            "Shit et al. (CVPR 2021). clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.",
            "https://arxiv.org/abs/2003.07311",
        ),
        (
            "Ahmadzadeh. Solar Filament Segmentation Challenge: Self-Evaluation Notebook.",
            "https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook",
        ),
        ("Kirillov et al. (CVPR 2019). Panoptic Segmentation.", "https://arxiv.org/abs/1801.00868"),
        (
            "segmentation-models-pytorch. U-Net implementation and pretrained encoder interface.",
            "https://github.com/qubvel-org/segmentation_models.pytorch",
        ),
        ("NSO GONG H-alpha image archive.", "https://gong2.nso.edu/HA/hag/"),
        ("Project source repository: filament-pq.", "https://github.com/zjy011/filament-pq"),
    ]
    for index, (title, url) in enumerate(references, 1):
        p(
            f"[{index}] {escape(title)} <link href='{escape(url)}' color='#155f7a'>{escape(url)}</link>",
            "Caption9",
        )
    p(
        "Key local artifacts: configs/study.yaml (fixed recipes); configs/frozen_classic/manifest.json "
        "(E1-E3 freeze); results/report/summary.json (all report numbers and source paths); "
        "results/final/ (immutable evaluation outputs); results/blind_review_100/ (two independent "
        "reviewer packages); docs/RUNNING_EXPERIMENTS.md (commands and resumption). The original "
        "proposal and plan remain available as design documents rather than claims of completed results."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5d6c73"))
        canvas.drawString(
            72, 35, "CS543 | MAGFiLO instance segmentation | Reproducible experimental report"
        )
        canvas.drawRightString(540, 35, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(args.out),
        pagesize=(612, 792),
        leftMargin=72,
        rightMargin=72,
        topMargin=48,
        bottomMargin=52,
        title="Solar Filament Instance Segmentation",
        author="CS543 project team (names pending confirmation)",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.out)


if __name__ == "__main__":
    main()
