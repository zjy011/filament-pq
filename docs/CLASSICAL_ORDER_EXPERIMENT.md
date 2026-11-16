# Classical filtering-order follow-up

This follow-up tests a hypothesis from the 150-observation tuning diagnostic:
small foreground components may be discarded before they can form a usable
filament instance. It does not change the original frozen main-test result.

## Fixed design

All variants use `configs/frozen_classic/full.yaml` for disk masking, background
correction, smoothing, candidate extraction, merge distance/angle and final
boundary expansion. Only the size/shape filtering rules change. No ground-truth
information enters the segmentation functions.

The twelve recipes comprise the original pipeline, the original without shape
filtering, and five recipes at each provisional component floor of 150 and 300
native pixels:

- `relax`: lower the pre-merge area threshold; retain pre-merge shape filtering.
- `relax_no_shape`: lower the threshold and omit shape filtering.
- `late_area`: retain pre-merge shape filtering, merge smaller components, then
  require each grouped instance to contain at least 600 pixels.
- `late_both`: merge smaller components without a shape restriction, then require
  grouped area >= 600 and grouped axis ratio >= 2.
- `late_area_no_shape`: merge smaller components, require grouped area >= 600,
  and omit shape filtering.

Area counts include all disconnected fragments assigned to the same instance.
Grouped shape uses the moments of those fragments together. Final one-pixel
expansion occurs after filtering, so expansion cannot rescue an undersized group.
Groups are not merged again after filtering.

## Evaluation and selection

First evaluate all recipes on the same seed-543 sample of 150 tuning observations
used by the stage diagnostic. Compare PQ, TP/FP/FN, error categories, size-stratified
recall and paired observation-bootstrap differences. Bootstrap intervals are
pointwise descriptive intervals, without correction for recipe selection or
multiple comparisons.

Before inspecting the completed pilot, the follow-up rule is: if any delayed-area
recipe improves pilot PQ over the original, evaluate the best delayed-area recipe
on the remaining 521 tuning observations, together with the original, original
without shape filtering, its matching relaxed-threshold control, and the relevant
same-floor shape-placement controls. Also retain the best relaxed-threshold recipe
if it is not already included. No settings are retuned on the remainder.
If no delayed-area recipe improves, retain the negative result and inspect the
TP/FP tradeoff before expanding that hypothesis.

The 521-observation remainder is additional tuning evidence, **not an untouched
test set**: it belongs to the partition used for the original classical search.
Combined 671-observation results are reported only for recipes evaluated on both
disjoint cohorts. Main-test scores from the original study remain separate.

## Reproduction

```powershell
python -m scripts.resilient_io scripts.classic_order_experiment --out results/classic_order_pilot --workers 4
python -m scripts.analyze_classic_order --runs results/classic_order_pilot
```

Add `--resume` to continue an interrupted run. Its manifest fixes image IDs,
recipes, input/configuration identity and implementation digests before evaluation.
Completed observations are saved separately. The comparison implementation has
tests for baseline identity, rescue of individually undersized fragments,
shape checks after grouping and noncontiguous instance IDs.

All operations are classical CPU processing; this follow-up does not train U-Net.

## Completed pilot findings

On the fixed 150-image pilot, the original recipe achieves PQ 0.241569.
Lowering the area threshold to 150 gives PQ 0.091929: TP rises from 502 to 730,
but FP rises from 590 to 7,840. Lowering it to 300 gives PQ 0.213089, with 706 TP
and 1,975 FP. Small-instance recall rises from 6.25% to 34.78% at threshold 300,
showing a substantial recall/false-positive tradeoff under the existing annotations.
These are tuning results, not replacements for the main-test score 0.231352.

The best delayed-area pilot recipe, `late_area_300`, gives PQ 0.245918,
a difference of +0.004349 with a paired 95% interval [-0.002302, 0.010733].
It gains 36 matched annotation instances and loses 11 previously matched
instances, for a net gain of 25. FP increases from 590 to 648. The interval
includes zero; the pilot alone does not establish improvement over the original.

`late_area_300` exceeds the simple `relax_300` control by 0.032829 PQ,
95% interval [0.019644, 0.045413], supporting the usefulness of a final area
check when smaller components are admitted before merging. Moving shape checks
after merging does not clearly improve upon `late_area_300` in this pilot.

Following the fixed rule, six recipes were selected for the complementary 521
tuning observations: `original`, `original_no_shape`, `relax_300`,
`late_area_300`, `late_both_300`, and `late_area_no_shape_300`.

Before the remainder finishes, the gate for a possible single exploratory
main-test follow-up is fixed: the pilot-selected `late_area_300` must improve
PQ on the remainder and its combined-tune paired 95% interval against the original
must exclude zero on the positive side. Otherwise retain the negative/inconclusive
result without a new test call. Passing this gate would still not make the already
inspected main test prospectively blind or establish equivalence to U-Net.

```powershell
python -m scripts.resilient_io scripts.classic_order_experiment --cohort remainder --variants original original_no_shape relax_300 late_area_300 late_both_300 late_area_no_shape_300 --out results/classic_order_remainder --workers 4
python -m scripts.analyze_classic_order --runs results/classic_order_pilot results/classic_order_remainder
```

## Completed remainder and combined tuning findings

On the complementary 521 observations, PQ is 0.226058 for the original and
0.229121 for `late_area_300`: difference +0.003063, paired 95% interval
[-0.000524, 0.006689]. The direction reproduces, but this cohort's interval
includes zero.

Across all 671 tuning observations, original PQ is 0.229454 and delayed-area
PQ is 0.232794: difference +0.003340, paired 95% interval [0.000177, 0.006896].
The combined interval narrowly excludes zero, and does not account for adaptive
selection or multiple comparisons. The effect is small. All 671 original-control
per-image count records exactly reproduce the previously saved original run.

In the combined cohort, delayed-area filtering gains 149 matched annotation
instances and loses 42, a net gain of 107. TP rises from 2,177 to 2,284, FP from
2,992 to 3,279, and FN falls from 4,972 to 4,865. Its advantage over merely lowering
the area threshold to 300 is +0.036098 PQ [0.030123, 0.042324]. Moving both area
and shape checks after merging reaches 0.231402; removing shape filtering from
the delayed-area recipe reaches 0.227607. Thus changing filter order alone has
not produced a large performance gain.

The recorded gate passes for the pilot-selected `late_area_300`. Its exact recipe
and implementation hashes are frozen before a single exploratory test follow-up
under `results/followup/classic_late_area_300/`. A separate receipt prevents reruns
of that frozen recipe, and the original formal test outputs remain unchanged.

## Completed exploratory test: no improvement

The single frozen follow-up on all 287 main-test observations obtains PQ
**0.231251**, compared with **0.231352** for the original. The paired difference
is **-0.000101**, 95% interval **[-0.004373, 0.004451]**. This does not support
an improvement; the original pipeline remains the primary method.

TP rises from 948 to 993, FP from 1,319 to 1,499, and FN falls from 2,147 to 2,102.
Zero-overlap misses fall from 1,492 to 1,362, but localization FN rise from 511
to 590 and fragmentation FN from 144 to 150. Recovering some pixel overlap is
therefore not the same as recovering a valid instance match. Small-instance
recall changes from 5.13% to 4.68%, medium-instance recall from 36.51% to 38.97%,
and large-instance recall from 39.75% to 40.75%. SQ slightly declines from
0.654277 to 0.650555 while RQ slightly rises from 0.353599 to 0.355468.

Against the saved 512 U-Net, the new pipeline's paired PQ difference is
-0.088135 [-0.102378, -0.074090]. The gap is not closed. This test was already
inspected during the original study, so this follow-up is explicitly exploratory.
No other new recipe was tested and no parameters were changed after this result.

The experiment supports a narrower conclusion than the initial stage diagnostic:
size filtering removes both annotated filaments and many predictions unmatched
to the references. Admitting smaller fragments increases coverage, but the current
grouping/filtering rules do not separate useful additions from harmful ones well
enough to improve final PQ. The next hypothesis should concern candidate
discrimination (for example local contrast and scale-dependent shape evidence),
not assume that changing filter order is sufficient. Such a hypothesis has not
yet been experimentally validated.

```powershell
python -m scripts.resilient_io scripts.run_order_followup --workers 4
```

The command returns the saved completed result on repeat invocation. Use
`--resume` only for a genuinely interrupted evaluation.

## What this says about the U-Net gap

A separate arithmetic diagnostic uses the original classical test counts:
948 TP, 1,319 FP, 2,147 FN and matched-IoU sum 620.255009. Holding TP, FN and
matched IoUs fixed while hypothetically removing every FP penalty gives
`620.255009 / (948 + 0.5 * 2147) = 0.306829` PQ, below the 512 U-Net's 0.319386.
This is an unattainable reference-aware diagnostic, not an achieved score or an
upper bound on classical methods generally. It shows that suppressing unmatched
predictions alone, without recovering instances or improving overlaps, cannot
close the observed point-estimate gap for the current fixed predictions.

New FP counts remain reference-dependent. The original 100-case blind-review
sample describes the original pipeline's sampled error population; it cannot
automatically be extrapolated to these new smaller-component predictions.

## Saved evidence

- [Tuning summaries and paired contrasts](evidence/classic_order/analysis.json)
- [Exploratory test metrics](evidence/classic_order/exploratory_test/metrics.json)
- [Frozen test recipe and provenance](evidence/classic_order/exploratory_test/provenance.json)
- [Tuning comparison figure](figures/classic_order_comparison.png)

Raw per-observation evaluation records remain under local `results/`. The exported
evidence manifest records hashes for the aggregate files and figure. No raw images,
trained model weights or human-review keys are included in this export.
