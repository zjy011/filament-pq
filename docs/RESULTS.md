# Measured results

2026-11-15. This page distinguishes completed held-out results from ongoing work.
The experiment recipes are fixed in `configs/study.yaml`.

## E1: completed main classical comparison

The full classical search ran 65 distinct configurations in two coordinate rounds
on 150 tune observations, then reranked five candidates on all 671 tune observations.
The winner's full-tune PQ is 0.229454. Frozen E1/E2/E3 configurations and the
sensitivity selection rule are recorded in `configs/frozen_classic/manifest.json`.

On the 287 held-out observations:

- Untuned baseline: **PQ 0.127786**, 95% CI [0.114653, 0.141833].
- Tuned classical pipeline: **PQ 0.231352**, 95% CI [0.216677, 0.245503].
  SQ 0.654277; RQ 0.353599; Dice 0.540362.
- Classical minus baseline: **+0.103566 PQ**, paired 95% CI [0.087714, 0.119193].
- U-Net minus classical: **+0.088034 PQ**, paired 95% CI [0.074461, 0.102020].

The final classical configuration also matches the original organizer evaluator
exactly on 20 tune images plus an empty-set fixture: maximum IoU difference 0,
both PQ values 0.21464073608729464. This check uses the current GPU-enabled
environment's Torch installation and runs the reference scorer on CPU.

Evidence: [baseline](evidence/baseline_test.json), [classical](evidence/classical_test.json),
[classical provenance](evidence/classical_provenance.json),
[classical improvement](evidence/classical_improvement.json),
[U-Net comparison](evidence/unet_vs_classic.json),
[official check](evidence/official_validation_final.json).

## E7a: completed 512-pixel U-Net reference

The ResNet34/ImageNet U-Net trained on 606 observations and selected its checkpoint
using 65 validation observations. Training stopped after epoch 22; epoch 14 was
best (validation PQ 0.347752). Two rounds of validation-only post-processing search
evaluated 51 distinct configurations, selecting probability threshold 0.5, opening
2, closing 2, minimum area 600, no axis-ratio filtering, fragment merging at distance
20 / angle 40, and no boundary dilation. Validation PQ became 0.354766.

The frozen model was evaluated once on all 287 main-test observations:

- PQ **0.319386**, observation-bootstrap 95% CI **[0.305430, 0.333958]**.
- SQ 0.642543; RQ 0.497066; pooled foreground Dice 0.656518.
- The same checkpoint with connected components only obtains PQ 0.251144.
  Shared post-processing adds **0.068243 PQ**, paired 95% CI **[0.060372, 0.076057]**.
- On the 123 multiply annotated test observations under identical references,
  human PQ is 0.326828 and method PQ is 0.323527. The paired method-minus-human
  difference is -0.003301, 95% CI [-0.025216, 0.017509]. This does not establish
  either superiority or equivalence.
- Among 801 FP annotation-comparison occurrences on multiply annotated images,
  296 (37.0%) match at least one other annotator at IoU > 0.5. This is evidence
  of reference dependence, not corrected expert ground truth.

Evidence: [metrics](evidence/e7a_test.json),
[provenance including checkpoint digest](evidence/e7a_provenance.json),
[paired post-processing comparison](evidence/e7a_postprocess_effect.json),
[selected post-processing configuration](../configs/unet_e7a_selected.yaml).
Model weights and per-image outputs remain in local `results/`.

## E2: completed ablations

The full-minus-ablated PQ differences and paired 95% intervals are:

- Background correction: +0.129132 [0.115435, 0.144275].
- Hysteresis versus single threshold: +0.032713 [0.020704, 0.044228].
- Shape filtering: -0.002668 [-0.008675, 0.003249].
- Fragment merging versus connected components: +0.013564 [0.006988, 0.020476].
- Boundary expansion: +0.007583 [0.002816, 0.012231].

Shape filtering shows no clear benefit in this frozen configuration. This negative
result is retained without changing the selected pipeline. The intervals are
pointwise exploratory comparisons; module effects are not additive.
Evidence: [paired ablation effects](evidence/ablation_effects.json).

## Work still running

All E3 held-out parameter settings are complete; full-tune curve points are still
running. The report generator supports explicitly marked incomplete drafts and
requires complete curves by default. The two 100-instance
independent-review ZIPs have been generated and checked: each contains exactly
100 crops and one blank reviewer CSV, with no curator key. Sampling selected
100 unique predictions from 336 eligible isolated-FP instances.

The 1024-pixel, three-seed/two-loss and six-site experiments are queued. Formal
1024 runs use true batch 4 after a successful 7.0 GiB capacity check; an earlier
accumulation prototype is archived and excluded. No missing experiment is
represented by an estimated result.

The objective E5 evidence is automatic; the auxiliary two-person review and
member names/contributions still require the team. See
[running instructions](RUNNING_EXPERIMENTS.md) for resumption and reviewer workflow.
