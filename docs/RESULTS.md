# Measured results

2026-11-15. This page distinguishes completed held-out results from ongoing work.
The experiment recipes are fixed in `configs/study.yaml`.

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

## Work still running

The full classical coordinate search evaluated 65 distinct configurations and is
reranking five finalists on all 671 tune observations. Main classical tests,
ablations, sensitivity curves and the 100-instance independent-review package
follow this freeze. The 1024-pixel, three-seed/two-loss and six-site experiments
are queued. No missing experiment is represented by an estimated result.

The objective E5 evidence is automatic; the auxiliary two-person review and
member names/contributions still require the team. See
[running instructions](RUNNING_EXPERIMENTS.md) for resumption and reviewer workflow.
