# CS543 / ECE549 Project Progress Report

Progress snapshot: November 15, 2026. Member names remain to be filled in before submission.

## 1. List of group members

[Member 1], [Member 2], [Member 3], [Member 4]

## 2. Informative project title

**How Far Can Classical Image Processing Go? Solar Filament Instance Segmentation Measured Against Inter-Annotator Agreement**

## 3. What you have done to date

We study how well classical image processing can identify individual solar filaments in full-disk H-alpha images, and how its performance compares with human annotation agreement and a small U-Net.

We have downloaded all 958 MAGFiLO observations, implemented the classical pipeline and evaluation tools, and completed the main experiments on a fixed 287-image test split. The starting baseline achieves panoptic quality (PQ) of **0.1278**, the tuned classical pipeline achieves **0.2314**, and a 512-pixel U-Net achieves **0.3194**. We have also completed ablations, parameter-sensitivity experiments, error analysis, and an objective analysis of annotation disagreement.

Our **minimum goal** is a reproducible evaluation that explains the classical method's performance, its gap to U-Net, and its main failure modes. Most computational work for this goal is complete; auxiliary human review and report preparation remain.

Our **maximum goal** adds three-seed experiments at 1024 resolution, a topology-aware loss, and leave-one-observatory-out evaluation. All three 1024-pixel Dice+BCE runs are complete, with test PQ **0.3481, 0.3537, and 0.3549** (mean **0.3522**, sample standard deviation **0.0037**). All six classical site-holdout experiments are also complete. We are deferring topology-aware training and learned-model site-holdout runs to prioritize further classical-method diagnosis.

## 4. How you did it

The classical pipeline detects the solar disk, corrects uneven brightness, extracts dark regions using thresholding, removes small artifacts, groups fragments into instances, and adjusts their boundaries. Parameters are selected using tuning observations before test evaluation.

We keep all annotations of the same photograph in the same partition to prevent leakage. Each prediction is evaluated separately against every available annotation, including empty annotation sets. We report PQ, segmentation and recognition quality, foreground Dice, and instance-level error categories. Confidence intervals use **1,000 observation-level bootstrap samples**.

Our evaluator matches the organizer's implementation on 20 tuning observations and an empty-annotation fixture. The U-Net uses an ImageNet-initialized ResNet34 encoder, with checkpoint and post-processing selection restricted to validation data. Human-machine comparisons use identical reference annotations.

## 5. Changes to your intentions

We moved training to a local RTX 4060 GPU and implemented checkpointing so interrupted experiments can resume.

Classical image processing remains our main focus; U-Net is a reference for interpreting its limitations. Background correction provides a substantial benefit, while shape filtering shows no clear PQ improvement. We are investigating where classical candidates disappear during extraction and filtering, using tuning observations only. The present score does not establish an upper bound on all classical methods.

A diagnostic on 150 tuning observations found that 672 of 787 zero-overlap misses lost their remaining candidate coverage at area filtering. We tested 12 filtering variants on those observations and six on the other 521 tuning observations. Simply lowering the area threshold increased recall but substantially increased false positives. Delaying the final area filter gave a small tuning gain, but a single frozen exploratory evaluation on the previously inspected test set produced PQ **0.2313**, versus **0.2314** originally: no supported improvement. It added 45 correct matches and 180 false-positive comparison occurrences. We retain the original method and this negative result; filtering order alone does not explain or close the U-Net gap.

The two methods share post-processing code but use separately selected settings, including different disk margins. We will therefore describe them as complete pipeline comparisons, rather than claiming that only the foreground-generation step changes.

## 6. What might go wrong

The main difficulties are separating neighboring filaments, reconnecting fragmented structures, and distinguishing genuine false detections from incomplete annotations. Some predictions rejected by one annotator are supported by another, so apparent errors do not always indicate physically incorrect detections.

Training variability and limited GPU resources may restrict the larger experiments. Site-holdout comparisons also involve different evaluation samples, so their differences require careful interpretation. Random observation splitting does not guarantee temporal independence.

Our minimum goal remains achievable using the completed experiments. Two group members will independently review 100 sampled false positives as auxiliary evidence; their non-expert judgments will not replace the main evaluation results.

---

## Supporting project records

The six answers above are the Canvas submission text. The following links provide supporting evidence within this repository:

- [Original proposal](CS543_proposal.md)
- [Experiment plan](CS543_plan.md)
- [Measured results](RESULTS.md)
- [Report summary and experiment metrics](evidence/report_summary.json)
- [Experiment and resumption instructions](RUNNING_EXPERIMENTS.md)

This is a dated progress snapshot. Unfinished experiments and human reviews are not represented as completed results.
