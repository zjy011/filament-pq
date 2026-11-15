# CS543 Project Proposal：太阳暗条实例分割（Canvas quiz 版）

2026-11-14 · 按 Canvas「Project Proposal Form」的 11 道题排列。四人每人各交一份；第 1–4 题和第 8 题按个人填写，其余各题全组相同。Canvas 编辑器不渲染 Markdown，所以答案都写成纯文本，可以直接复制粘贴。

---

**Q1. Your name:**

[Full name]

**Q2. NetID:**

[NetID]

**Q3. List of other group members:**

[Name 2], [Name 3], [Name 4]

**Q4. Project group NetIDs:**

[NetID 1], [NetID 2], [NetID 3], [NetID 4]

**Q5. Informative project title.**

How Far Can Classical Image Processing Go? Solar Filament Instance Segmentation Measured Against Inter-Annotator Agreement

**Q6. What the group wants to do (1–3 sentences).**

Following the course's "Compete" option, we use the data and scoring metric (Panoptic Quality, PQ) of the Kaggle Solar Filament Segmentation Challenge 2026 to ask how far classical image processing, with no learning, can go in finding individual solar filaments in full-disk H-alpha images. Minimum goal: a tuned classical pipeline, evaluated once on a held-out test set and compared with human annotators and a small U-Net, so that we can say how large the gap is and which errors cause it. Maximum goal: a stronger U-Net, and tests on observatories not seen during tuning.

**Q7. How you will do it.**

We will build the pipeline from standard operations in OpenCV and scikit-image. It corrects the uneven brightness of the solar disk, thresholds dark regions, cleans them up with morphology, and merges fragments that belong to the same filament. We will tune parameters on 70% of the images and use the remaining 30% once, for testing. We will score with the competition's PQ metric and remove one stage at a time to see what each contributes. Annotators themselves agree only at about PQ 0.33, so to see how close we are to human level, we will score our method and a held-out annotator against the same remaining annotators. The U-Net will reuse our post-processing, so the comparison isolates the pixel-level step.

**Q8. What you will do (as opposed to other members).**

[Name 1]: data preparation and the evaluation code (the PQ evaluator, the train/test splits and the comparison with annotators).

（另外三人各自提交时改成下面对应的一句）

[Name 2]: brightness correction, thresholding and parameter tuning for the classical pipeline.

[Name 3]: fragment merging, the ablation study and the error analysis.

[Name 4]: the U-Net reference and, if time allows, the tests on unseen observatories.

**Q9. What resources you will need.**

We need the MAGFiLO dataset (about 700 MB of images and annotations) and about 2 GB of disk space. The classical pipeline runs on our own laptops. Only the U-Net needs a GPU, and we expect it to use a few GPU-hours in total, which free Colab or Kaggle GPUs should cover. If the course can provide GPU access, we would be glad to use it.

**Q10. How you intend to get the resources you will need.**

The data is public on Harvard Dataverse and the NSO GONG archive, and we have already downloaded it. We will train the U-Net on Colab or Kaggle GPUs.

**Q11. What might go wrong.**

- Technical: The hardest step is deciding when two dark fragments belong to the same filament. Merging too little splits one filament into several, merging too much joins neighbouring ones, and the metric penalises both.
- Data: The labels are incomplete, so some of our "false positives" may be real filaments that no one labeled. We will estimate how often this happens by checking how many of them another annotator did mark.
- Results: Our untuned baseline scores only 0.08–0.10 PQ, compared with about 0.33 between annotators, so our final score may stay well below human agreement.

Even so, the minimum goal is realistic: the data, the evaluator and a working baseline already exist, and the error analysis is useful whatever the final score.
