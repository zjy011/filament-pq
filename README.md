# filament-pq

**How Far Can Classical Image Processing Go? Solar Filament Instance Segmentation Measured Against Inter-Annotator Agreement**

**经典图像处理能走多远？以标注者间一致性为参照的太阳暗条实例分割**

CS543 / ECE549 Computer Vision, Fall 2026 · Course project ("Compete" option)

**Implementation status (2026-11-15):** data and evaluation tools are complete;
the held-out tests give baseline **PQ 0.1278**, tuned classical **0.2314**,
and U-Net **0.3194**. The 100-sample blind-review packages are ready locally.
Ablations, sensitivity and extended experiments are running. See [measured results](docs/RESULTS.md)
and [resumable experiment commands](docs/RUNNING_EXPERIMENTS.md).
The proposal-era feasibility values below are historical context, not current test results.

[English](#english) · [中文](#中文)

---

## English

### What we want to do

Following the course's "Compete" option, we use the data and scoring metric (Panoptic Quality, PQ) of the [Kaggle Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). We ask how far classical image processing, with no learning, can go in finding individual solar filaments in full-disk H-alpha images. We are not trying to top the leaderboard. The goal is to measure the gap between a classical pipeline, human annotators and a learned model, and to explain which errors cause it.

- **Minimum goal:** a tuned classical pipeline, evaluated once on a held-out test set and compared with human annotators and a small U-Net.
- **Maximum goal:** a stronger U-Net, and tests on observatories not seen during tuning.

### Data and metric

- **Dataset:** [MAGFiLO v1.0](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J6JNVK). It has 958 GONG H-alpha images (2048×2048) from 6 observatories, with 1,593 annotation sets and 10,244 filaments. 410 images are labeled by 2–3 annotators. License: CC BY-NC 4.0.
- **Metric:** Panoptic Quality, PQ = SQ × RQ. A prediction matches a ground-truth instance when IoU > 0.5, following the organizers' [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).
- **Reference points we have already measured:**
  - Annotators agree with each other at only about **PQ 0.33** (860 annotator pairs).
  - Our untuned classical baseline scores **PQ 0.08–0.10**, at about 0.2 s per image on a CPU.

### How we will do it

We will build the pipeline from standard operations in OpenCV and scikit-image. It corrects the uneven brightness of the solar disk, thresholds dark regions and cleans them up with morphology. It then merges fragments that belong to the same filament.

We will tune parameters on 70% of the images and use the remaining 30% once, for testing. We will score with the competition's PQ metric and remove one stage at a time to see what each contributes.

Because annotators themselves agree only at about PQ 0.33, we compare against humans under the same reference. We score both our method and a held-out annotator against the remaining annotators. The U-Net reuses our post-processing, so the comparison with it isolates the pixel-level step.

### Division of work

| Member | Responsibility |
| --- | --- |
| 1 | Data preparation and evaluation code: the PQ evaluator, the train/test splits and the comparison with annotators |
| 2 | Brightness correction, thresholding and parameter tuning for the classical pipeline |
| 3 | Fragment merging, the ablation study and the error analysis |
| 4 | The U-Net reference and, if time allows, the tests on unseen observatories |

### Resources

- **Data:** the MAGFiLO dataset (about 700 MB of images and annotations) and about 2 GB of disk space. The data is public on Harvard Dataverse and the NSO GONG archive, and we have already downloaded it.
- **Compute:** the classical pipeline runs on our own laptops. Only the U-Net needs a GPU, a few GPU-hours in total. We will use free Colab or Kaggle GPUs, and would gladly use course GPUs if they are available.

### What might go wrong

- **Technical:** the hardest step is deciding when two dark fragments belong to the same filament. Merging too little splits one filament into several, and merging too much joins neighbouring ones. The metric penalises both.
- **Data:** the labels are incomplete, so some of our "false positives" may be real filaments that no one labeled. We will estimate how often this happens by checking how many of them another annotator did mark.
- **Results:** our untuned baseline scores only 0.08–0.10 PQ, compared with about 0.33 between annotators, so our final score may stay well below human agreement.

Even so, the minimum goal is realistic. The data, the evaluator and a working baseline already exist, and the error analysis is useful whatever the final score.

---

## 中文

### 我们要做什么

我们按照课程的 "Compete" 选项，使用 [Kaggle Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) 的数据和评分指标（Panoptic Quality，PQ）。我们要回答的问题是：不用任何学习方法，经典图像处理在全日面 H-alpha 图像上找出每一条太阳暗条，能做到什么程度？目标不是冲排行榜，而是量化经典流程、人类标注者和学习方法之间的差距，并解释差距来自哪类错误。

- **最小目标：** 调好参数的经典流程，在留出的测试集上只评估一次，并和人类标注者、一个小 U-Net 对比。
- **最大目标：** 更强的 U-Net，以及在调参时没见过的观测站上测试。

### 数据与指标

- **数据集：** [MAGFiLO v1.0](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J6JNVK)，共 958 张来自 6 个观测站的 GONG H-alpha 图像（2048×2048），1,593 套标注、10,244 条暗条，其中 410 张图有 2–3 人标注。许可：CC BY-NC 4.0。
- **指标：** Panoptic Quality，PQ = SQ × RQ。预测和真值的 IoU > 0.5 时算匹配，规则与组织者的 [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) 一致。
- **已经测出的参照数字：**
  - 标注者之间的一致性只有约 **PQ 0.33**（860 对标注）。
  - 我们未调参的经典基线为 **PQ 0.08–0.10**，CPU 上每张图约 0.2 秒。

### 怎么做

我们用 OpenCV 和 scikit-image 的标准操作搭建流程：先校正日面亮度不均，再对暗区做阈值分割，并用形态学清理，最后把属于同一条暗条的碎片合并起来。

70% 的图用来调参，剩下 30% 只在最后测试时用一次。评分使用比赛的 PQ 指标，并且每次去掉流程中的一步，看每一步各贡献多少。

标注者之间本身只有约 PQ 0.33 的一致性，所以和人比较时要用相同的参照：把我们的方法和一名留出的标注者，分别拿去和其余标注者比较。U-Net 复用我们的后处理，因此和它的对比只反映"像素级分割"这一步的差别。

### 分工

| 成员 | 负责内容 |
| --- | --- |
| 1 | 数据准备和评估代码：PQ 评估器、训练/测试划分、与标注者的对比 |
| 2 | 经典流程的亮度校正、阈值分割和调参 |
| 3 | 碎片合并、消融实验和误差分析 |
| 4 | U-Net 对照；时间允许的话，做未见观测站上的测试 |

### 资源

- **数据：** MAGFiLO 数据集（图像和标注约 700 MB），约 2 GB 磁盘空间。数据在 Harvard Dataverse 和 NSO GONG 存档上公开，我们已经下载好。
- **算力：** 经典流程在自己的笔记本上跑。只有 U-Net 需要 GPU，总共几个 GPU 小时。我们计划用免费的 Colab 或 Kaggle GPU；如果课程能提供 GPU，也很乐意使用。

### 可能的风险

- **技术难点：** 最难的是判断两个暗色碎片是否属于同一条暗条。合并得太少，一条暗条会被拆成好几段；合并得太多，相邻的暗条会被连成一条。指标对这两种错误都扣分。
- **数据问题：** 标注不完整，所以我们的一部分"误检"可能其实是没人标的真实暗条。我们会统计这些误检中有多少被另一名标注者标过，以此估计这种情况的比例。
- **结果风险：** 未调参的基线只有 0.08–0.10 PQ，而标注者之间约为 0.33，最终分数可能仍远低于人的一致性。

即便如此，最小目标是现实的：数据、评估器和可运行的基线都已经有了，而且无论最终分数如何，误差分析本身都有价值。

---

## Repository layout / 仓库结构

```
filament-pq/
├── filament/             # Data, masks, M0–M6, metrics, visualization, optional U-Net
├── scripts/              # Download, experiments, tuning, human comparison, E5 crops, training
├── configs/              # Baseline, extended, full/quick coordinate searches
├── splits/               # Frozen observation IDs: main + six site holdouts
├── tests/                # Synthetic regression tests; no dataset download required
├── data_manifest.json    # Public image URLs, sizes and SHA-256 checksums
├── docs/                 # Proposal, plan and implementation notes
├── data/                 # Downloaded locally, ignored by Git
└── results/              # Local experiment artifacts, ignored by Git
```

The first implementation is available. See [implementation notes](docs/IMPLEMENTATION.md)
for verified behavior and remaining experiments. The figures earlier in this README
are from the proposal's feasibility study; `configs/baseline.yaml` is a new, explicit
starting configuration, not a recovered copy of `feasibility.py`.

第一版实现已可运行，验证情况和后续工作见[实现说明](docs/IMPLEMENTATION.md)。
上面的历史数字来自 proposal；当前 baseline 是从零实现的明确配置。
原始数据、缓存、模型权重和实验输出保存在本地，不提交到 Git。

Initial development run: on all 671 tuning observations, the baseline scores PQ
0.1274 and `configs/quick_tuned.yaml` scores PQ 0.1891 after a short coordinate
search. These are tuning scores, not held-out test results. Local tests (27) and
Windows/Linux core CI passed; the 20-image official Torch comparison matches exactly.

## Quick start / 开始运行

Python 3.11+ (tested on Windows with Python 3.12). Run from this repository's root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m scripts.prepare_data --download-images
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.run_experiment --config configs/baseline.yaml --out results/baseline_tune --partition tune
```

On Linux/macOS use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.
The downloader resumes existing files and verifies checksums and dimensions. It
fetches the 61 MB annotation JSON and about 606 MB of images from Dataverse/GONG.
`requirements-tested.txt` records the full development environment, including
optional CPU Torch packages; normal installations should use the extras above.

```powershell
# Full planned coordinate search: 150 tuning images, two rounds, top five rechecked on all tune
.\.venv\Scripts\python.exe -m scripts.tune --out results/tuning
# Short development search; still rechecks finalists on all 671 tune observations
.\.venv\Scripts\python.exe -m scripts.tune --search configs/search_quick.yaml --sample-size 24 --rounds 1 --top-k 2 --out results/quick_tuning
# Annotator agreement (tune only by default)
.\.venv\Scripts\python.exe -m scripts.inter_annotator
# Create E2 configurations from the selected method
.\.venv\Scripts\python.exe -m scripts.make_ablations --config results/tuning/best.yaml
# Blinded E5 crops and two independent reviewer CSVs
.\.venv\Scripts\python.exe -m scripts.label_fp --run results/baseline_tune --out results/fp_review
```

An experiment writes `provenance.json`, `metrics.json`, `per_image.json`, prediction
RLE JSON/CSV, and PNG overlays. Metrics pool counts across annotation sets; bootstrap
resamples whole observations. `--cache cache/preprocess` enables M0/M1 caching;
leave it off for latency measurements. Timing excludes image IO and metric computation.

```powershell
# Optional E7a U-Net: defaults to 512 px, ResNet-34/ImageNet, Dice+BCE, validation-PQ early stopping
.\.venv\Scripts\python.exe -m pip install -e ".[unet]"
.\.venv\Scripts\python.exe -m scripts.train_unet --out results/unet
.\.venv\Scripts\python.exe -m scripts.run_experiment --checkpoint results/unet/best.pt --partition val --config configs/baseline.yaml --out results/unet_val
```

Tune the U-Net probability threshold and post-processing **only on `val`** before
freezing its final configuration. The full automated U-Net post-processing search,
clDice extension and formal E1–E7 results are still pending. The classical tuning
script uses `tune` only and must not be reused for U-Net validation selection.

Test evaluation requires `--partition test --finalize-test`, disallows `--limit`,
and records a local receipt keyed by the data split, method configuration and model
checkpoint. It refuses to repeat the same test run. Inspect an interrupted run's
receipt before manually removing it; never delete completed receipts to retune.
Do not run final test evaluation until all method choices are frozen.

## Official metric check / 官方评分核对

Download and inspect the organizer's notebook (Apache-2.0); it is kept outside Git:

```powershell
curl.exe -L --fail https://www.kaggle.com/api/v1/kernels/pull/azimahmadzadeh/self-evaluation-notebook -o data/official-notebook-response.json
.\.venv\Scripts\python.exe -m scripts.validate_official --notebook data/official-notebook-response.json --count 20
```

The check runs its original Torch overlap/PQ functions against our RLE implementation.
Empty annotation sets are preserved by our dataset index. The notebook's scorer
handles empty sets, but its CSV-only iterator cannot enumerate sets with no rows;
our runner explicitly includes them, as required by the project protocol.

## References / 参考

- MAGFiLO dataset, [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J6JNVK); paper in [Scientific Data, 2024](https://doi.org/10.1038/s41597-024-03876-y)
- [GONG H-alpha archive](https://gong2.nso.edu/HA/hag/)
- Kirillov et al., [Panoptic Segmentation](https://arxiv.org/abs/1801.00868), CVPR 2019
- Shih & Kowalski 2003, Solar Physics 218, 99–122; [Qu et al. 2005](https://doi.org/10.1007/s11207-005-5780-1); [Fuller, Aboudarham & Bentley 2005](https://doi.org/10.1007/s11207-005-8364-1); [Bernasconi, Rust & Hakim 2005](https://doi.org/10.1007/s11207-005-2766-y)
- [Course project page](http://luthuli.cs.uiuc.edu/~daf/Courses/CV2026Fall/FallCV26projects.html)
