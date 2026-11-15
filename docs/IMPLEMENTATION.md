# 第一版实现说明

2026-11-15。此版本从零实现，不依赖 plan 中提到但未提供的 `feasibility.py`。
它是可复现的实验基础设施和初步调参结果，不是已经完成的课程研究。

## 数据和运行环境

- 从 [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/J6JNVK)
  下载 `magfilo_2024_v1.0.json`，按文件中的 NSO GONG URL 下载全部 958 张图。
- 实测为 1,593 套标注、10,244 个实例、154 套空标注；图像共 605,772,795 字节。
  标注文件 SHA-256 为 `6f0cbe45e12d8e7a49847664fe9c9ab9d8ddd4efd9019af907af8e01db585bb6`。
- `data_manifest.json` 保存每张图的 URL、字节数和 SHA-256。下载器可续传已完成的文件，
  校验解码尺寸和清单校验和。原始数据、权重、缓存、结果不进入 Git。
- 数据按 MAGFiLO 的 CC BY-NC 4.0 条款使用；保留 GONG 的 NSO/AURA/NSF 来源信息。
- 主划分种子 543：671 张 tune / 287 张 test；tune 内再分为 606 张 train / 65 张 val。
  按观测站与标注人数联合分层，同一文件名的全部标注始终归在一起。
- 已写入主划分和六个留一站划分，划分文件绑定标注文件校验和，禁止静默覆盖不同划分。
  checkpoint 使用规范化 JSON 的划分摘要，避免 Windows/Colab 换行符差异导致误判。
- 在 Windows / Python 3.12 上验证；现已升级为 PyTorch 2.11.0 + CUDA 12.8，使用本机 RTX 4060 8 GB。
  `requirements-tested.txt` 记录测试时的完整依赖，不要求经典流程安装 Torch。

## 已实现的功能

`filament/data.py` 和 `masks.py` 读取真实的 COCO 标注结构，使用 COCO 多边形栅格化和压缩 RLE。
保留空标注，不把同一张图的多人标注合并成一个真值。

`disk.py` 和 `pipeline.py` 提供 M0–M6：日面检测、常数/径向/中值背景、Gaussian、单阈值/
滞后阈值/Frangi、开闭运算与面积/轴比过滤、连通域/骨架方向与距离合并/可选 watershed、
无重叠边界扩展。M4–M6 同时用于经典方法和 U-Net。

日面检查发现 Otsu 的最小外接圆会受灰色外环和日珥影响，因此增加了 512 px 上的 Hough
校正。半径搜索范围为图像短边的 0.35–0.48，适用于本项目的标准化全日面 GONG 图像，
不是通用裁剪图检测器。Hough 与轮廓拟合明显不同时才替换；未检测到合适圆时保留轮廓结果。
六站各 10 张 tune 图的抽查结果保存在本地 `results/disk_qa_v1/`。

中值背景在日面外延续最近的日面边界像素，避免用全局中位数填充产生假暗环。
M0/M1 缓存键包含图像内容、相关配置和实现版本，使用原子写入以支持多进程。
面积过滤先于昂贵的形状统计，减少碎点造成的运行开销。

`evaluate.py` 实现 pooled PQ/SQ/RQ、汇总前景像素 Dice、IoU > 0.3 匈牙利匹配、
面积分档召回、重叠度数、误差分类、按图 bootstrap、相同参照的人机比较和其他标注者对 FP
的佐证统计。面积分档为 `<600`、`600–2400`、`>2400`，端点不留空隙。

在 IoU > 0.5 下严格保留官方 hit-matrix 规则。若输入多边形内部有重叠，理论上的唯一匹配
可能不成立；代码单独报告 `ambiguous_matches`，不静默改成另一种评分定义。
低阈值的匹配在合格边中最大化 IoU 总和，并强制一对一。

计划的误差表没有给出所有单重叠 FP 的独立名称；实现补充 `fragment_related` 和 FP 侧的
`localization`，使每个未匹配 GT/预测都恰好归入一个类别。FN 和 FP 分侧统计，不能把
一个碎片化事件及其关联 FP 当成同一个总数。

`scripts/tune.py` 实现只访问 tune 的分阶段坐标搜索、去重、四进程评估和完整 tune 复核。
`search.yaml` 用于计划中的搜索；`search_quick.yaml` 是初步开发搜索，不代替完整实验。
`make_ablations.py` 生成 E2 配置；`compare_runs.py` 在相同图像集合上做配对 bootstrap。

`label_fp.py` 对孤立 FP 去重抽样，生成只有候选轮廓的裁剪图和两个空白评审 CSV。
不向评审者展示配置、其他标注者标签或来源 ID；映射单独存放在 `curator_only/`。
它不自动产生人工判断，也不把非专家意见当作修正真值。

U-Net 的 E7a 入口包含 ResNet-34/ImageNet、单通道输入、每 epoch 每图随机选择一套标注、
翻转/90 度旋转/亮度对比度增强、Dice+BCE、Adam，以及原分辨率后处理后的验证 PQ 早停。
训练仅用 train，早停仅用 val。模型推理复用经典 M4–M6，并记录 checkpoint 校验和。
64 px / ResNet-18 / 无预训练 / 两张训练图的 CPU 单 epoch 仅用于验证训练保存与加载，
不代表 E7a 的性能或训练耗时。

## 评分和实验规范

组织者的 [self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
通过公开 Kaggle API 下载到本地 `data/`，不随项目分发。
对照脚本加载已检查的四个原始评分函数，不执行 notebook 的顶层单元格。
源码校验和、比较图像 ID、IoU 差异和两边 PQ 保存在对照结果 JSON。

需要特别区分：官方评分函数支持空标注，但它的 CSV 行枚举器发现不了没有实例行的标注套。
本项目从完整标注索引枚举每一套标注，空标注上的所有预测都计 FP，并在对照中额外测试空套。

每次运行保存配置、标注和 split 校验和、代码摘要、Git 状态、依赖版本及图像 ID；
输出目录非空时拒绝覆盖。测试集运行要求显式 `--finalize-test`，同一方法/划分的本地记录
防止无意重复评估。该机制是流程辅助，不代替研究人员遵守盲测协议。

目前所有开发、参数选择和图像检查只使用 tune/train/val。test 尚未评估。
耗时记录包括分割与 RLE，不包括图像读取与评估；缓存和并行负载会影响耗时，不能与
proposal 的历史 0.2 s/image 直接比较。正式 CPU 延迟实验应单进程、关闭缓存、单独运行。

## 验证与后续工作

2026-11-15 后续进展：完整调参和正式 GPU 训练已启动，恢复运行、验证集后处理搜索、
clDice、多种子/跨站实验队列和独立盲审工具已补齐。当前使用方法见
[完整实验运行说明](RUNNING_EXPERIMENTS.md)。以下保留第一阶段的历史结果。

- 本地测试：27 项通过；GitHub Actions 的 Windows/Linux 核心测试均通过。
  官方 notebook 对照：20 张真实 tune 图，另加空标注样例，IoU 逐元素相等且 PQ 精确相等。
- 当前基线 `configs/baseline.yaml` 在全部 671 张 tune 图上得到 PQ 0.127420、
  SQ 0.647015、RQ 0.196935、Dice 0.378695；按图 1,000 次 bootstrap 的 PQ 95% 区间为
  [0.117461, 0.137762]。本地结果：`results/baseline_v1/metrics.json`。
- tune 中 287 张多人标注图共有 601 对标注，标注者间 PQ 0.329582。
  这不是数学上限；人机结论应使用 runner 输出的 `same_reference` 比较。
- 快速搜索使用 24 张 tune 图、1 轮坐标搜索、8 个不同配置，再在全部 671 张 tune 图上
  复核前两名。选中配置已保存为 `configs/quick_tuned.yaml`：阈值 0.92、最小面积 600、
  边界扩展 1 px；完整 tune 上 PQ 0.189096、SQ 0.662334、RQ 0.285499、Dice 0.486688。
  原始记录为 `results/tuning_v1/{provenance,trials,finalists}.json` 和 `best.yaml`。
  这是开发调参结果，尚无最终 test 分数，也没有把它作为经过完整搜索的最终方法。
- 单元与回归测试覆盖 RLE 坐标、严格 IoU 阈值、空套、汇总规则、低阈值一对一匹配、
  全覆盖误差分类、配对 bootstrap、数据隔离、日面灰环、背景边界、合并和缓存失效等。
  本地完整测试包含可选 U-Net 前向/反向传播。CI 在 Windows/Linux 上测试经典核心。
- 下载器已完成全数据获取；官方原函数在 20 张 tune 图上做逐元素/精确 PQ 比较。
- baseline、快速坐标搜索、完整 tune 复核、人工一致性、U-Net 训练与推理入口已实际运行。
- 待完成：完整 150 张/两轮参数搜索；确认 M3/M5 各变体的有效性；正式 E1–E5 和图表；
  E7a 的 512 px 训练与仅 val 上的阈值/M4–M6 调参；冻结配置后的一次最终 test 评估。
- 当前 watershed 是按面积触发、保证分出的部分不小于最小面积的保守实验版本。
  分叉骨架触发、更完整的参数扫描、clDice/多种子/跨站重训练属于后续工作。

首次运行方式及命令见 [README](../README.md)。原始 [plan](CS543_plan.md) 和
[proposal](CS543_proposal.md) 保留原样，作为研究要求而非完成情况清单。
