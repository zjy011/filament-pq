# Running the source-only submission

The course ZIP intentionally contains only `filament/`, `scripts/` and `configs/`.
Run commands from the extracted directory containing those three folders.
Python 3.11+ is required; the measured environment uses Python 3.12.

The [full repository](https://github.com/zjy011/filament-pq) also contains frozen
split files, the image-checksum manifest, tests, exact tested dependency versions,
aggregate evidence and report instructions. Use that checkout for complete
reproduction. Raw observations and trained checkpoints are downloaded/generated
locally and are not in this archive.

For a standalone classical-method run, create and activate a virtual environment,
then install the direct dependencies:

```shell
python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2" "opencv-python-headless>=4.9,<6" "scikit-image>=0.23,<0.27" "PyYAML>=6,<7" "pycocotools>=2.0.7,<3"
python -m scripts.prepare_data --download-images
python -m scripts.run_experiment --config configs/baseline.yaml --partition tune --out results/baseline_tune
```

The data command verifies the annotation checksum, downloads observations and
generates the grouped main/site splits with seed 543. When the repository's
`data_manifest.json` is present at the root, it additionally compares image bytes
against the original recorded checksums; keep this manifest for strict reproduction.

Optional learned models require compatible Torch/Torchvision builds plus
`segmentation-models-pytorch`. Report generation requires Matplotlib, ReportLab
and pypdf. Installation details for the measured CUDA environment and commands
for all E1-E7 experiments are in the repository's
[running guide](https://github.com/zjy011/filament-pq/blob/main/docs/RUNNING_EXPERIMENTS.md).

Completed test configurations must stay frozen. The queue skips completed work
and resumes interrupted training/evaluation. Do not start duplicate queue copies.
Human-review judgments and member contributions must be provided by the team.

MAGFiLO annotations: CC BY-NC 4.0, Ahmadzadeh et al.,
[Harvard Dataverse](https://doi.org/10.7910/DVN/J6JNVK).
Images: NSO/AURA/NSF/GONG, CC BY 4.0. The adapted clDice code's MIT license is
included at `filament/unet/LICENSE.clDice.txt`.
