# STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization

> Official implementation of STAR, a two-stage framework for learning diverse robot skill abstractions with rotation-augmented residual quantization and autoregressive skill composition.

## Authors

**Hao Li**<sup>1,2</sup>, **Qi Lv**<sup>1,2</sup>, **Rui Shao**<sup>1*</sup>, **Xiang Deng**<sup>1*</sup>, **Yinchuan Li**<sup>2</sup>, **Jianye Hao**<sup>2</sup>, **Liqiang Nie**<sup>1</sup>

<sup>1</sup> School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)  
<sup>2</sup> Huawei Noah's Ark Lab  
\* Corresponding authors

## Links

- **Paper**: [STAR on arXiv](https://arxiv.org/abs/2506.03863)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [06/2025] Release arXiv paper.
- [04/2026] Release STAR codebase with paper-aligned naming, training configs, and README.

---

## Introduction

This repository contains the implementation of **STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization**.

STAR addresses multitask visuomotor policy learning by converting continuous action sequences into structured, reusable skill abstractions and then composing them autoregressively for long-horizon manipulation. The method is built around two components:

- **RaRSQ**: Rotation-augmented Residual Skill Quantization for stage-0 skill abstraction.
- **CST**: Causal Skill Transformer for stage-1 autoregressive skill prediction and composition.

Compared with standard VQ-based skill learning pipelines, STAR is designed to mitigate codebook collapse and better model the causal dependency between learned skills. According to the paper, STAR achieves strong performance on LIBERO and real-world tasks, with about **12% improvement over baselines**.

This repository currently provides:

- training code for stage-0 RaRSQ and stage-1 CST
- a single-process training entry point
- MetaWorld data collection scripts
- evaluation scripts
- benchmark configurations for **LIBERO** and **MetaWorld**

The current public codebase mainly focuses on the simulation pipeline used in the paper.

---

## Highlights

- Two-stage policy learning framework: **RaRSQ + CST**
- Supports **LIBERO** and **MetaWorld** benchmarks
- Provides **data collection**, **training**, and **evaluation** scripts
- Includes paper-aligned configs such as [`config/train_rarsq.yaml`](config/train_rarsq.yaml) and [`config/train_cst.yaml`](config/train_cst.yaml)
- Useful for paper reproduction, ablation studies, and follow-up research on discrete skill learning

---

## Method / Framework

STAR consists of two stages:

### Stage 0: RaRSQ

RaRSQ learns discrete robot skill abstractions from action chunks with residual quantization. Its key idea is to inject relative angular information into the gradient flow through a rotation-based mechanism, which helps preserve diversity in the learned codebook and alleviates codebook collapse.

### Stage 1: CST

CST autoregressively models dependencies between skill representations and predicts skill codes from coarse to fine levels. It further refines the generated actions through an offset prediction mechanism, improving precision for long-horizon manipulation.

### Paper Terminology and Config Mapping

- **STAR**: the full two-stage policy
- **RaRSQ**: stage-0 skill abstraction, configured by [`config/train_rarsq.yaml`](config/train_rarsq.yaml)
- **CST**: stage-1 skill prediction, configured by [`config/train_cst.yaml`](config/train_cst.yaml)

---

## Project Structure

```text
.
├── 2506.03863v2.pdf
├── config/                  # Hydra configs for tasks, algorithms, and training
├── scripts/
│   └── generate_metaworld_dataset.py
├── star/
│   ├── algos/               # STAR policy, quantizer, transformer, and utilities
│   ├── env_runner/          # Benchmark rollout runners
│   └── utils/               # Dataset, logging, benchmark, and helper utilities
├── third_parties/
│   └── LIBERO/              # Bundled LIBERO dependency
├── outputs/                 # Training logs, checkpoints, collected videos, and evaluations
├── train.py                 # Single-process training entry point
├── evaluate.py              # Evaluation entry point
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd STAR
```

### 2. Create environment

```bash
conda create -n star python=3.10.14
conda activate star
```

### 3. Install dependencies

```bash
python -m pip install torch==2.2.0 torchvision==0.17.0
python -m pip install -e .
python -m pip install -e third_parties/LIBERO
```

`requirements.txt` pins `robosuite<1.5` to stay compatible with the bundled LIBERO environment code.

### 4. Optional: headless rendering for MetaWorld data collection

If you are collecting MetaWorld data on a headless server, you may need:

```bash
export MUJOCO_GL=osmesa
```

The repository depends on the `metaworld` package from [`requirements.txt`](requirements.txt) and uses the bundled LIBERO source under [`third_parties/LIBERO`](third_parties/LIBERO).

For offline LIBERO runs, you can point the CLIP text encoder to a local checkpoint with:

```bash
task.task_embedding_model_path=/path/to/clip-vit-base-patch32
```

The cleaned codebase keeps the paper's CLIP text encoder path for LIBERO and removes older alternative language encoders.

---

## Checkpoints / Models

Pretrained checkpoints are **not included** in the repository at the moment.

Once released, this section can be updated with:

- pretrained stage-0 RaRSQ checkpoints
- pretrained stage-1 CST checkpoints
- additional benchmark-specific models

---

## Dataset / Benchmark

This repository supports two benchmark families:

- **LIBERO**
- **MetaWorld**

### LIBERO

Please follow the official LIBERO dataset instructions:

- [LIBERO dataset documentation](https://lifelong-robot-learning.github.io/LIBERO/html/algo_data/datasets.html#datasets)

In the bundled LIBERO release, the long-horizon 10-task suite used in the paper is exposed as `libero_10`.

### MetaWorld

MetaWorld data can be collected with:

```bash
python scripts/generate_metaworld_dataset.py
```

By default, the provided collection setup generates:

- **100** demonstrations for each of the 50 MT50 tasks

For quick debugging or single-task collection, you can override the task list from the command line:

```bash
python scripts/generate_metaworld_dataset.py \
    task=metaworld_mt50 \
    task.env_names='[assembly-v2]' \
    +task.demos_per_env=1 \
    rollout.rollouts_per_env=1
```

The cleaned defaults assume:

- MetaWorld data under `data/`
- runtime outputs under `outputs/`
- LIBERO datasets under `third_parties/LIBERO/libero/datasets` via a CLI override
- CLIP text embeddings for LIBERO

---

## Usage

### 1. Stage-0 RaRSQ Training

`config/train_rarsq.yaml` now carries the paper-style defaults used in this repository:

- block size `8`
- codebook size `16`
- quantization depth `2`
- rotation augmentation enabled
- batch size `1024`
- `100` epochs

Train RaRSQ on LIBERO with:

```bash
python train.py --config-name=train_rarsq \
    task=libero_10 \
    data_prefix=third_parties/LIBERO/libero/datasets \
    exp_name=final \
    variant_name=block_8 \
    seed=0
```

If a single GPU cannot hold the paper batch size, override it directly, for example:

```bash
train_dataloader.batch_size=128
```

### 2. Stage-1 CST Training

`config/train_cst.yaml` keeps the cleaned stage-1 defaults:

- batch size `512`
- learning rate `8e-4`
- `500` epochs
- first / second code prediction weights `2.0 / 1.0`
- action refinement loss weight `20`

For single-process or single-GPU training, stage-1 CST can be trained with `train.py` by explicitly loading a stage-0 checkpoint:

```bash
python train.py --config-name=train_cst \
    task=libero_10 \
    data_prefix=third_parties/LIBERO/libero/datasets \
    exp_name=final \
    variant_name=block_8 \
    checkpoint_path=/path/to/stage0_checkpoint.pth \
    seed=0
```

You can also set `training.auto_continue=true` to load the latest compatible stage-0 checkpoint from the matching experiment directory. This requires stage-0 and stage-1 runs to share the same task, experiment name, variant, and major algorithm hyperparameters.

### 3. Evaluation

Evaluate a trained STAR model with:

```bash
python evaluate.py \
    task=libero_10 \
    algo=star \
    data_prefix=third_parties/LIBERO/libero/datasets \
    exp_name=final \
    variant_name=block_8 \
    stage=1 \
    training.use_tqdm=false \
    seed=0
```

If `checkpoint_path` is left as `null`, evaluation automatically resolves the latest checkpoint from the matching experiment directory. You can also pass `checkpoint_path=/path/to/checkpoint.pth` explicitly.

## TODO

- [ ] Release pretrained checkpoints
- [ ] Release project page and demo assets
- [ ] Add framework and qualitative result figures
- [ ] Add more benchmark reproduction commands
- [ ] Provide clearer single-GPU training recipes

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2025star,
  title={STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization},
  author={Li, Hao and Lv, Qi and Shao, Rui and Deng, Xiang and Li, Yinchuan and Hao, Jianye and Nie, Liqiang},
  journal={arXiv preprint arXiv:2506.03863},
  year={2025}
}
```

---

## Acknowledgement

- Thanks to the authors of **QueST** for the baseline codebase and engineering foundation.
- Thanks to the teams behind **LIBERO** and **MetaWorld** for the benchmarks and datasets used in this work.
- Thanks to the open-source community for making research reproduction easier.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.
