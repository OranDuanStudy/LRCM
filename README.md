# LRCM: Listen to Rhythm, Choose Movements

![arXiv](https://img.shields.io/badge/arXiv-2601.03323-red?style=flat-square&logo=arxiv)
![GitHub](https://img.shields.io/badge/GitHub-OranDuanStudy/LRCM-blue?style=flat-square&logo=github)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-Research-yellow?style=flat-square)

**Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset**

[[Paper]](https://arxiv.org/abs/2601.03323) · [[GitHub]](https://github.com/OranDuanStudy/LRCM) · [[Project Page]](https://oranduanstudy.github.io/LRCM)

---

**LRCM** (Listen to Rhythm, Choose Movements) is a multimodal-guided diffusion framework for dance motion generation that simultaneously leverages **audio rhythm** and **hierarchical text descriptions** (global style + local movements) for high-quality, controllable dance synthesis.

## Overview

Current dance motion generation methods suffer from **coarse semantic control** and **poor coherence in long sequences**. LRCM addresses these through:

1. **Decoupled Multimodal Dance Dataset Paradigm** — Fine-grained semantic decoupling of motion, audio, and text
2. **Heterogeneous Multimodal-Guided Diffusion Architecture** — Audio-latent Conformer + Text-latent Cross-Conformer
3. **Motion Temporal Mamba Module (MTMM)** — State space model-based autoregressive extension for long-sequence generation

### Key Features

- **Dual-modality conditioning**: Audio rhythm + Text descriptions (global + local)
- **Autoregressive generation**: Efficient long-sequence synthesis via Mamba SSM
- **7 dance genres**: Hip-hop, Jazz, Krump, Popping, Locking, Charleston, Tap

---

## Installation

```bash
# Clone the repository
git clone https://github.com/OranDuanStudy/LRCM.git
cd LRCM

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- CUDA 12.x
- PyTorch 2.4+
- 4× RTX 4090 (24GB) for training

### Key Dependencies

```
pytorch-lightning==1.9.5
mamba-ssm
causal-conv1d
openai-clip
librosa
scipy
scikit-learn
torch>=2.4.0
```

---

## Project Structure

```
.
├── models/
│   ├── LightningModel.py      # Main Lightning model
│   ├── BaseModel.py
│   ├── nn.py                  # Neural network building blocks
│   ├── mamba/                 # Motion Temporal Mamba Module
│   │   └── mambamotion.py
│   ├── lgtm/                  # Text encoders and diffusion components
│   │   ├── conformer.py
│   │   ├── text_encoder.py
│   │   ├── motion_diffusion.py
│   │   └── utils/
│   └── transformer/           # Transformer components
│       └── tisa_transformer.py
├── utils/
│   ├── motion_dataset.py      # Dataset loaders
│   └── hparams.py             # Hyperparameter management
├── pymo/                      # Motion preprocessing (BVH, rotations)
├── hparams/
│   ├── LRCM_stage1.yaml       # Phase 1: Global text + Audio
│   ├── LRCM_stage2.yaml       # Phase 2: Add Local text
│   └── LRCM_stage3.yaml       # Phase 3: Enable MTMM
├── train.py                   # Training script
├── synthesize.py              # Inference script
└── requirements.txt
```

---

## Inference

### Quick Generation

```bash
python synthesize.py \
    --checkpoints ckpt/dance_LRCM_stage3.ckpt \
    --data_dirs data/Multimodal_Text_dataset_updating/ \
    --input_files sample_input.pkl \
    --input_text "dynamic hip-hop dance with arm waves and body rolls" \
    --dest_dir results/
```

### Batch Generation

```bash
# Full text prompts
bash experiments/LRCM_manbadance_duainput_memory.sh

# Global/Local text from JSON
bash experiments/LRCM_duainput_memory_json.sh
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-c, --checkpoints` | Path to model checkpoint | Required |
| `-d, --data_dirs` | Path to data directory | Required |
| `-f, --input_files` | Input motion file | Required |
| `-t, --input_text` | Text description (global style) | Required |
| `-r, --seed` | Random seed | 42 |
| `--dest_dir` | Output directory | "results" |
| `-g, --gf` | Guidance factor(s) | None |
| `-k, --gpu` | GPU device | "cuda:0" |
| `-m, --segment-frames` | Segment frame length | 300 |

---

## Training

### Three-Phase Training Strategy

**Phase 1**: Global text + Audio (Foundation)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py \
    --dataset_root data/Multimodal_Text_dataset_updating \
    --hparams_file ./hparams/LRCM_stage1.yaml \
    --ckpt_file None
```

**Phase 2**: Add Local text (Fine-tuning)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py \
    --dataset_root data/Multimodal_Text_dataset_updating \
    --hparams_file ./hparams/LRCM_stage2.yaml \
    --ckpt_file ./pretrained_models/dance_LRCM_stage1.ckpt
```

**Phase 3**: Enable MTMM (Autoregressive)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python train.py \
    --dataset_root data/Multimodal_Text_dataset_updating \
    --hparams_file ./hparams/LRCM_stage3.yaml \
    --ckpt_file ./pretrained_models/dance_LRCM_stage2.ckpt
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (weight decay: 1.0e-4) |
| Diffusion steps | 200 (DDPM sampler) |
| Residual blocks | 20 |
| Model size | ~316M parameters |
| Noise injection | 0.05 probability per modality |

---

## Visual Overview

### Overview

![Overview](docs/graphs/fig1.png)

### Architecture

![Architecture](docs/graphs/fig3.png)

---

## Citation

```bibtex
@misc{lrcm2025,
  title = {Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset},
  author = {Oran Duan and Yinghua Shen and Yingzhu Lv and Luyang Jie and Yaxin Liu and Qiong Wu},
  year = {2025},
  eprint = {2601.03323},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```
