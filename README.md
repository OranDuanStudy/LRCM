# LRCM: Listen to Rhythm, Choose Movements

![arXiv](https://img.shields.io/badge/arXiv-2601.03323-red?style=flat-square&logo=arxiv)
![GitHub](https://img.shields.io/badge/GitHub-OranDuanStudy/LRCM-blue?style=flat-square&logo=github)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-Research-yellow?style=flat-square)

**Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset**

[[Paper]](https://arxiv.org/abs/2601.03323)
[[GitHub]](https://github.com/OranDuanStudy/LRCM)
[[Project Page]](https://oranduanstudy.github.io/LRCM)

---

**LRCM** (Listen to Rhythm, Choose Movements) is a multimodal-guided diffusion framework for dance motion generation that simultaneously leverages **audio rhythm** and **hierarchical text descriptions** (global style + local movements) for high-quality, controllable dance synthesis.

## Key Innovations

1. **Decoupled Multimodal Dance Dataset Paradigm** — Fine-grained semantic decoupling of motion capture data, audio rhythm, and professionally annotated global/local text descriptions
2. **Heterogeneous Multimodal-Guided Diffusion Architecture** — Audio-latent Conformer + Text-latent Cross-Conformer for simultaneous audio and text conditioning
3. **Motion Temporal Mamba Module (MTMM)** — State space model-based autoregressive extension for efficient long-sequence generation

## Features

- **Dual-modality conditioning**: Audio rhythm + Text descriptions (global + local)
- **Autoregressive generation**: Efficient long-sequence synthesis via Mamba SSM
- **Multiple dance styles**: Hip-hop, Jazz, Krump, Popping, Locking, Charleston, Tap

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dance motion
python synthesize.py \
    --checkpoints ckpt/dance_LRCM_stage3.ckpt \
    --data_dirs data/Multimodal_Text_dataset_updating/ \
    --input_files sample_input.pkl \
    --input_text "dynamic hip-hop dance with arm waves and body rolls" \
    --dest_dir results/
```

## Citation

```bibtex
@misc{lrcm2025,
  title={Listen to Rhythm, Choose Movements: Autoregressive Multimodal Dance Generation via Diffusion and Mamba with Decoupled Dance Dataset},
  author={Oran Duan and Yinghua Shen and Yingzhu Lv and Luyang Jie and Yaxin Liu and Qiong Wu},
  year={2025},
  eprint={2601.03323},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
