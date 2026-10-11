# grpo_qwen3.0.6b-sft

Experiments on GRPO / RLCD calibration for small Qwen models (Qwen3-0.6B, Qwen3.5-0.8B),
including supervised fine-tuning (SFT), continued pre-training (CPT), GRPO-based
reinforcement learning and calibration-aware variants.

## Contents

| Path | Description |
| --- | --- |
| `config/` | VERL / training configuration files |
| `scripts/` | Training and data-preparation scripts |
| `data/` | Small GSM8K data sample used by the smoke run |
| `logs/` | Training logs (zipped as `logs.zip`) |
| `GRPO_RLCD_校准决策实验报告.docx` | Experiment report (calibration-aware RLCD study) |

## Note on model weights

Model weights and training checkpoints (about 19 GB in total) are **not** stored in this
repository because GitHub rejects files larger than 100 MB. The excluded directories are
listed in `.gitignore`:

- `models/` - base model weights (Qwen3-0.6B, Qwen3.5-0.8B)
- `checkpoints/` - training checkpoints, including optimizer states

## Training environment

VERL + TRL based training, Python 3.12 environment, single-node multi-GPU.
