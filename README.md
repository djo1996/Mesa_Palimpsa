# Mesa_Palimpsa
### Learning to Remember, Learn, and Forget in Attention-Based Models

![Palimpsa Logo](https://github.com/user-attachments/assets/7fa41f32-0976-42c9-8d32-2a602e56289f)

[![Palimpsa Paper](https://img.shields.io/badge/Paper-ICML%202026-success)](https://arxiv.org/abs/2602.09075)
[![Mesanet Paper](https://img.shields.io/badge/Paper-Mesanet-blue)](https://arxiv.org/html/2506.05233v1)
[![Framework](https://img.shields.io/badge/Built%20On-Flame%20%26%20FLA-firebrick)](https://github.com/fla-org/flame)
[![License](https://img.shields.io/badge/License-MIT-green)]()

**Palimpsa** is a novel attention mechanism that views In-Context Learning (ICL) as a continual learning problem. It introduces **Bayesian Metaplasticity** to transformer architectures—dynamically adjusting the plasticity of memory states based on their uncertainty. 

This repository (`Mesa_Palimpsa`) acts as the root workspace, containing the core Palimpsa codebase alongside custom forks of the Flame training engine and the Flash Linear Attention (FLA) kernels adapted for **Mesanet**.

---
## 📂 Repository Structure

The workspace is structured to support large-scale pretraining and rapid experimentation:

```text
Mesa_Palimpsa/
├── Palimpsa/                 # Main Research Repo (layers, models, eval)
├── flame/                    # Training engine
├── flash-linear-attention/   # Custom fused kernels (Modified for Mesanet)
├── exp/                      # Dump folder for checkpoints and logs
└── palimpsa_env/             # Virtual environment (Generated during setup)
```

### 🔧 Custom FLA Modifications (Mesanet Support)

The `flash-linear-attention` submodule included here contains critical fixes required to run `mesanet` efficiently. Specifically, in `fla/ops/`:

*   **Precision Fix:** In `chunk.py`, `float16` was replaced with `bfloat16` to prevent overflow issues during chunking.
*   **ExpandV Support:** Modified several core chunking operations (`chunk_cg_solver_fwd.py`, `chunk_h_fwd.py`, `chunk_h_kv_intra_bwd_separate.py`, `chunk_h_kv_intra_bwd.py`, and `chunk.py`) so that `mesanet` fully supports `expandv > 1` arguments.
*   **Config Fix:** Fixed a minor typo in the codebase to ensure the `use_output_gate` argument is correctly parsed and passed through the model configuration.

---
## 🛠️ Installation & Setup

### 1. Workspace & Dependencies
We use `uv` for high-speed dependency management inside a standard virtual environment. Run this at the root of the repository:

```shell
# Set Up Venv
python3 -m venv palimpsa_env
source palimpsa_env/bin/activate
pip install uv

# Install Build Tools & Kernels
uv pip install ninja packaging setuptools wheel
uv pip install causal-conv1d
uv pip install -e ./flash-linear-attention
uv pip install -e ./Palimpsa
uv pip install git+[https://github.com/pytorch/torchtitan.git@0b44d4c](https://github.com/pytorch/torchtitan.git@0b44d4c)
uv pip install -e ./flame
```

### 2. Dataset Preparation (FineWeb-Edu)
Flame requires the dataset to be cached locally. **Do this only once.** 

```shell
cd Palimpsa
python3 data/download_fineweb.py --cache_dir /path/to/your/fast/storage/.cache
```

---

## 🚀 Quick Start: Shakespeare (NanoGPT)

Verify model convergence and ensure your environment is fully functional on a small scale before launching Slurm jobs.

```shell
cd Palimpsa
python3 data/shakespeare_char/prepare.py
python3 train_nano.py --model palimpsa --batch_size 16
```

---

## 🔬 Launching Training (Slurm)

To train the `mesanet-760M` model using Flame and our custom FLA kernels, use the following `sbatch` script. 

**Important:** Update the `HF_DATASETS_CACHE` path to match where you stored FineWeb-Edu, and update the W&B credentials.

```shell
#!/bin/bash
#SBATCH --job-name=FINEWEB_MPU
#SBATCH --error=runs/%x_%j.err
#SBATCH --output=runs/%x_%j.out
#SBATCH --gpus=8
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --partition=pgi15-h100

# 1. Setup Environment
cd ~/Mesa_Palimpsa
source palimpsa_env/bin/activate

# 2. Critical Exports (UPDATE THESE FOR YOUR ENVIRONMENT)
export HF_DATASETS_CACHE="/path/to/your/downloaded/cache"
export WANDB_PROJECT="Palimpsa_Google_Run"
export WANDB_NAME="mesanet-760M-test"
export OMP_NUM_THREADS=16

# 3. Launch Training
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    Palimpsa/train.py \
    --job.config_file flame/flame/models/fla.toml \
    --job.dump_folder exp/mesanet_760M_Test \
    --model.name palimpsa \
    --model.config Palimpsa/config/mesanet_760M.json \
    --model.tokenizer_path meta-llama/Llama-2-7b-chat-hf \
    --optimizer.lr 1.25e-3 \
    --lr_scheduler.warmup_steps 2000 \
    --training.batch_size 2 \
    --training.gradient_accumulation_steps 8 \
    --training.seq_len 4096 \
    --training.steps 60000 \
    --training.dataset HuggingFaceFW/fineweb-edu \
    --training.dataset_name sample-100BT \
    --training.dataset_split train \
    --training.num_workers 8 \
    --checkpoint.interval 2000 \
    --metrics.log_freq 50
```

---
## ⚖️ Model Evaluation

To evaluate distributed checkpoints (DCP) produced by the training engine:

```shell
# 1. Convert DCP to HuggingFace format
python3 Palimpsa/tools/convert_dcp_to_hf.py --exp exp/mesanet_760M_Test --step 2000

# 2. Run Benchmarks via lm-evaluation-harness
bash Palimpsa/evaluation/run_eval.sh 0 mesanet_760M_Test 2000 "wikitext,hellaswag,piqa"
```
