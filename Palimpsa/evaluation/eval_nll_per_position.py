"""
evaluation/eval_nll_per_position.py

Evaluates NLL at each token position on WikiText-103 (test split).
Strategy:
  - GPT-style: concatenate all documents with EOS separator into one long stream
  - Chunk into non-overlapping blocks of `max_length` tokens
  - For each chunk, collect per-position cross-entropy loss
  - Average across all chunks → NLL(position)
  - Save results to a local .npz file AND log to WandB

Adds (optional) GoodNet slot accounting:
  - If the model uses Palimpsa layers with token_mixer_type='simple_palimpsa'
    and self.good_net=True, the layer accumulates n_states_used into a module-
    level running total during forward. After eval we summarize and log:
        * goodnet_avg_slots_per_head    — mean n_states_used across (layers, samples, heads)
        * goodnet_baseline_per_head     — 1.0 (no-spawn reference)
        * goodnet_ratio_vs_no_spawn     — avg / 1.0
        * goodnet_total_slot_obs        — raw running sum
        * goodnet_n_observations        — total observation count
  - If the import fails (eval-ing a non-Palimpsa model), the accounting is a
    no-op and only NLL/PPL is reported. No code changes needed per-model.

Usage:
    python evaluation/eval_nll_per_position.py \
        --model_path /path/to/hf_model_step_3000 \
        --run_name palimpsa-170M-step3000 \
        --max_length 4096 \
        --wandb_project BMA-eval-djo \
        --wandb_entity hybrid_nns
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# ── Model registration (mirrors launcher.py) ──────────────────────────────────
from palimpsa.models.palimpsa import PalimpsaConfig, PalimpsaForCausalLM
from palimpsa.models.meta_mamba2 import MetaMamba2Config, MetaMamba2ForCausalLM

AutoConfig.register("palimpsa", PalimpsaConfig, exist_ok=True)
AutoModelForCausalLM.register(PalimpsaConfig, PalimpsaForCausalLM, exist_ok=True)
AutoConfig.register("meta_mamba2", MetaMamba2Config, exist_ok=True)
AutoModelForCausalLM.register(MetaMamba2Config, MetaMamba2ForCausalLM, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────

# ── GoodNet slot accounting (optional) ────────────────────────────────────────
# Adjust the import path if your Palimpsa layer module path differs.
try:
    from palimpsa.layers.palimpsa import (
        get_goodnet_slot_summary,
        reset_goodnet_slot_accounting,
    )
    _HAS_GOODNET_ACCOUNTING = True
except ImportError:
    _HAS_GOODNET_ACCOUNTING = False

    def get_goodnet_slot_summary():
        return {}

    def reset_goodnet_slot_accounting():
        pass
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path to HF model folder (config.json + weights)")
    p.add_argument("--tokenizer_path", default=None,
                   help="Path to tokenizer. Defaults to model_path.")
    p.add_argument("--run_name", default=None,
                   help="WandB run name and output file prefix. Defaults to model folder name.")
    p.add_argument("--max_length", type=int, default=4096,
                   help="Sequence length to chunk at (should match training length).")
    p.add_argument("--output_path", default=None,
                   help="Directory for .npz and .json outputs. Defaults to model_path/../")
    p.add_argument("--dataset", default="wikitext",
                   choices=["wikitext", "wikitext-2"])
    p.add_argument("--dataset_split", default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    # WandB
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_job_type", default="nll_per_pos_eval")
    p.add_argument("--max_chunks", type=int, default=None,
                   help="Max number of chunks to evaluate (None = all).")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def build_token_stream(dataset_name, split, tokenizer, eos_token_id):
    dataset_config = "wikitext-103-raw-v1" if dataset_name == "wikitext" else "wikitext-2-raw-v1"
    print(f"Loading {dataset_config} ({split} split)...")
    ds = load_dataset("wikitext", dataset_config, split=split)

    all_ids = []
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos_token_id)

    print(f"Total tokens in stream: {len(all_ids):,}")
    return np.array(all_ids, dtype=np.int32)


def make_chunks(token_stream, max_length):
    n_chunks = len(token_stream) // max_length
    trimmed = token_stream[: n_chunks * max_length]
    chunks = trimmed.reshape(n_chunks, max_length)
    print(f"Number of chunks ({max_length}-token): {n_chunks:,}")
    return chunks


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, chunks, batch_size, device, max_length):
    model.eval()
    nll_sum   = np.zeros(max_length - 1, dtype=np.float64)
    nll_count = np.zeros(max_length - 1, dtype=np.int64)

    # Clear any stale running totals from prior imports/runs in the same process.
    reset_goodnet_slot_accounting()

    n_chunks  = len(chunks)
    n_batches = (n_chunks + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, n_chunks)
        batch = chunks[start:end]

        input_ids = torch.tensor(batch, dtype=torch.long, device=device)

        outputs = model(input_ids=input_ids, use_cache=False)
        logits  = outputs.logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        B, L, V = shift_logits.shape

        per_token_nll = F.cross_entropy(
            shift_logits.reshape(B * L, V),
            shift_labels.reshape(B * L),
            reduction="none",
        ).reshape(B, L).cpu().numpy()

        nll_sum[:L]   += per_token_nll.sum(axis=0)
        nll_count[:L] += B

        if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
            print(f"  batch {batch_idx+1}/{n_batches} done")

    return nll_sum, nll_count


# ── WandB ─────────────────────────────────────────────────────────────────────

def log_to_wandb(nll_mean, slot_summary, run_name, args, metadata):
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        job_type=args.wandb_job_type,
        config={**metadata, "model_path": args.model_path},
    )

    positions = np.arange(1, len(nll_mean) + 1)
    ppl_mean  = np.exp(nll_mean)

    # Line chart with token_position as x-axis
    for pos, nll, ppl in zip(positions, nll_mean, ppl_mean):
        run.log({"nll": float(nll), "ppl": float(ppl)}, step=int(pos))

    # Downloadable table
    table = wandb.Table(columns=["token_position", "nll", "ppl"])
    for pos, nll, ppl in zip(positions, nll_mean, ppl_mean):
        table.add_data(int(pos), float(nll), float(ppl))
    run.log({"nll_per_position_table": table})

    run.summary["mean_nll"] = float(np.nanmean(nll_mean))
    run.summary["mean_ppl"] = float(np.exp(np.nanmean(nll_mean)))

    # GoodNet slot diagnostic — single numbers per run, kept on run.summary so
    # they don't get tied to a step (cleaner for cross-run comparison).
    for k, v in slot_summary.items():
        run.summary[k] = v

    run.finish()
    print(f"✅ Logged to WandB run: {run_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype  = dtype_map[args.dtype]
    device = torch.device(args.device)

    model_path     = args.model_path
    tokenizer_path = args.tokenizer_path or model_path
    run_name       = args.run_name or Path(model_path).name

    # ── Load tokenizer & model ────────────────────────────────────────────────
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=False,
    ).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.1f}M  |  max_length: {args.max_length}")

    eos_token_id = tokenizer.eos_token_id

    # ── Data ──────────────────────────────────────────────────────────────────
    token_stream = build_token_stream(args.dataset, args.dataset_split, tokenizer, eos_token_id)
    chunks = make_chunks(token_stream, args.max_length)
    if args.max_chunks is not None:
        chunks = chunks[:args.max_chunks]
        print(f"Using {len(chunks)} chunks (limited)")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f"\nRunning evaluation (batch_size={args.batch_size}, dtype={args.dtype})...")
    nll_sum, nll_count = evaluate(model, chunks, args.batch_size, device, args.max_length)

    valid    = nll_count > 0
    nll_mean = np.where(valid, nll_sum / np.maximum(nll_count, 1), np.nan)
    positions = np.arange(1, args.max_length)

    aggregate_nll = float(np.nanmean(nll_mean))
    aggregate_ppl = float(np.exp(aggregate_nll))
    print(f"\n📊 Aggregate NLL : {aggregate_nll:.4f}")
    print(f"📊 Aggregate PPL : {aggregate_ppl:.2f}")

    # ── GoodNet slot summary ──────────────────────────────────────────────────
    slot_summary = get_goodnet_slot_summary()
    if slot_summary and slot_summary.get("goodnet_n_observations", 0) > 0:
        print("\n🧠 GoodNet slot accounting:")
        for k, v in slot_summary.items():
            print(f"  {k:<32s} {v}")
    else:
        print("\n(No GoodNet slot stats — model didn't use the GoodNet path.)")
        if not _HAS_GOODNET_ACCOUNTING:
            print("  (palimpsa.layers.palimpsa not importable — accounting hooks not registered)")

    # ── Save locally ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_path) if args.output_path else Path(model_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path  = out_dir / f"{run_name}_nll_per_pos.npz"
    json_path = out_dir / f"{run_name}_nll_per_pos_meta.json"

    np.savez(npz_path, positions=positions, nll_mean=nll_mean,
             nll_sum=nll_sum, nll_count=nll_count)

    metadata = {
        "run_name":      run_name,
        "model_path":    model_path,
        "dataset":       args.dataset,
        "split":         args.dataset_split,
        "max_length":    args.max_length,
        "n_chunks":      int(len(chunks)),
        "total_tokens":  int(len(token_stream)),
        "aggregate_nll": aggregate_nll,
        "aggregate_ppl": aggregate_ppl,
        "dtype":         args.dtype,
        "n_params_M":    round(n_params / 1e6, 1),
        **slot_summary,
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved: {npz_path}")
    print(f"✅ Saved: {json_path}")

    # ── WandB ──────────────────────────────────────────────────────────────────
    if args.wandb_project:
        log_to_wandb(nll_mean, slot_summary, run_name, args, metadata)
    else:
        print("\nSkipping WandB (no --wandb_project provided).")


if __name__ == "__main__":
    main()
