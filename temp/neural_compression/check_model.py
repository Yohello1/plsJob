#!/usr/bin/env python3
"""Sanity-check a trained checkpoint against the same protocol used in training.

Reports per-session metrics so you can see the spread, not just the mean --
with only a handful of validation trajectories the aggregate is very noisy.
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from compressor import SPHDataset, FullModel, get_global_stats_gpu

# Mirrors the training config from active_train_parallel.sh
FLUID_WEIGHT = 35.0
MASS_WEIGHT = 2.5
GRAD_WEIGHT = 15.0
MEAN_WEIGHT = 25.0
FLUID_THRESHOLD = 0.05


def find_session_dirs(data_dir):
    """Locate session dirs, tolerating both data/<session>/ and data/<run>/<session>/."""
    if not os.path.isdir(data_dir):
        return []

    sessions = []
    for entry in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, entry)
        if not os.path.isdir(path):
            continue
        if os.path.exists(os.path.join(path, "sim_data.bin")):
            sessions.append(path)
        else:
            # One level deeper: a run folder wrapping individual sessions
            for sub in sorted(os.listdir(path)):
                sub_path = os.path.join(path, sub)
                if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "sim_data.bin")):
                    sessions.append(sub_path)

    return sessions


def hybrid_loss(pred, target, f_weight=FLUID_WEIGHT, m_weight=MASS_WEIGHT,
                grad_weight=GRAD_WEIGHT, mean_weight=MEAN_WEIGHT):
    """Copy of the training criterion so numbers are directly comparable."""
    fluid_mask = (target[:, 0:1] > FLUID_THRESHOLD).float()
    background_mask = 1.0 - fluid_mask

    mse_fluid = torch.sum(fluid_mask * (pred - target) ** 2) / (fluid_mask.sum() * pred.shape[1] + 1e-6)
    mse_bg = torch.sum(background_mask * (pred - target) ** 2) / (background_mask.sum() * pred.shape[1] + 1e-6)
    total = f_weight * mse_fluid + mse_bg

    fn_mask = fluid_mask * (pred[:, 0:1] < FLUID_THRESHOLD).float()
    fn_loss = torch.sum(fn_mask * (target[:, 0:1] - pred[:, 0:1]) ** 2) / (fn_mask.sum() + 1e-6)
    total = total + f_weight * 5.0 * fn_loss

    pred_fluid_mean = torch.sum(fluid_mask * pred[:, 0:1]) / (fluid_mask.sum() + 1e-6)
    gt_fluid_mean = torch.sum(fluid_mask * target[:, 0:1]) / (fluid_mask.sum() + 1e-6)
    total = total + mean_weight * (pred_fluid_mean - gt_fluid_mean) ** 2

    in_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    in_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tg_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
    tg_dy = target[:, :, :, 1:] - target[:, :, :, :-1]
    f_dx = fluid_mask[:, :, 1:, :]
    f_dy = fluid_mask[:, :, :, 1:]

    grad_loss = (torch.sum(f_dx * (in_dx - tg_dx) ** 2) / (f_dx.sum() * pred.shape[1] + 1e-6) +
                 torch.sum(f_dy * (in_dy - tg_dy) ** 2) / (f_dy.sum() * pred.shape[1] + 1e-6))
    total = total + grad_weight * grad_loss

    mass_pred = torch.mean(pred[:, 0:1], dim=(1, 2, 3))
    mass_gt = torch.mean(target[:, 0:1], dim=(1, 2, 3))
    total = total + m_weight * torch.mean((mass_pred - mass_gt) ** 2)

    return total


def per_sample_var(x):
    """Variance over spatial dims only, per sample -> shape [B]."""
    return x.flatten(1).var(dim=1, unbiased=False)


@torch.no_grad()
def evaluate_session(model, dataset, device, n_steps, batch_size, max_batches, use_amp, amp_dtype):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    tot = {"loss": 0.0, "step": [0.0] * n_steps, "zero": 0.0, "ident": 0.0,
           "pred_var": 0.0, "gt_var": 0.0, "pred_d_mse": 0.0, "count": 0}

    for bi, (p_d, p_v, c_ds, c_vs, mask) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break

        p_d, p_v = p_d.to(device), p_v.to(device)
        c_ds, c_vs, mask = c_ds.to(device), c_vs.to(device), mask.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            p_d_init, p_v_init = p_d.clone(), p_v.clone()
            batch_loss = 0.0

            # Autoregressive rollout, identical to the training validation loop
            for step in range(n_steps):
                curr_d_gt, curr_v_gt = c_ds[:, step], c_vs[:, step]
                output = model(p_d, p_v, curr_d_gt, curr_v_gt, mask)
                pred_d, pred_v = output[:, 0:1], output[:, 1:3]
                target = torch.cat([curr_d_gt, curr_v_gt], dim=1)

                step_loss = hybrid_loss(output.float(), target.float())
                tot["step"][step] += step_loss.item()
                batch_loss += step_loss.item()

                p_d = pred_d
                if step + 1 < n_steps:
                    p_v = pred_v

            tot["loss"] += batch_loss / n_steps

            # Baselines on step 0, scored with the same criterion
            target0 = torch.cat([c_ds[:, 0], c_vs[:, 0]], dim=1)
            tot["zero"] += hybrid_loss(torch.zeros_like(target0).float(), target0.float()).item()
            ident = torch.cat([p_d_init, p_v_init], dim=1)
            tot["ident"] += hybrid_loss(ident.float(), target0.float()).item()

            # Variance at the SAME timestep: last rollout step vs its own GT
            gt_last = c_ds[:, n_steps - 1]
            tot["pred_var"] += per_sample_var(pred_d.float()).mean().item()
            tot["gt_var"] += per_sample_var(gt_last.float()).mean().item()
            tot["pred_d_mse"] += torch.mean((pred_d.float() - gt_last.float()) ** 2).item()
            tot["count"] += 1

    n = max(1, tot.pop("count"))
    return {k: (v / n if not isinstance(v, list) else [s / n for s in v]) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="Directory containing sessions (or a run folder wrapping them)")
    ap.add_argument("--checkpoint", default="best_model.pth")
    ap.add_argument("--n_steps", type=int, default=5, help="Must match the AR length the model was trained on")
    ap.add_argument("--skip_frames", type=int, default=5)
    ap.add_argument("--skip_initial", type=int, default=1, help=">1 subsamples frames, matching training")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_sessions", type=int, default=6, help="0 = evaluate every session found")
    ap.add_argument("--max_batches", type=int, default=8, help="0 = full pass")
    ap.add_argument("--bf16", action="store_true", help="Match the precision the checkpoint was trained in")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.bf16 or args.fp16
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    session_dirs = find_session_dirs(args.data_dir)
    if not session_dirs:
        print(f"No sessions with sim_data.bin found under {args.data_dir}")
        return
    if args.max_sessions:
        session_dirs = session_dirs[:args.max_sessions]
    print(f"Found {len(session_dirs)} session(s) under {args.data_dir}")

    if not os.path.exists(args.checkpoint):
        print(f"{args.checkpoint} not found.")
        return

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model = FullModel().to(device)
    model.load_state_dict(state_dict)
    model.to(amp_dtype if args.bf16 else torch.float32)
    print(f"Loaded {args.checkpoint} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Reproduce training's normalization exactly (compressor.py:435)
    max_d, max_v = get_global_stats_gpu(session_dirs)
    density_norm, velocity_norm = 1.0 / max_d, 1.0 / max_v
    print(f"Normalization: Density={density_norm:.6f}, Velocity={velocity_norm:.6f}")

    rows = []
    for sd in session_dirs:
        ds = SPHDataset([sd], skip=args.skip_frames, n_steps=args.n_steps,
                        skip_initial=args.skip_initial, augment=False)
        if len(ds) == 0:
            print(f"  {os.path.basename(sd)}: no usable frames, skipped")
            continue
        ds.set_norms(density_norm, velocity_norm)
        m = evaluate_session(model, ds, device, args.n_steps, args.batch_size,
                             args.max_batches, use_amp, amp_dtype)
        m["name"] = os.path.basename(sd)
        rows.append(m)

    if not rows:
        print("No sessions produced metrics.")
        return

    print("\n" + "=" * 78)
    print(f"{'session':<26}{'loss':>9}{'zero':>9}{'ident':>9}{'predVar':>10}{'gtVar':>10}{'ratio':>8}")
    print("-" * 78)
    for r in rows:
        ratio = r["pred_var"] / r["gt_var"] if r["gt_var"] > 0 else float("nan")
        print(f"{r['name']:<26}{r['loss']:>9.4f}{r['zero']:>9.4f}{r['ident']:>9.4f}"
              f"{r['pred_var']:>10.5f}{r['gt_var']:>10.5f}{ratio:>8.2f}")
    print("-" * 78)

    def agg(k):
        return float(np.mean([r[k] for r in rows]))

    steps = np.array([r["step"] for r in rows]).mean(axis=0)
    print(f"{'MEAN':<26}{agg('loss'):>9.4f}{agg('zero'):>9.4f}{agg('ident'):>9.4f}"
          f"{agg('pred_var'):>10.5f}{agg('gt_var'):>10.5f}"
          f"{agg('pred_var') / agg('gt_var') if agg('gt_var') > 0 else float('nan'):>8.2f}")

    losses = np.array([r["loss"] for r in rows])
    print(f"\nPer-session loss spread: std={losses.std():.4f}  "
          f"min={losses.min():.4f}  max={losses.max():.4f}  "
          f"({100 * losses.std() / losses.mean():.1f}% of mean)")
    print("Per-step mean loss: " + ", ".join(f"S{i + 1}: {v:.4f}" for i, v in enumerate(steps)))

    if losses.std() > 0.05 * losses.mean():
        print("\nWARNING: session-to-session spread exceeds 5% of the mean.")
        print("Differences smaller than this between epochs are not meaningful.")

    print("\nRESULT: " + ("Model beats zero-field baseline." if agg("loss") < agg("zero")
                          else "Model does NOT beat zero-field baseline."))
    print("RESULT: " + ("Model beats identity baseline." if agg("loss") < agg("ident")
                          else "Model does NOT beat identity baseline."))
    print("=" * 78)


if __name__ == "__main__":
    main()
