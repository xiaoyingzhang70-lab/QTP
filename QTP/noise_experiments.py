# -*- coding: utf-8 -*-
"""
noise_experiments.py
====================
Dedicated noise-study script for the Lorenz dataset.

This file separates two experimental regimes clearly:

1. gate_noise_only():
   Deterministic gate-noise study with default.mixed and shots=None.
   This isolates the effect of gate-channel noise only.

2. gate_noise_finite_shot():
   Gate noise + finite-shot measurement fluctuation, using the same
   clean model as the first experiment and without retraining.

Outputs:
- CSV summaries
- NPY dictionaries
- PNG/PDF figures
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import mean_squared_error, r2_score

from DataLoader import DataConfig, DataProcessor
from Model import DEVICE, ModelConfig
from noise_model_overrides import SymmetricGateNoiseQTP
from Trainer import SEED, seed_everything


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

DATASET = "mackey"
WINDOW = 12
K_RATIO = 0.33
QTP_LAYERS = 1
K = max(1, round(WINDOW * K_RATIO))

NOISE_TYPES = ["depolarizing", "bit_flip", "phase_flip", "amplitude_damp"]
NOISE_PROBS = [
    0.0,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.008,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,
    0.080,
    0.100,
]
LOW_NOISE_MAX = 0.02
MEAS_SEEDS = [11, 22, 33, 44, 55, 66, 77]
SHOTS = 2048

RESULT_ROOT = SCRIPT_DIR / "result" / "noise_study_mackey"
GATE_ONLY_ROOT = RESULT_ROOT / "gate_noise_only"
FINITE_SHOT_ROOT = RESULT_ROOT / "gate_noise_finite_shot"

CLEAN_CANDIDATES = [
    SCRIPT_DIR / "result" / "planC_mackey" / "clean_model" / "best_model.pt",
    SCRIPT_DIR / "result" / "planC_mackey" / "clean_model" / "best_model.pt",
    SCRIPT_DIR / "result" / "planB" / "QTP_mackey_W12_K4_L1_P36" / "best_model.pt",
]

COLORS = {
    "depolarizing": "#2E7D32",
    "bit_flip": "#A5A5A5",
    "phase_flip": "#5A4D5C",
    "amplitude_damp": "#C87A3E",
}

LABELS = {
    "depolarizing": "Depolarizing",
    "bit_flip": "Bit flip",
    "phase_flip": "Phase flip",
    "amplitude_damp": "Amplitude damping",
}


def _metric_summary(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "lo": float(np.percentile(arr, 16)),
        "hi": float(np.percentile(arr, 84)),
        "lo95": float(np.percentile(arr, 2.5)),
        "hi95": float(np.percentile(arr, 97.5)),
        "n": int(len(arr)),
    }


def _save_csv(rows, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_csv_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_clean_model_and_testset():
    seed_everything(SEED)
    data_cfg = DataConfig(dataname=DATASET, window_size=WINDOW, use_pos=True)
    processor = DataProcessor(data_cfg)
    (_, _), (_, _), (x_te, y_te) = processor.get_dataset()
    nf = processor.n_features

    model_cfg = ModelConfig(windows=WINDOW, n_features=nf, layers=QTP_LAYERS, K=K)
    model = SymmetricGateNoiseQTP(model_cfg).to(DEVICE).double()

    clean_weights = None
    for candidate in CLEAN_CANDIDATES:
        if candidate.exists():
            clean_weights = candidate
            break
    if clean_weights is None:
        tried = "\n".join(str(p) for p in CLEAN_CANDIDATES)
        raise FileNotFoundError(
            "Could not find a clean Lorenz checkpoint. Tried:\n" + tried
        )

    model.load_state_dict(
        torch.load(clean_weights, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    x_te = x_te.to(dtype=torch.float64)
    y_te = y_te.to(dtype=torch.float64)
    return model, x_te, y_te, clean_weights


def _predict_noisy_batch(model, x_te, noise_type, noise_p, shots=None, meas_seed=None):
    noisy_net = model._build_noisy_circuit(shots=shots, meas_seed=meas_seed)
    preds = []
    with torch.no_grad():
        for i in range(len(x_te)):
            pred = model.forward_noisy(
                x_te[i],
                noise_type=noise_type,
                noise_p=noise_p,
                shots=shots,
                meas_seed=meas_seed,
                noisy_net=noisy_net,
            )
            preds.append(pred.detach().cpu().view(-1))
    return torch.cat(preds, dim=0).numpy()


def _make_r2_plot(summary, output_root: Path, title: str, with_band: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_r2 = summary[("depolarizing", 0.0)]["r2"]["mean"]

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    for noise_type in NOISE_TYPES:
        xs = np.asarray(NOISE_PROBS, dtype=float)
        mean = np.asarray(
            [summary[(noise_type, p)]["r2"]["mean"] for p in NOISE_PROBS], dtype=float
        )

        if with_band:
            lo = np.asarray(
                [summary[(noise_type, p)]["r2"]["lo"] for p in NOISE_PROBS], dtype=float
            )
            hi = np.asarray(
                [summary[(noise_type, p)]["r2"]["hi"] for p in NOISE_PROBS], dtype=float
            )
            ax.fill_between(xs, lo, hi, color=COLORS[noise_type], alpha=0.18, linewidth=0)

        ax.plot(
            xs,
            mean,
            "o-",
            color=COLORS[noise_type],
            label=LABELS[noise_type],
            lw=2.0,
            ms=5.8,
            mec="black",
            mew=0.45,
        )

    ax.axhline(
        baseline_r2,
        color="gray",
        linestyle="--",
        lw=1.1,
        label=f"No noise ($R^2$={baseline_r2:.4f})",
    )
    ax.set_xlabel("Noise probability $p$")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(title)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    low_ps = [p for p in NOISE_PROBS if p <= LOW_NOISE_MAX]
    axins = inset_axes(ax, width="41%", height="41%", loc="lower left", borderpad=1.15)
    for noise_type in NOISE_TYPES:
        xs = np.asarray(low_ps, dtype=float)
        mean = np.asarray(
            [summary[(noise_type, p)]["r2"]["mean"] for p in low_ps], dtype=float
        )
        if with_band:
            lo = np.asarray(
                [summary[(noise_type, p)]["r2"]["lo"] for p in low_ps], dtype=float
            )
            hi = np.asarray(
                [summary[(noise_type, p)]["r2"]["hi"] for p in low_ps], dtype=float
            )
            axins.fill_between(xs, lo, hi, color=COLORS[noise_type], alpha=0.16, linewidth=0)
        axins.plot(xs, mean, "o-", color=COLORS[noise_type], lw=1.55, ms=3.8)

    axins.axhline(baseline_r2, color="gray", linestyle="--", lw=0.9)
    axins.set_xlim(0.0, LOW_NOISE_MAX)
    axins.set_title("Low-noise", fontsize=8)
    axins.grid(True, alpha=0.18)
    axins.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(output_root / "r2_curve.png", dpi=180)
    plt.savefig(output_root / "r2_curve.pdf")
    plt.close(fig)


def redraw_saved_noise_plot(
    experiment_root: Path = FINITE_SHOT_ROOT,
    title: str | None = None,
    band_mode: str = "minmax",
    show_raw: bool = True,
):
    """
    Reload saved results and redraw a more visible uncertainty band.

    Parameters
    ----------
    experiment_root:
        Folder containing raw_records.csv and summary.csv.
    band_mode:
        "minmax" -> raw min/max envelope across meas_seed
        "pct95"  -> 2.5%-97.5% envelope from summary.csv
        "pct68"  -> 16%-84% envelope from summary.csv
    show_raw:
        Whether to overlay raw finite-shot points.
    """
    experiment_root = Path(experiment_root)
    raw_path = experiment_root / "raw_records.csv"
    summary_path = experiment_root / "summary.csv"
    out_png = experiment_root / "r2_curve_reloaded.png"
    out_pdf = experiment_root / "r2_curve_reloaded.pdf"

    raw_rows = _load_csv_rows(raw_path) if raw_path.exists() else []
    summary_rows = _load_csv_rows(summary_path)

    summary_map = {}
    for row in summary_rows:
        key = (row["noise_type"], float(row["noise_p"]))
        summary_map[key] = row

    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["noise_type"], float(row["noise_p"]))
        grouped[key].append(float(row["r2"]))

    if title is None:
        title = f"Gate noise + finite-shot robustness - {DATASET} (reloaded)"

    baseline_r2 = float(summary_map[("depolarizing", 0.0)]["r2_mean"])
    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    for noise_type in NOISE_TYPES:
        xs = np.asarray(NOISE_PROBS, dtype=float)
        mean = np.asarray(
            [float(summary_map[(noise_type, p)]["r2_mean"]) for p in NOISE_PROBS],
            dtype=float,
        )

        if band_mode == "minmax" and raw_rows:
            lo = np.asarray([min(grouped[(noise_type, p)]) for p in NOISE_PROBS], dtype=float)
            hi = np.asarray([max(grouped[(noise_type, p)]) for p in NOISE_PROBS], dtype=float)
        elif band_mode == "pct95":
            lo = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_lo95"]) for p in NOISE_PROBS],
                dtype=float,
            )
            hi = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_hi95"]) for p in NOISE_PROBS],
                dtype=float,
            )
        else:
            lo = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_lo"]) for p in NOISE_PROBS],
                dtype=float,
            )
            hi = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_hi"]) for p in NOISE_PROBS],
                dtype=float,
            )

        ax.fill_between(xs, lo, hi, color=COLORS[noise_type], alpha=0.25, linewidth=0)
        ax.plot(
            xs,
            mean,
            "o-",
            color=COLORS[noise_type],
            label=LABELS[noise_type],
            lw=2.0,
            ms=5.8,
            mec="black",
            mew=0.45,
            zorder=3,
        )

        if show_raw and raw_rows:
            for p in NOISE_PROBS:
                ys = grouped[(noise_type, p)]
                xj = np.full(len(ys), p, dtype=float)
                ax.scatter(
                    xj,
                    ys,
                    s=14,
                    color=COLORS[noise_type],
                    alpha=0.35,
                    edgecolors="none",
                    zorder=2,
                )

    ax.axhline(
        baseline_r2,
        color="gray",
        linestyle="--",
        lw=1.1,
        label=f"No noise ($R^2$={baseline_r2:.4f})",
    )
    ax.set_xlabel("Noise probability $p$")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(title)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    low_ps = [p for p in NOISE_PROBS if p <= LOW_NOISE_MAX]
    axins = inset_axes(ax, width="41%", height="41%", loc="lower left", borderpad=1.15)
    for noise_type in NOISE_TYPES:
        xs = np.asarray(low_ps, dtype=float)
        mean = np.asarray(
            [float(summary_map[(noise_type, p)]["r2_mean"]) for p in low_ps], dtype=float
        )

        if band_mode == "minmax" and raw_rows:
            lo = np.asarray([min(grouped[(noise_type, p)]) for p in low_ps], dtype=float)
            hi = np.asarray([max(grouped[(noise_type, p)]) for p in low_ps], dtype=float)
        elif band_mode == "pct95":
            lo = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_lo95"]) for p in low_ps],
                dtype=float,
            )
            hi = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_hi95"]) for p in low_ps],
                dtype=float,
            )
        else:
            lo = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_lo"]) for p in low_ps],
                dtype=float,
            )
            hi = np.asarray(
                [float(summary_map[(noise_type, p)]["r2_hi"]) for p in low_ps],
                dtype=float,
            )

        axins.fill_between(xs, lo, hi, color=COLORS[noise_type], alpha=0.22, linewidth=0)
        axins.plot(xs, mean, "o-", color=COLORS[noise_type], lw=1.55, ms=3.8)

        if show_raw and raw_rows:
            for p in low_ps:
                ys = grouped[(noise_type, p)]
                xj = np.full(len(ys), p, dtype=float)
                axins.scatter(
                    xj,
                    ys,
                    s=10,
                    color=COLORS[noise_type],
                    alpha=0.30,
                    edgecolors="none",
                    zorder=2,
                )

    axins.axhline(baseline_r2, color="gray", linestyle="--", lw=0.9)
    axins.set_xlim(0.0, LOW_NOISE_MAX)
    axins.set_title("Low-noise", fontsize=8)
    axins.grid(True, alpha=0.18)
    axins.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"Reloaded plot saved to:\n  {out_png}\n  {out_pdf}")


def redraw_saved_noise_plot_with_spread_panel(
    experiment_root: Path = FINITE_SHOT_ROOT,
    title_left: str | None = None,
    title_right: str = "Finite-shot spread around mean",
    spread_scale: float = 1e3,
):
    """
    Redraw finite-shot results with:
    1) the usual R^2 robustness panel
    2) a second panel that explicitly shows the spread around the mean

    The second panel plots (raw_r2 - mean_r2) * spread_scale, which makes
    the width visible without faking uncertainty in the main plot.
    """
    experiment_root = Path(experiment_root)
    raw_path = experiment_root / "raw_records.csv"
    summary_path = experiment_root / "summary.csv"
    out_png = experiment_root / "r2_curve_with_spread.png"
    out_pdf = experiment_root / "r2_curve_with_spread.pdf"

    raw_rows = _load_csv_rows(raw_path)
    summary_rows = _load_csv_rows(summary_path)

    summary_map = {}
    for row in summary_rows:
        key = (row["noise_type"], float(row["noise_p"]))
        summary_map[key] = row

    grouped = defaultdict(list)
    for row in raw_rows:
        key = (row["noise_type"], float(row["noise_p"]))
        grouped[key].append(float(row["r2"]))

    if title_left is None:
        title_left = f"Gate noise + finite-shot robustness - {DATASET}"

    baseline_r2 = float(summary_map[("depolarizing", 0.0)]["r2_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Panel 1: standard robustness curve
    ax = axes[0]
    for noise_type in NOISE_TYPES:
        xs = np.asarray(NOISE_PROBS, dtype=float)
        mean = np.asarray(
            [float(summary_map[(noise_type, p)]["r2_mean"]) for p in NOISE_PROBS],
            dtype=float,
        )
        lo = np.asarray([min(grouped[(noise_type, p)]) for p in NOISE_PROBS], dtype=float)
        hi = np.asarray([max(grouped[(noise_type, p)]) for p in NOISE_PROBS], dtype=float)

        ax.fill_between(xs, lo, hi, color=COLORS[noise_type], alpha=0.24, linewidth=0)
        ax.errorbar(
            xs,
            mean,
            yerr=[mean - lo, hi - mean],
            fmt="o-",
            color=COLORS[noise_type],
            ecolor=COLORS[noise_type],
            elinewidth=1.1,
            capsize=2.2,
            lw=2.0,
            ms=5.6,
            mec="black",
            mew=0.45,
            label=LABELS[noise_type],
            zorder=3,
        )

    ax.axhline(
        baseline_r2,
        color="gray",
        linestyle="--",
        lw=1.1,
        label=f"No noise ($R^2$={baseline_r2:.4f})",
    )
    ax.set_xlabel("Noise probability $p$")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(title_left)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    axins = inset_axes(ax, width="41%", height="41%", loc="lower left", borderpad=1.15)
    low_ps = [p for p in NOISE_PROBS if p <= LOW_NOISE_MAX]
    for noise_type in NOISE_TYPES:
        xs = np.asarray(low_ps, dtype=float)
        mean = np.asarray(
            [float(summary_map[(noise_type, p)]["r2_mean"]) for p in low_ps], dtype=float
        )
        lo = np.asarray([min(grouped[(noise_type, p)]) for p in low_ps], dtype=float)
        hi = np.asarray([max(grouped[(noise_type, p)]) for p in low_ps], dtype=float)
        axins.fill_between(xs, lo, hi, color=COLORS[noise_type], alpha=0.22, linewidth=0)
        axins.errorbar(
            xs,
            mean,
            yerr=[mean - lo, hi - mean],
            fmt="o-",
            color=COLORS[noise_type],
            ecolor=COLORS[noise_type],
            elinewidth=0.8,
            capsize=1.8,
            lw=1.4,
            ms=3.8,
        )
    axins.axhline(baseline_r2, color="gray", linestyle="--", lw=0.9)
    axins.set_xlim(0.0, LOW_NOISE_MAX)
    axins.set_title("Low-noise", fontsize=8)
    axins.grid(True, alpha=0.18)
    axins.tick_params(labelsize=7)

    # Panel 2: explicit spread display
    ax = axes[1]
    for noise_type in NOISE_TYPES:
        xs_mean = []
        spread_mean = []
        spread_lo = []
        spread_hi = []

        for p in NOISE_PROBS:
            ys = np.asarray(grouped[(noise_type, p)], dtype=float)
            mu = ys.mean()
            delta = (ys - mu) * spread_scale

            # small deterministic x-jitter to avoid complete overlap
            if len(delta) > 1:
                jitter = np.linspace(-0.00045, 0.00045, len(delta))
            else:
                jitter = np.zeros_like(delta)

            ax.scatter(
                np.full(len(delta), p, dtype=float) + jitter,
                delta,
                s=18,
                color=COLORS[noise_type],
                alpha=0.42,
                edgecolors="none",
            )

            xs_mean.append(p)
            spread_mean.append(0.0)
            spread_lo.append(delta.min())
            spread_hi.append(delta.max())

        xs_mean = np.asarray(xs_mean, dtype=float)
        spread_mean = np.asarray(spread_mean, dtype=float)
        spread_lo = np.asarray(spread_lo, dtype=float)
        spread_hi = np.asarray(spread_hi, dtype=float)

        ax.fill_between(xs_mean, spread_lo, spread_hi, color=COLORS[noise_type], alpha=0.18, linewidth=0)
        ax.plot(xs_mean, spread_mean, color=COLORS[noise_type], lw=1.4, alpha=0.9)

    ax.axhline(0.0, color="gray", linestyle="--", lw=1.0)
    ax.set_xlabel("Noise probability $p$")
    ax.set_ylabel(rf"$(R^2 - \overline{{R^2}})\times {int(spread_scale)}$")
    ax.set_title(title_right)
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.28)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close(fig)
    print(f"Reloaded spread plot saved to:\n  {out_png}\n  {out_pdf}")


def gate_noise_only():
    """
    Deterministic gate-noise-only experiment.
    Uses default.mixed with shots=None and no meas_seed.
    """
    print("\n" + "=" * 68)
    print("gate_noise_only: deterministic gate-channel robustness on Lorenz")
    print("=" * 68)

    output_root = GATE_ONLY_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    model, x_te, y_te, clean_weights = _load_clean_model_and_testset()
    yt = y_te.detach().cpu().view(-1).numpy()
    print(f"Loaded clean checkpoint: {clean_weights}")

    summary = {}
    summary_rows = []

    for noise_type in NOISE_TYPES:
        print(f"\nNoise type: {noise_type}")
        for p in NOISE_PROBS:
            yp = _predict_noisy_batch(
                model=model,
                x_te=x_te,
                noise_type=noise_type,
                noise_p=p,
                shots=None,
                meas_seed=None,
            )
            r2 = r2_score(yt, yp)
            mse = mean_squared_error(yt, yp)
            summary[(noise_type, p)] = {
                "r2": {"mean": float(r2), "std": 0.0, "lo": float(r2), "hi": float(r2), "n": 1},
                "mse": {"mean": float(mse), "std": 0.0, "lo": float(mse), "hi": float(mse), "n": 1},
            }
            summary_rows.append(
                {
                    "dataset": DATASET,
                    "noise_type": noise_type,
                    "noise_p": p,
                    "shots": "None",
                    "meas_seed": "None",
                    "r2": r2,
                    "mse": mse,
                }
            )
            print(f"  p={p:>7.4f}  R^2={r2:>9.6f}  MSE={mse:>10.6f}")

    np.save(output_root / "summary.npy", summary, allow_pickle=True)
    _save_csv(summary_rows, output_root / "summary.csv")
    _make_r2_plot(
        summary=summary,
        output_root=output_root,
        title=f"Gate-noise-only robustness - {DATASET}",
        with_band=False,
    )
    print(f"\nSaved gate-noise-only outputs to: {output_root}")


def gate_noise_finite_shot():
    """
    Gate noise + finite-shot measurement fluctuation.
    Uses the same clean model as gate_noise_only() and does not retrain.
    """
    print("\n" + "=" * 68)
    print("gate_noise_finite_shot: gate noise + finite-shot fluctuation on Lorenz")
    print("=" * 68)

    output_root = FINITE_SHOT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    model, x_te, y_te, clean_weights = _load_clean_model_and_testset()
    yt = y_te.detach().cpu().view(-1).numpy()
    print(f"Loaded clean checkpoint: {clean_weights}")
    print(f"Finite-shot setting: shots={SHOTS}, meas_seeds={MEAS_SEEDS}")

    raw_rows = []
    grouped_r2 = defaultdict(list)
    grouped_mse = defaultdict(list)
    total_runs = len(NOISE_TYPES) * len(NOISE_PROBS) * len(MEAS_SEEDS)
    run_idx = 0

    for noise_type in NOISE_TYPES:
        print(f"\nNoise type: {noise_type}")
        for p in NOISE_PROBS:
            for meas_seed in MEAS_SEEDS:
                run_idx += 1
                yp = _predict_noisy_batch(
                    model=model,
                    x_te=x_te,
                    noise_type=noise_type,
                    noise_p=p,
                    shots=SHOTS,
                    meas_seed=meas_seed,
                )
                r2 = r2_score(yt, yp)
                mse = mean_squared_error(yt, yp)
                grouped_r2[(noise_type, p)].append(r2)
                grouped_mse[(noise_type, p)].append(mse)
                raw_rows.append(
                    {
                        "dataset": DATASET,
                        "shots": SHOTS,
                        "noise_type": noise_type,
                        "noise_p": p,
                        "meas_seed": meas_seed,
                        "r2": r2,
                        "mse": mse,
                    }
                )
                print(
                    f"  [{run_idx:03d}/{total_runs}] "
                    f"p={p:>7.4f}  seed={meas_seed:>3d}  "
                    f"R^2={r2:>9.6f}  MSE={mse:>10.6f}"
                )

    summary = {}
    summary_rows = []
    for noise_type in NOISE_TYPES:
        for p in NOISE_PROBS:
            r2_stats = _metric_summary(grouped_r2[(noise_type, p)])
            mse_stats = _metric_summary(grouped_mse[(noise_type, p)])
            summary[(noise_type, p)] = {"r2": r2_stats, "mse": mse_stats}
            summary_rows.append(
                {
                    "dataset": DATASET,
                    "shots": SHOTS,
                    "noise_type": noise_type,
                    "noise_p": p,
                    "n_runs": r2_stats["n"],
                    "r2_mean": r2_stats["mean"],
                    "r2_std": r2_stats["std"],
                    "r2_lo": r2_stats["lo"],
                    "r2_hi": r2_stats["hi"],
                    "r2_lo95": r2_stats["lo95"],
                    "r2_hi95": r2_stats["hi95"],
                    "mse_mean": mse_stats["mean"],
                    "mse_std": mse_stats["std"],
                    "mse_lo": mse_stats["lo"],
                    "mse_hi": mse_stats["hi"],
                }
            )

    np.save(output_root / "summary.npy", summary, allow_pickle=True)
    _save_csv(raw_rows, output_root / "raw_records.csv")
    _save_csv(summary_rows, output_root / "summary.csv")
    _make_r2_plot(
        summary=summary,
        output_root=output_root,
        title=f"Gate noise + finite-shot robustness - {DATASET}",
        with_band=True,
    )
    print(f"\nSaved gate-noise+finite-shot outputs to: {output_root}")


def run_all():
    gate_noise_only()
    # gate_noise_finite_shot()
    # redraw_saved_noise_plot(experiment_root="D:\\Codex\\test02\\code\\result\\noise_study_lorenz\\gate_noise_finite_shot")
    # redraw_saved_noise_plot_with_spread_panel(experiment_root="D:\\Codex\\test02\\code\\result\\noise_study_lorenz\\gate_noise_finite_shot")
if __name__ == "__main__":
    run_all()
