# -*- coding: utf-8 -*-
"""
Mechanism validation experiments for QTP.

This script creates three toy datasets that isolate:
  1. temporal order sensitivity (q_pos / time-flow),
  2. sparse long-range marker memory (q_his / snapshot-injection),
  3. multivariate feature interaction (direct feature embedding + entanglement).

It follows the existing project structure and reuses Model.py / Trainer.py.
"""

import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Model import ModelConfig, QuantumTemporalPredictor, build_model, DEVICE
from Trainer import Trainer, seed_everything


WINDOW = 12
K_RATIO = 0.33
QTP_LAYERS = 1
EPOCHS = 35
LR_QTP = 5e-2
LR_VQC = 1e-2
BATCH_SIZE = 16
PATIENCE = 6
SEEDS = [42, 52, 62]
RESULT_ROOT = os.path.join(ROOT_DIR, "result", "mechanism_toys")


@dataclass
class ToyBundle:
    name: str
    n_features: int
    x_qtp_tr: torch.Tensor
    x_qtp_va: torch.Tensor
    x_qtp_te: torch.Tensor
    x_qtp_const_tr: torch.Tensor
    x_qtp_const_va: torch.Tensor
    x_qtp_const_te: torch.Tensor
    x_plain_tr: torch.Tensor
    x_plain_va: torch.Tensor
    x_plain_te: torch.Tensor
    y_tr: torch.Tensor
    y_va: torch.Tensor
    y_te: torch.Tensor
    y_te_raw: np.ndarray
    y_mu: float
    y_std: float


def _build_pos(window: int) -> np.ndarray:
    return np.linspace(-0.9 * np.pi, 0.9 * np.pi, window, dtype=np.float32).reshape(-1, 1)


def _zscore_from_train(arr: np.ndarray, n_train: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = arr[:n_train].mean(axis=0, keepdims=True)
    std = arr[:n_train].std(axis=0, keepdims=True) + 1e-8
    return (arr - mu) / std, mu, std


def _make_windows_from_features(
    feat_raw: np.ndarray,
    targets_raw: np.ndarray,
    window: int,
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    const_pos_value: float = np.pi / 2,
) -> ToyBundle:
    n_total = feat_raw.shape[0]
    idx_train = int(n_total * split[0])

    feat_z, feat_mu, feat_std = _zscore_from_train(feat_raw, idx_train)
    y_z, y_mu_arr, y_std_arr = _zscore_from_train(targets_raw.reshape(-1, 1), idx_train)
    y_z = y_z.reshape(-1)
    y_mu = float(y_mu_arr.reshape(-1)[0])
    y_std = float(y_std_arr.reshape(-1)[0])

    pos = _build_pos(window)
    pos_const = np.full((window, 1), const_pos_value, dtype=np.float32)

    x_plain = []
    x_qtp = []
    x_qtp_const = []
    ys = []
    ys_raw = []

    feat_angle = (2.0 * np.arctan(feat_z)).astype(np.float32)
    feat_plain = feat_z.astype(np.float32)

    for i in range(n_total - window):
        win_plain = feat_plain[i:i + window].copy()
        win_qtp = feat_angle[i:i + window].copy()

        x_plain.append(win_plain)
        x_qtp.append(np.concatenate([pos, win_qtp], axis=1))
        x_qtp_const.append(np.concatenate([pos_const, win_qtp], axis=1))
        ys.append(y_z[i + window])
        ys_raw.append(targets_raw[i + window])

    x_plain = np.asarray(x_plain, dtype=np.float32)
    x_qtp = np.asarray(x_qtp, dtype=np.float32)
    x_qtp_const = np.asarray(x_qtp_const, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32).reshape(-1, 1)
    ys_raw = np.asarray(ys_raw, dtype=np.float32).reshape(-1, 1)

    n_windows = len(x_plain)
    i1 = int(n_windows * split[0])
    i2 = int(n_windows * (split[0] + split[1]))

    return (
        torch.tensor(x_qtp[:i1]),
        torch.tensor(x_qtp[i1:i2]),
        torch.tensor(x_qtp[i2:]),
        torch.tensor(x_qtp_const[:i1]),
        torch.tensor(x_qtp_const[i1:i2]),
        torch.tensor(x_qtp_const[i2:]),
        torch.tensor(x_plain[:i1]),
        torch.tensor(x_plain[i1:i2]),
        torch.tensor(x_plain[i2:]),
        torch.tensor(ys[:i1]),
        torch.tensor(ys[i1:i2]),
        torch.tensor(ys[i2:]),
        ys_raw[i2:].reshape(-1),
        y_mu,
        y_std,
    )


def make_toy_order(T: int = 1200, window: int = WINDOW) -> ToyBundle:
    rng = np.random.default_rng(123)
    x = np.zeros(T, dtype=np.float32)
    eps = rng.normal(0.0, 0.35, size=T).astype(np.float32)
    for t in range(1, T):
        x[t] = 0.72 * x[t - 1] + eps[t]

    alpha = np.linspace(-1.0, 1.0, window, dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    for t in range(window, T):
        win = x[t - window:t]
        y[t] = float(np.dot(alpha, win) + 0.15 * np.tanh(win[-1]))

    feat = x.reshape(-1, 1)
    data = _make_windows_from_features(feat, y, window)
    return ToyBundle(
        name="order",
        n_features=1,
        x_qtp_tr=data[0], x_qtp_va=data[1], x_qtp_te=data[2],
        x_qtp_const_tr=data[3], x_qtp_const_va=data[4], x_qtp_const_te=data[5],
        x_plain_tr=data[6], x_plain_va=data[7], x_plain_te=data[8],
        y_tr=data[9], y_va=data[10], y_te=data[11],
        y_te_raw=data[12], y_mu=data[13], y_std=data[14],
    )


def make_toy_marker(T: int = 1400, window: int = WINDOW, K: int = 4) -> ToyBundle:
    rng = np.random.default_rng(456)
    marker = np.zeros(T, dtype=np.float32)
    value = np.zeros(T, dtype=np.float32)
    signs = np.ones(T, dtype=np.float32)

    current_sign = 1.0
    for t in range(T):
        if t % K == 0:
            current_sign = rng.choice([-1.0, 1.0])
            marker[t] = current_sign + rng.normal(0.0, 0.03)
        else:
            marker[t] = rng.normal(0.0, 0.03)
        signs[t] = current_sign

    eps = rng.normal(0.0, 0.20, size=T).astype(np.float32)
    for t in range(1, T):
        value[t] = 0.82 * value[t - 1] + eps[t]

    y = np.zeros(T, dtype=np.float32)
    for t in range(window, T):
        last_idx = max([j for j in range(t - window, t) if j % K == 0])
        gate = signs[last_idx]
        y[t] = gate * value[t - 1] + 0.10 * value[t - 2]

    feat = np.stack([marker, value], axis=1)
    data = _make_windows_from_features(feat, y, window)
    return ToyBundle(
        name="marker",
        n_features=2,
        x_qtp_tr=data[0], x_qtp_va=data[1], x_qtp_te=data[2],
        x_qtp_const_tr=data[3], x_qtp_const_va=data[4], x_qtp_const_te=data[5],
        x_plain_tr=data[6], x_plain_va=data[7], x_plain_te=data[8],
        y_tr=data[9], y_va=data[10], y_te=data[11],
        y_te_raw=data[12], y_mu=data[13], y_std=data[14],
    )


def make_toy_interaction(T: int = 1300, window: int = WINDOW) -> ToyBundle:
    rng = np.random.default_rng(789)
    x1 = np.zeros(T, dtype=np.float32)
    x2 = np.zeros(T, dtype=np.float32)
    x3 = np.zeros(T, dtype=np.float32)
    e1 = rng.normal(0.0, 0.28, size=T).astype(np.float32)
    e2 = rng.normal(0.0, 0.25, size=T).astype(np.float32)
    e3 = rng.normal(0.0, 0.22, size=T).astype(np.float32)
    for t in range(1, T):
        x1[t] = 0.68 * x1[t - 1] + e1[t]
        x2[t] = 0.62 * x2[t - 1] + 0.15 * x1[t - 1] + e2[t]
        x3[t] = 0.65 * x3[t - 1] - 0.10 * x2[t - 1] + e3[t]

    y = np.zeros(T, dtype=np.float32)
    for t in range(window, T):
        y[t] = (
            x1[t - 1] * x2[t - 2]
            + 0.50 * x2[t - 3] * x3[t - 4]
            + 0.20 * np.sin(x3[t - 1])
        )

    feat = np.stack([x1, x2, x3], axis=1)
    data = _make_windows_from_features(feat, y, window)
    return ToyBundle(
        name="interaction",
        n_features=3,
        x_qtp_tr=data[0], x_qtp_va=data[1], x_qtp_te=data[2],
        x_qtp_const_tr=data[3], x_qtp_const_va=data[4], x_qtp_const_te=data[5],
        x_plain_tr=data[6], x_plain_va=data[7], x_plain_te=data[8],
        y_tr=data[9], y_va=data[10], y_te=data[11],
        y_te_raw=data[12], y_mu=data[13], y_std=data[14],
    )


def _disable_snapshot_module(model: QuantumTemporalPredictor):
    for name, p in model.named_parameters():
        if name.startswith("snap_") or name.startswith("bas_"):
            with torch.no_grad():
                p.zero_()
            p.requires_grad_(False)


def _build_variant(model_name: str, window: int, n_features: int, K: int, layers: int):
    cfg = ModelConfig(windows=window, n_features=n_features, layers=layers, K=K)
    if model_name in ("FullQTP", "NoSnapshot", "NoTimeflow"):
        model = QuantumTemporalPredictor(cfg).to(DEVICE)
        if model_name == "NoSnapshot":
            _disable_snapshot_module(model)
        return model
    if model_name == "VQC":
        return build_model("VQC", window, n_features, None)
    raise ValueError(f"Unknown model_name: {model_name}")


def _get_data_for_model(bundle: ToyBundle, model_name: str):
    if model_name == "FullQTP":
        return bundle.x_qtp_tr, bundle.x_qtp_va, bundle.x_qtp_te
    if model_name == "NoSnapshot":
        return bundle.x_qtp_tr, bundle.x_qtp_va, bundle.x_qtp_te
    if model_name == "NoTimeflow":
        return bundle.x_qtp_const_tr, bundle.x_qtp_const_va, bundle.x_qtp_const_te
    if model_name == "VQC":
        return bundle.x_plain_tr, bundle.x_plain_va, bundle.x_plain_te
    raise ValueError(model_name)


def _inverse_target(y_scaled: np.ndarray, mu: float, std: float) -> np.ndarray:
    return y_scaled.reshape(-1) * std + mu


def run_mechanism_experiment():
    os.makedirs(RESULT_ROOT, exist_ok=True)

    toys = [
        make_toy_order(),
        make_toy_marker(),
        make_toy_interaction(),
    ]
    model_names = ["FullQTP", "NoSnapshot", "NoTimeflow", "VQC"]
    K = max(1, round(WINDOW * K_RATIO))

    colors = {
        "FullQTP": "#C97C3B",
        "NoSnapshot": "#9D8F7E",
        "NoTimeflow": "#7E90B8",
        "VQC": "#5F5568",
    }

    all_rows: List[Dict] = []
    aggregated: Dict[Tuple[str, str], Dict] = {}
    representative_predictions: Dict[Tuple[str, str], np.ndarray] = {}

    for toy in toys:
        toy_root = os.path.join(RESULT_ROOT, toy.name)
        os.makedirs(toy_root, exist_ok=True)

        for model_name in model_names:
            seed_metrics = []

            for seed in SEEDS:
                seed_everything(seed)
                model = _build_variant(model_name, WINDOW, toy.n_features, K, QTP_LAYERS)
                lr = LR_QTP if model_name != "VQC" else LR_VQC
                optimizer = torch.optim.Adam(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=lr,
                )

                x_tr, x_va, x_te = _get_data_for_model(toy, model_name)
                save_dir = os.path.join(toy_root, f"{model_name}_seed{seed}")
                trainer = Trainer(
                    model=model,
                    optimizer=optimizer,
                    batch_size=BATCH_SIZE,
                    patience=PATIENCE,
                    save_path=save_dir,
                )
                trainer.train(x_tr, toy.y_tr, x_va, toy.y_va, epochs=EPOCHS)
                _, preds = trainer.evaluate(x_te, toy.y_te)

                yt = toy.y_te.view(-1).numpy()
                yp = preds.view(-1).numpy()
                r2 = r2_score(yt, yp)
                mse = mean_squared_error(yt, yp)
                seed_metrics.append((r2, mse, yp))

                all_rows.append({
                    "toy": toy.name,
                    "model": model_name,
                    "seed": seed,
                    "r2": r2,
                    "mse": mse,
                })

            r2s = [m[0] for m in seed_metrics]
            mses = [m[1] for m in seed_metrics]
            aggregated[(toy.name, model_name)] = {
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0,
            }

            best_idx = int(np.argmax(r2s))
            representative_predictions[(toy.name, model_name)] = seed_metrics[best_idx][2]

    _save_rows(all_rows, os.path.join(RESULT_ROOT, "mechanism_metrics.csv"))
    _save_aggregated(aggregated, os.path.join(RESULT_ROOT, "mechanism_summary.csv"))
    plot_mechanism_summary(toys, model_names, aggregated, representative_predictions, colors)


def _save_rows(rows: List[Dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_aggregated(aggregated: Dict[Tuple[str, str], Dict], path: str):
    rows = []
    for (toy, model), stats in aggregated.items():
        row = {"toy": toy, "model": model}
        row.update(stats)
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_mechanism_summary(
    toys: List[ToyBundle],
    model_names: List[str],
    aggregated: Dict[Tuple[str, str], Dict],
    rep_preds: Dict[Tuple[str, str], np.ndarray],
    colors: Dict[str, str],
):
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5))

    display_names = {
        "order": "Toy 1: Order-sensitive",
        "marker": "Toy 2: Sparse-marker memory",
        "interaction": "Toy 3: Feature interaction",
    }
    highlight = {
        "order": ["FullQTP", "NoTimeflow", "VQC"],
        "marker": ["FullQTP", "NoSnapshot", "VQC"],
        "interaction": ["FullQTP", "NoTimeflow", "VQC"],
    }

    for col, toy in enumerate(toys):
        ax = axes[0, col]
        means = [aggregated[(toy.name, m)]["r2_mean"] for m in model_names]
        errs = [aggregated[(toy.name, m)]["r2_std"] for m in model_names]
        bar_colors = [colors[m] for m in model_names]
        ax.bar(np.arange(len(model_names)), means, yerr=errs, capsize=3, color=bar_colors, alpha=0.9)
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=18)
        ax.set_ylabel(r"$R^2$")
        ax.set_title(display_names[toy.name], fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)

        ax = axes[1, col]
        idx = slice(0, min(140, len(toy.y_te_raw)))
        y_true = toy.y_te_raw[idx]
        ax.plot(y_true, color="#222222", linewidth=1.8, label="Ground truth")
        for model_name in highlight[toy.name]:
            yp = rep_preds[(toy.name, model_name)]
            yp_raw = _inverse_target(yp, toy.y_mu, toy.y_std)[idx]
            ax.plot(yp_raw, linewidth=1.4, color=colors[model_name], label=model_name)
        ax.set_title(f"{display_names[toy.name]}: representative test trace", fontsize=10)
        ax.set_xlabel("Test step")
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Target")
        ax.legend(fontsize=8, frameon=True)

    plt.tight_layout()
    png_path = os.path.join(RESULT_ROOT, "mechanism_summary.png")
    pdf_path = os.path.join(RESULT_ROOT, "mechanism_summary.pdf")
    plt.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_mechanism_experiment()
