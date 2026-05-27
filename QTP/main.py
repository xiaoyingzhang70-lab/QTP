# -*- coding: utf-8 -*-
"""
main.py
=======
实验入口文件，定义所有实验方案：

  planA  QTP 在 mackey 数据集上的消融实验（窗口 / 快照密度 / 层数）
  planB  多模型 × 多数据集横向对比
  planC  噪声鲁棒性实验（depolarizing / bit_flip / phase_flip / amplitude_damp）

所有 print 输出同时写入 log.log。
"""

import logging
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── 本地模块 ──────────────────────────────────────────────────────────────────
from DataLoader import DataConfig, DataProcessor, DATASET_CONFIGS
from Model      import (ModelConfig, build_model,
                        QuantumTemporalPredictor,QuantumTemporalPredictor_last)
from Trainer    import (seed_everything, Trainer,
                        print_metrics, plot_results, SEED, DEVICE)
import setproctitle

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  日志：同时输出到 stdout 和 log.log
# ══════════════════════════════════════════════════════════════════════════════

class _TeeStream:
    """将写入同时转发到两个流。"""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file   = open("log_PlanA_lorenz.log", "a", encoding="utf-8")
sys.stdout  = _TeeStream(sys.__stdout__, _log_file)
sys.stderr  = _TeeStream(sys.__stderr__, _log_file)


# ══════════════════════════════════════════════════════════════════════════════
#  公共超参（可在各 plan 内单独覆盖）
# ══════════════════════════════════════════════════════════════════════════════

WINDOW     = 12
K_RATIO    = 0.33
QTP_LAYERS = 1
EPOCHS     = 50
LR_QTP     = 5e-2
LR_CLASSIC = 1e-3
LR_QUANTUM = 1e-2
BATCH_SIZE = 16
PATIENCE   = 10

# 量子对比模型超参
NUM_QUBITS        = 4
HIDDEN_SIZE       = 4
VQC_LAYERS        = 1
QLSTM_LAYERS      = 1
NUM_QUBITS_HIDDEN = 2


# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _lr_for(name: str) -> float:
    """根据模型名返回推荐学习率。"""
    if name == "QTP":
        return LR_QTP
    elif name in ("VQC", "QLSTM", "QRNN"):
        return LR_QUANTUM
    else:
        return LR_CLASSIC


def _use_pos(name: str) -> bool:
    """只有 QTP 需要位置编码列。"""
    return name == "QTP"


def _model_cfg_for_qtp(window: int, n_features: int,
                        layers: int, k: int) -> ModelConfig:
    return ModelConfig(windows=window, n_features=n_features,
                       layers=layers, K=k)


def run_one(
    model_name:  str,
    dataset:     str,
    window:      int,
    k:           int,
    layers:      int,
    epochs:      int,
    result_root: str,
    seed:        int = SEED,
) -> dict:
    """
    完整训练一个模型并返回指标字典。
    ARIMAX 绕过 Trainer，直接 fit。
    """
    seed_everything(seed)

    # 数据
    data_cfg  = DataConfig(
        dataname  = dataset,
        window_size = window,
        use_pos   = _use_pos(model_name),
    )
    processor = DataProcessor(data_cfg)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = processor.get_dataset()
    nf = processor.n_features

    # 模型
    mcfg  = _model_cfg_for_qtp(window, nf, layers, k) if model_name == "QTP" or model_name == "QTP_last" \
            else None
    model = build_model(model_name, window, nf, mcfg)
    n_p   = sum(p.numel() for p in model.parameters())

    save_dir = os.path.join(
        result_root,
        f"{model_name}_{dataset}_W{window}_K{k}_L{layers}_P{n_p}"
    )

    tr_losses, va_losses = [], []
    t0 = time.time()

    if model_name == "ARIMAX":
        endog, exog = processor.get_raw_train_series()
        model.fit(endog, exog)
        trainer = Trainer(model=model,
                          optimizer=torch.optim.Adam(
                              [model._dummy], lr=1e-4),
                          save_path=save_dir)
        # 不 train，直接 evaluate
        _, preds = trainer.evaluate(x_te, y_te)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=_lr_for(model_name))
        trainer   = Trainer(model=model, optimizer=optimizer,
                            batch_size=BATCH_SIZE, patience=PATIENCE,
                            save_path=save_dir)
        tr_losses, va_losses = trainer.train(
            x_tr, y_tr, x_va, y_va, epochs=epochs)
        _, preds = trainer.evaluate(x_te, y_te)

    elapsed = time.time() - t0
    metrics = print_metrics(model_name, dataset, y_te, preds, n_params=n_p)
    metrics.update({
        "n_params":  n_p,
        "elapsed":   elapsed,
        "tr_losses": tr_losses,
        "va_losses": va_losses,
        "preds":     preds.numpy(),
        "y_te":      y_te.numpy(),
    })

    np.save(os.path.join(save_dir, "preds.npy"), preds.numpy())
    np.save(os.path.join(save_dir, "y_te.npy"),  y_te.numpy())
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  planA  QTP 在 mackey 上的消融实验
# ══════════════════════════════════════════════════════════════════════════════

def planA():
    """
    三个层面的消融：
      A1 窗口大小：W ∈ {4,8,12,16,24}，K=round(W*0.33)，Layer=1
      A2 快照密度：W=12，ratio ∈ {0.25,0.33,0.50,0.75,1.00}，Layer=1
      A3 层数×窗口：W ∈ {4,8,12,16,24}，Layer ∈ {1,2,3,5,10}，热力图
    """
    print("\n" + "█"*60)
    print("  planA：QTP × mackey 消融实验")
    print("█"*60)

    DATASET = "lorenz"
    ROOT    = "./result/planA_lorenz"
    os.makedirs(ROOT, exist_ok=True)

    # ── A1：窗口大小 ──────────────────────────────────────────────────
    print("\n── A1: 窗口大小消融 ──")
    WINDOWS_A1 = [4, 8, 12, 16, 24]
    res_a1 = {}
    for W in WINDOWS_A1:
        K = max(1, round(W * K_RATIO))
        print(f"\n  W={W}  K={K}  L=1")
        res_a1[W] = run_one("QTP", DATASET, W, K, 1, EPOCHS, ROOT)

    # A1 绘图
    fig, ax = plt.subplots(figsize=(8, 4))
    r2s = [res_a1[W]["r2"]  for W in WINDOWS_A1]
    ms  = [res_a1[W]["mse"] for W in WINDOWS_A1]
    ax2 = ax.twinx()
    ax.plot(WINDOWS_A1, r2s,  "o-", color="#378ADD", label="R²",  lw=2)
    ax2.plot(WINDOWS_A1, ms,  "s--", color="#D85A30", label="MSE", lw=2)
    ax.set_xlabel("Window size W"); ax.set_ylabel("R²", color="#378ADD")
    ax2.set_ylabel("MSE", color="#D85A30")
    ax.set_title("A1: Window size ablation — mackey")
    ax.set_xticks(WINDOWS_A1)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{ROOT}/A1_window.png", dpi=150)
    plt.show()

    # ── A2：快照密度 ──────────────────────────────────────────────────
    print("\n── A2: 快照密度消融 ──")
    RATIOS = [0.25, 0.33, 0.50, 0.75, 1.00]
    res_a2 = {}
    for ratio in RATIOS:
        K = max(1, round(WINDOW * ratio))
        print(f"\n  W={WINDOW}  ratio={ratio}  K={K}  L=1")
        res_a2[ratio] = run_one("QTP", DATASET, WINDOW, K, 1, EPOCHS, ROOT)

    fig, ax = plt.subplots(figsize=(8, 4))
    r2s = [res_a2[r]["r2"] for r in RATIOS]
    ax.plot(RATIOS, r2s, "o-", color="#378ADD", lw=2)
    ax.set_xlabel("Snapshot ratio K/W"); ax.set_ylabel("R²")
    ax.set_title("A2: Snapshot density ablation — mackey")
    ax.set_xticks(RATIOS); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{ROOT}/A2_snapshot.png", dpi=150)
    plt.show()

    # ── A3：层数 × 窗口 热力图 ───────────────────────────────────────
    print("\n── A3: 层数×窗口 联合消融 ──")
    WINDOWS_A3 = [4,  8,  12, 16,24] #4,  8,  12, 16,24
    LAYERS_A3  = [1,2,3,5,10]
    SKIP       = {(16,5),(16,10),(24,3),(24,5),(24,10)}
    r2_mat     = np.full((len(LAYERS_A3), len(WINDOWS_A3)), np.nan)

    for wi, W in enumerate(WINDOWS_A3):
        for li, L in enumerate(LAYERS_A3):
            if (W, L) in SKIP:
                print(f"  Skip W={W} L={L}")
                continue
            K = max(1, round(W * K_RATIO))
            print(f"\n  W={W}  K={K}  L={L}")
            m = run_one("QTP", DATASET, W, K, L, EPOCHS, ROOT)
            r2_mat[li, wi] = m["r2"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(r2_mat, ax=ax, mask=np.isnan(r2_mat),
                annot=True, fmt=".4f",
                xticklabels=WINDOWS_A3,
                yticklabels=[f"L={L}" for L in LAYERS_A3],
                cmap="YlOrRd_r", linewidths=0.5)
    ax.set_xlabel("Window size W"); ax.set_ylabel("Layers L")
    ax.set_title("A3: Layer × Window heatmap (R²) — mackey")
    plt.tight_layout()
    plt.savefig(f"{ROOT}/A3_heatmap.png", dpi=150)
    plt.show()

    np.save(f"{ROOT}/res_a1.npy", res_a1)
    np.save(f"{ROOT}/res_a2.npy", res_a2)
    np.save(f"{ROOT}/r2_mat_a3.npy", r2_mat)
    print("\n  planA 完成 → ./result/planA/")


# ══════════════════════════════════════════════════════════════════════════════
#  planB  多模型 × 多数据集横向对比
# ══════════════════════════════════════════════════════════════════════════════

def planB():
    """
    对比模型：QTP / QLSTM / QRNN / VQC / LSTM / Transformer / MLP / ARIMAX
    数据集：lorenz / henon / mackey / synthetic
            QLSTM / QRNN / VQC 不在 ETTh1 上实验（计算量过大且量子模型在真实数据表现差）
    """
    print("\n" + "█"*60)
    print("  planB：多模型 × 多数据集横向对比")
    print("█"*60)

    ROOT = "./result/planB"
    os.makedirs(ROOT, exist_ok=True)

    K = max(1, round(WINDOW * K_RATIO))

    # 模型分组
    # ALL_MODELS     = ["QTP", "QLSTM", "QRNN", "VQC",
    #                   "LSTM", "Transformer", "MLP", "ARIMAX"]
    
    ALL_MODELS     = ["LSTM"]
    
    # QLSTM/QRNN/VQC 不跑 ETTh1
    NO_ETTh1_MODELS = {"QLSTM", "QRNN", "VQC"}

    DATASETS = ["lorenz","henon","synthetic"]#"mackey", "lorenz","henon" , "ETTh1"

    all_results = {}   # {dataset: {model: metrics}}

    for dataset in DATASETS:
        print(f"\n{'='*55}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*55}")
        all_results[dataset] = {}

        for model_name in ALL_MODELS:
            if dataset == "ETTh1" and model_name in NO_ETTh1_MODELS:
                print(f"  Skip {model_name} on ETTh1")
                continue

            print(f"\n  --- {model_name} ---")
            try:
                m = run_one(model_name, dataset, WINDOW, K,
                            QTP_LAYERS, EPOCHS, ROOT)
                all_results[dataset][model_name] = m
            except Exception as e:                
                print(f"  [ERROR] {model_name} on {dataset}: {e}")
                all_results[dataset][model_name] = None

    # ── 汇总打印 ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  planB 汇总")
    print(f"{'═'*70}")
    header = f"  {'Model':<14}" + "".join(
        f"{'R²_'+d:>14}" for d in DATASETS)
    print(header)
    print("  " + "-" * (14 + 14 * len(DATASETS)))
    for model_name in ALL_MODELS:
        row = f"  {model_name:<14}"
        for dataset in DATASETS:
            m = all_results[dataset].get(model_name)
            if m is None:
                row += f"{'—':>14}"
            else:
                row += f"{m['r2']:>14.6f}"
        print(row)

    # ── 绘图：各数据集 R² 柱状图 ─────────────────────────────────────
    colors = ["#378ADD","#1D9E75","#BA7517","#D85A30",
              "#993556","#7F77DD","#AAAAAA","#444441"]
    fig, axes = plt.subplots(1, len(DATASETS),
                             figsize=(4.5 * len(DATASETS), 5))
    if len(DATASETS) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASETS):
        names, r2s, clrs = [], [], []
        for i, name in enumerate(ALL_MODELS):
            m = all_results[dataset].get(name)
            if m is not None:
                names.append(name)
                r2s.append(max(m["r2"], -0.5))
                clrs.append(colors[i % len(colors)])
        ax.bar(names, r2s, color=clrs, edgecolor="black", linewidth=0.4)
        ax.set_title(f"{dataset}", fontsize=9)
        ax.set_ylabel("R²")
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xticklabels(names, rotation=45, fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("planB — R² comparison across datasets", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{ROOT}/planB_comparison.png", dpi=150)
    plt.show()

    np.save(f"{ROOT}/all_results.npy", all_results)
    print("\n  planB 完成 → ./result/planB/")

# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    setproctitle.setproctitle("ZXY")
    seed_everything(SEED)
    planA()
    planB()
