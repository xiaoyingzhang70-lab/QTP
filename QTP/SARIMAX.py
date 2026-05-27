# -*- coding: utf-8 -*-
"""
arimax_experiment.py
====================
独立的统计 ARIMAX 实验文件。

核心设计：
  - 使用 statsmodels SARIMAX，最大似然估计（MLE），非梯度下降
  - 评估方式：apply().fittedvalues
    即：条件均值预测 E[y_t | y_{t-1},...,y_1, X_t]
    等价于 1-step-ahead 预测，是时序预测的规范评估方式
  - 数据预处理：z-score（与其他模型完全一致），不做 arctan
  - 不使用滑动窗口，直接在完整序列上拟合/预测

运行：
  python arimax_experiment.py
结果保存到 ./result/arimax/
"""

import math
import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error,
                              mean_squared_error, r2_score)
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  数据集配置（与 Dataloader.py 保持一致）
# ══════════════════════════════════════════════════════════════════════════════

DATASET_CONFIGS = {
    "mackey": {
        "path":        "./Dataset/mackey/mackey_tau30_1000.csv",
        "endog_col":   "mackey_glass",     # 目标变量
        "exog_cols":   ["mackey_glass"],   # mackey 自回归：exog = endog
        "target_as_exog": True,            # 目标列同时作为外生变量
    },
    "lorenz": {
        "path":        "./Dataset/lorenz/lorenz_1000.csv",
        "endog_col":   "z_coords",
        "exog_cols":   ["x_coords", "y_coords"],
        "target_as_exog": False,
    },
    "henon": {
        "path":        "./Dataset/henon/henon_1000.csv",
        "endog_col":   "y_coords",
        "exog_cols":   ["x_coords"],
        "target_as_exog": False,
    },
    "heisenberg": {
        "path":        "./Dataset/heisenberg/heisenberg_N8_J1p0_Sz_1000.csv",
        "endog_col":   "Sz_3",
        "exog_cols":   ["Sz_0","Sz_1", "Sz_2"],
        "target_as_exog": False,
    },
    "synthetic": {
        "path":        "./Dataset/Synthetic/synthetic.csv",
        "endog_col":   "OT",
        "exog_cols":   ["feature_0", "feature_1", "feature_2"],
        "target_as_exog": False,
    },

}

# 训练/验证/测试 切分比例
SPLIT = (0.6, 0.2, 0.2)

# ARIMAX 阶数（对所有数据集统一，简化比较）
P, D, Q = 2, 0, 1


# ══════════════════════════════════════════════════════════════════════════════
#  数据加载与预处理
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(cfg: dict) -> Tuple[
    np.ndarray, np.ndarray,   # endog_train, exog_train
    np.ndarray, np.ndarray,   # endog_val,   exog_val
    np.ndarray, np.ndarray,   # endog_test,  exog_test
]:
    """
    加载数据并做 z-score 归一化（用训练集统计量）。

    返回 endog（目标列）和 exog（特征列）的训练/验证/测试序列。
    不做 arctan 压缩（ARIMAX 是线性模型，不需要限制输入范围）。
    不做滑动窗口（ARIMAX 直接在完整序列上操作）。
    """
    path      = cfg["path"]
    endog_col = cfg["endog_col"]
    exog_cols = cfg["exog_cols"]

    df       = pd.read_csv(path)
    date_col = [c for c in df.columns if c.lower() == "date"]
    df       = df.drop(columns=date_col)

    n  = len(df)
    i1 = math.ceil(n * SPLIT[0])
    i2 = math.ceil(n * (SPLIT[0] + SPLIT[1]))

    # z-score：只用训练集统计量
    train_df = df.iloc[:i1]
    mean_    = train_df.mean()
    std_     = train_df.std()
    df_norm  = (df - mean_) / (std_ + 1e-8)

    endog = df_norm[endog_col].values.astype(float)

    # 外生变量处理：mackey 的 exog 与 endog 是同一列，直接用即可
    exog = df_norm[exog_cols].values.astype(float)

    return (
        endog[:i1],  exog[:i1],    # train
        endog[i1:i2], exog[i1:i2], # val（不用于拟合，但记录切分位置）
        endog[i2:],  exog[i2:],    # test
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ARIMAX 拟合与评估
# ══════════════════════════════════════════════════════════════════════════════

def run_arimax(dataset_name: str) -> dict:
    """
    在单个数据集上运行 ARIMAX，返回测试集指标。

    评估策略：
      1. 在训练集上 MLE 拟合 ARIMAX(P,D,Q) + exog
      2. 用 apply(endog=test, exog=test_exog) 在测试集上续跑
         fittedvalues = 条件均值预测，等价于 1-step-ahead 预测
      3. 计算 MAE / MSE / RMSE / R²
    """
    print(f"\n{'─'*50}")
    print(f"  [ARIMAX] Dataset: {dataset_name}")
    print(f"{'─'*50}")

    cfg = DATASET_CONFIGS[dataset_name]

    (endog_tr, exog_tr,
     endog_va, exog_va,
     endog_te, exog_te) = load_and_preprocess(cfg)

    print(f"  Train: {len(endog_tr)}  Val: {len(endog_va)}  "
          f"Test: {len(endog_te)}")
    print(f"  endog: {cfg['endog_col']}  "
          f"exog: {cfg['exog_cols']}  "
          f"n_exog: {exog_tr.shape[1]}")

    # ── 拟合 ──────────────────────────────────────────────────────────
    model = SARIMAX(
        endog_tr,
        exog                 = exog_tr,
        order                = (P, D, Q),
        trend                = 'n',
        enforce_stationarity  = False,
        enforce_invertibility = False,
    )
    result = model.fit(disp=False, maxiter=500)

    n_params = len(result.params)
    print(f"  拟合完成  实际参数: {n_params}  "
          f"AIC: {result.aic:.4f}  BIC: {result.bic:.4f}")
    print(f"  参数估计: {dict(zip(result.param_names, result.params.round(4)))}")

    # ── 测试集预测 ─────────────────────────────────────────────────────
    # apply：将已拟合参数应用到测试集序列上
    # fittedvalues[t] = E[y_t | y_{t-1},...,y_1, x_t]
    # 这是规范的 1-step-ahead 条件均值预测
    result_test = result.apply(
        endog  = endog_te,
        exog   = exog_te,
        refit  = False,           # 固定参数，只更新状态
    )
    preds = np.asarray(result_test.fittedvalues)

    # ── 指标计算 ───────────────────────────────────────────────────────
    mae  = mean_absolute_error(endog_te, preds)
    mse  = mean_squared_error(endog_te, preds)
    r2   = r2_score(endog_te, preds)
    rmse = mse ** 0.5

    print(f"  Test  MAE:{mae:.6f}  MSE:{mse:.6f}  "
          f"RMSE:{rmse:.6f}  R²:{r2:.6f}")

    return {
        "mae":      mae,
        "mse":      mse,
        "rmse":     rmse,
        "r2":       r2,
        "n_params": n_params,
        "aic":      result.aic,
        "bic":      result.bic,
        "preds":    preds,
        "y_te":     endog_te,
        "result":   result,       # 保留完整结果对象
    }


# ══════════════════════════════════════════════════════════════════════════════
#  绘图
# ══════════════════════════════════════════════════════════════════════════════

def plot_arimax_results(all_metrics: Dict[str, dict],
                        save_dir: str) -> None:
    """绘制各数据集的预测曲线和指标汇总图。"""
    datasets = list(all_metrics.keys())
    n        = len(datasets)

    # 每个数据集一张预测图
    for dataset, m in all_metrics.items():
        y_true = m["y_te"]
        y_pred = m["preds"]
        n_plot = min(200, len(y_true))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # 左：预测 vs 真实
        ax = axes[0]
        ax.plot(y_true[:n_plot], color="#444441",
                lw=1.8, alpha=0.9, label="Ground truth")
        ax.plot(y_pred[:n_plot], color="#D85A30",
                lw=1.4, linestyle="--", alpha=0.85,
                label=f"ARIMAX (R²={m['r2']:.4f})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value (z-score)")
        ax.set_title(f"ARIMAX Prediction — {dataset}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 右：残差分布
        ax = axes[1]
        residuals = y_true - y_pred
        ax.hist(residuals, bins=40, color="#378ADD", alpha=0.7,
                edgecolor="black", linewidth=0.4)
        ax.axvline(0, color="red", lw=1.5, linestyle="--")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.set_title(f"Residual distribution — {dataset}"
                     f"\nmean={residuals.mean():.4f}  "
                     f"std={residuals.std():.4f}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"arimax_{dataset}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Figure saved → arimax_{dataset}.png")

    # 指标汇总图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # R² 柱状图
    ax     = axes[0]
    r2s    = [all_metrics[d]["r2"] for d in datasets]
    colors = ["#378ADD" if r > 0.9 else
              "#FFA500" if r > 0.5 else
              "#D85A30" for r in r2s]
    ax.bar(datasets, r2s, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_ylabel("R²")
    ax.set_title("ARIMAX R² across datasets")
    for i, (d, r) in enumerate(zip(datasets, r2s)):
        ax.text(i, r + 0.01, f"{r:.4f}", ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 参数量 + AIC
    ax  = axes[1]
    nps = [all_metrics[d]["n_params"] for d in datasets]
    ax.bar(datasets, nps, color="#1D9E75", edgecolor="black", lw=0.5)
    ax.set_ylabel("Number of parameters")
    ax.set_title("ARIMAX parameter count")
    for i, (d, p) in enumerate(zip(datasets, nps)):
        ax.text(i, p + 0.1, str(p), ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "arimax_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → arimax_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    save_dir = "./result/arimax"
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("  ARIMAX 实验（statsmodels SARIMAX，MLE 估计）")
    print(f"  阶数：ARIMA({P},{D},{Q}) + 外生变量")
    print(f"  评估：apply().fittedvalues（1-step-ahead 条件均值预测）")
    print("=" * 60)

    all_metrics = {}
    for dataset_name in DATASET_CONFIGS:
        try:
            m = run_arimax(dataset_name)
            all_metrics[dataset_name] = m
            np.save(os.path.join(save_dir, f"{dataset_name}_preds.npy"),
                    m["preds"])
            np.save(os.path.join(save_dir, f"{dataset_name}_y_te.npy"),
                    m["y_te"])
        except Exception as e:
            print(f"  [ERROR] {dataset_name}: {e}")
            all_metrics[dataset_name] = None

    # ── 汇总打印 ──────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  ARIMAX 实验汇总")
    print(f"{'═'*65}")
    print(f"  {'Dataset':<15} {'Params':>8} {'MAE':>10} "
          f"{'MSE':>10} {'RMSE':>10} {'R²':>10} {'AIC':>10}")
    print(f"  {'─'*65}")
    for d, m in all_metrics.items():
        if m is None:
            print(f"  {d:<15}  FAILED")
            continue
        print(f"  {d:<15} {m['n_params']:>8} {m['mae']:>10.6f} "
              f"{m['mse']:>10.6f} {m['rmse']:>10.6f} "
              f"{m['r2']:>10.6f} {m['aic']:>10.2f}")

    # ── 与论文中其他模型对比用的数据格式 ─────────────────────────────
    print(f"\n  论文对比格式（直接填入 planB MODEL_RESULTS）：")
    print(f"  {'─'*50}")
    for d, m in all_metrics.items():
        if m is None:
            continue
        print(f'  "{d}": {{'
              f'"ARIMAX": {{"r2":{m["r2"]:.6f}, '
              f'"mse":{m["mse"]:.6f}, '
              f'"mae":{m["mae"]:.6f}, '
              f'"n_params":{m["n_params"]}}}')

    # ── 保存汇总 & 绘图 ───────────────────────────────────────────────
    summary = {d: {k: v for k, v in m.items()
                   if k not in ("preds", "y_te", "result")}
               for d, m in all_metrics.items() if m is not None}
    np.save(os.path.join(save_dir, "summary.npy"), summary)

    valid_metrics = {d: m for d, m in all_metrics.items()
                     if m is not None}
    if valid_metrics:
        plot_arimax_results(valid_metrics, save_dir)

    print(f"\n  所有结果已保存到 {save_dir}/")


if __name__ == "__main__":
    main()