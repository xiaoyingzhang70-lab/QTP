# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:50:27 2026

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Trainer.py
==========
训练与评估模块，包含：
  - seed_everything    全局随机种子
  - Trainer            通用训练器
  - print_metrics      指标打印与返回
  - plot_results       训练曲线 + 预测对比图
"""

import os
import random
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")
SEED   = 42


# ══════════════════════════════════════════════════════════════════════════════
#  全局随机种子
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = SEED) -> None:
    """设置所有随机源，确保实验可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
#  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    通用训练器，适配所有实现 forward(x) → scalar 的模型。

    Parameters
    ----------
    model      : nn.Module
    optimizer  : torch 优化器
    criterion  : 损失函数，默认 MSELoss
    batch_size : 每批样本数
    patience   : early stopping 等待轮数
    save_path  : 模型与损失曲线保存目录
    """

    def __init__(
        self,
        model:      nn.Module,
        optimizer:  torch.optim.Optimizer,
        criterion:  nn.Module = nn.MSELoss(),
        batch_size: int       = 16,
        patience:   int       = 10,
        save_path:  str       = "./result/tmp",
    ):
        self.model      = model
        self.optimizer  = optimizer
        self.criterion  = criterion
        self.batch_size = batch_size
        self.patience   = patience
        self.save_path  = save_path
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience)
        os.makedirs(save_path, exist_ok=True)

    def train(
        self,
        x_tr: torch.Tensor, y_tr: torch.Tensor,
        x_va: torch.Tensor, y_va: torch.Tensor,
        epochs: int = 50,
    ) -> Tuple[List[float], List[float]]:
        """
        训练模型，支持 early stopping 和学习率衰减。

        Returns
        -------
        tr_losses, va_losses : per-epoch MSE 列表
        """
        # 训练开始前重置随机种子，确保 randperm 顺序可复现
        seed_everything(SEED)

        best_va   = float("inf")
        wait      = 0
        tr_losses, va_losses = [], []

        for epoch in range(1, epochs + 1):
            tr_loss    = self._train_epoch(x_tr, y_tr)
            va_loss, _ = self.evaluate(x_va, y_va)
            self.scheduler.step(va_loss)
            tr_losses.append(tr_loss)
            va_losses.append(va_loss)
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train:{tr_loss:.6f}  valid:{va_loss:.6f}")

            if va_loss < best_va:
                best_va = va_loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_path, "best_model.pt"))
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        self.model.load_state_dict(
            torch.load(os.path.join(self.save_path, "best_model.pt"),
                       weights_only=True))
        np.save(os.path.join(self.save_path, "tr_losses.npy"),
                np.array(tr_losses))
        np.save(os.path.join(self.save_path, "va_losses.npy"),
                np.array(va_losses))
        print(f"  Best valid MSE: {best_va:.6f}")
        return tr_losses, va_losses

    @torch.no_grad()
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """返回 (mse, predictions_tensor)"""
        self.model.eval()
        preds = torch.stack(
            [self.model(x[i].to(DEVICE)) for i in range(len(x))])
        loss = self.criterion(preds.view(-1), y.view(-1).to(DEVICE))
        return loss.item(), preds.cpu()

    def _train_epoch(self, x_tr, y_tr) -> float:
        self.model.train()
        n     = x_tr.shape[0]
        idx   = torch.randperm(n)
        total = 0.0
        for start in range(0, n, self.batch_size):
            bi    = idx[start : start + self.batch_size]
            x_b   = x_tr[bi].to(DEVICE)
            y_b   = y_tr[bi].to(DEVICE)
            self.optimizer.zero_grad()
            preds = torch.stack(
                [self.model(x_b[i]) for i in range(len(x_b))])
            loss  = self.criterion(preds.view(-1), y_b.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item() * len(x_b)
        return total / n


# ══════════════════════════════════════════════════════════════════════════════
#  评估工具
# ══════════════════════════════════════════════════════════════════════════════

def print_metrics(
    name:    str,
    dataset: str,
    y_true:  torch.Tensor,
    y_pred:  torch.Tensor,
    n_params: int = 0,
) -> dict:
    """打印并返回指标字典。"""
    yt  = y_true.view(-1).detach().numpy()
    yp  = y_pred.view(-1).detach().numpy()
    mae = mean_absolute_error(yt, yp)
    mse = mean_squared_error(yt, yp)
    r2  = r2_score(yt, yp)
    suffix = f"  Params:{n_params}" if n_params else ""
    print(f"  [{name:<12}] [{dataset}]  "
          f"MAE:{mae:.6f}  MSE:{mse:.6f}  "
          f"RMSE:{mse**0.5:.6f}  R²:{r2:.6f}{suffix}")
    return {"mae": mae, "mse": mse, "rmse": mse**0.5, "r2": r2}


def plot_results(
    y_true:       torch.Tensor,
    predictions:  Dict[str, torch.Tensor],
    train_losses: Dict[str, List[float]],
    valid_losses: Dict[str, List[float]],
    dataset_name: str = "",
    save_path:    str = "results.png",
) -> None:
    """绘制训练曲线 + 测试集预测对比图。"""
    colors = ["#378ADD", "#D85A30", "#1D9E75",
              "#BA7517", "#993556", "#7F77DD", "#888888", "#333333"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sfx = f" — {dataset_name}" if dataset_name else ""

    ax = axes[0]
    for i, name in enumerate(train_losses):
        c = colors[i % len(colors)]
        ax.plot(train_losses[name], color=c, linewidth=1.5,
                label=f"{name} train")
        ax.plot(valid_losses[name], color=c, linewidth=1.2,
                linestyle="--", alpha=0.7, label=f"{name} valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalised space)")
    ax.set_title(f"Training curve{sfx}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(y_true.numpy(), color="#444441", linewidth=1.8,
            alpha=0.9, label="Ground truth")
    for i, (name, preds) in enumerate(predictions.items()):
        c = colors[i % len(colors)]
        ax.plot(preds.numpy(), color=c, linewidth=1.4,
                linestyle="--", alpha=0.85, label=name)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Target (normalised space)")
    ax.set_title(f"Test set prediction{sfx}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved → {save_path}")
    plt.show()