# -*- coding: utf-8 -*-
"""
Dataloader.py
=============
数据处理模块，包含：
  - DataConfig   数据配置
  - DataProcessor 数据处理类（z-score + arctan + 位置编码 + 滑窗）

DATA_PATH 格式：
  {数据集名: [文件路径, [输入特征列], [目标列]]}

use_target_as_feature 接口：
  True  → 目标列也作为特征输入（适用于多变量场景）
  False → 目标列只作为标签，不进入特征矩阵
  
  Mackey 特殊处理：
    mackey 的特征列和目标列是同一列，强制 use_target_as_feature=True
    且只保留 1 个特征列（去掉重复），滑窗时 x[t:t+W] 预测 y[t+W]
"""

# ══════════════════════════════════════════════════════════════════════════════
"""
数据处理流程：
  CSV → 列校验 → z-score（训练集统计量）
  → arctan 压缩（仅 use_pos=True，即 QTP 模型）
  → 位置编码拼接（仅 use_pos=True）
  → 滑动窗口 → Tensor
"""
 
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
 
import numpy as np
import pandas as pd
import torch
 
# warnings.filterwarnings("ignore")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  数据集配置
# ══════════════════════════════════════════════════════════════════════════════
 
DATASET_CONFIGS: Dict[str, dict] = {
    "mackey": {
        "path":           "./Dataset/mackey/mackey_tau30_1000.csv",
        "endog_col":      "mackey_glass",
        "exog_cols":      ["mackey_glass"],
        "target_as_exog": True,
    },
    "lorenz": {
        "path":           "./Dataset/lorenz/lorenz_1000.csv",
        "endog_col":      "z_coords",
        "exog_cols":      ["x_coords", "y_coords"],
        "target_as_exog": False,
    },
    "henon": {
        "path":           "./Dataset/henon/henon_1000.csv",
        "endog_col":      "y_coords",
        "exog_cols":      ["x_coords"],
        "target_as_exog": False,
    },
    "heisenberg": {
        "path":           "./Dataset/heisenberg/heisenberg_N8_J1p0_Sz_1000.csv",
        "endog_col":      "Sz_3",
        "exog_cols":      ["Sz_0", "Sz_1", "Sz_2"],
        "target_as_exog": False,
    },
    "synthetic": {
        "path":           "./Dataset/Synthetic/synthetic.csv",
        "endog_col":      "OT",
        "exog_cols":      ["feature_0", "feature_1", "feature_2"],
        "target_as_exog": False,
    },
}
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  DataConfig
# ══════════════════════════════════════════════════════════════════════════════
 
@dataclass
class DataConfig:
    """
    数据加载配置
 
    Parameters
    ----------
    dataname    : DATASET_CONFIGS 中的 key
    window_size : 滑动窗口长度
    split       : (train, valid, test) 比例，三者之和为 1
    use_pos     : 是否拼入位置编码列（QTP 需要 True，其余模型 False）
    """
    dataname:    str                      = "mackey"
    window_size: int                      = 12
    split:       Tuple[float,float,float] = (0.6, 0.2, 0.2)
    use_pos:     bool                     = True
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  DataProcessor
# ══════════════════════════════════════════════════════════════════════════════
 
class DataProcessor:
    """
    统一数据处理类。
 
    特征列构建规则（由 DATASET_CONFIGS 中的 target_as_exog 控制）：
      target_as_exog=True  → feat_cols = exog_cols + [endog_col]（去重保序）
      target_as_exog=False → feat_cols = exog_cols
 
    Attributes
    ----------
    n_features   : 实际特征列数（不含 pos 列），供 ModelConfig 使用
    scaler_mean  : 训练集均值（pd.Series），用于反归一化
    scaler_std   : 训练集标准差（pd.Series），用于反归一化
    """
 
    def __init__(self, cfg: DataConfig):
        if cfg.dataname not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataname '{cfg.dataname}'. "
                f"Available: {list(DATASET_CONFIGS.keys())}"
            )
        if abs(sum(cfg.split) - 1.0) > 1e-6:
            raise ValueError(
                f"split 之和须为 1，当前: {cfg.split}")
 
        self.cfg = cfg
        ds_cfg   = DATASET_CONFIGS[cfg.dataname]
 
        path = ds_cfg["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")
 
        # 读取并去掉 date 列
        df        = pd.read_csv(path)
        date_cols = [c for c in df.columns if c.lower() == "date"]
        self._data_df = df.drop(columns=date_cols)
 
        self._endog_col = ds_cfg["endog_col"]
        _exog_cols      = ds_cfg["exog_cols"]
        _taf            = ds_cfg["target_as_exog"]
 
        # ── 构建特征列列表 ─────────────────────────────────────────────
        if _taf:
            # 目标列加入特征，顺序去重（保留 exog_cols 顺序，末尾补充 endog）
            _combined = list(_exog_cols) + [self._endog_col]
            seen, self._feat_cols = set(), []
            for c in _combined:
                if c not in seen:
                    seen.add(c)
                    self._feat_cols.append(c)
        else:
            self._feat_cols = list(_exog_cols)
 
        # ── 列校验 ────────────────────────────────────────────────────
        all_cols = list(self._data_df.columns)
        for c in self._feat_cols + [self._endog_col]:
            if c not in all_cols:
                raise ValueError(
                    f"列 '{c}' 不在数据集 '{cfg.dataname}' 中。"
                    f"可用列: {all_cols}"
                )
 
        self.n_features: int                  = len(self._feat_cols)
        self.scaler_mean: Optional[pd.Series] = None
        self.scaler_std:  Optional[pd.Series] = None
 
    # ── 公共接口 ──────────────────────────────────────────────────────
 
    def get_dataset(self) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Returns
        -------
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te)
 
        x shape:
          use_pos=True  → (N, W, 1+n_features)   第 0 列为位置编码
          use_pos=False → (N, W, n_features)
        y shape: (N, 1)
        """
        cfg = self.cfg
        df  = self._data_df.copy()
        n   = len(df)
        i1  = math.ceil(n * cfg.split[0])
        i2  = math.ceil(n * (cfg.split[0] + cfg.split[1]))
 
        # z-score：仅用训练集统计量
        train_df         = df.iloc[:i1]
        self.scaler_mean = train_df.mean()
        self.scaler_std  = train_df.std()
        df_norm          = self._z_normalise(df)
 
        feat_arr   = df_norm[self._feat_cols].values.astype(np.float32)
        target_arr = df_norm[self._endog_col].values.astype(np.float32)
 
        # arctan 压缩（仅 QTP，将特征值压缩到量子旋转角范围 (-π, π)）
        if cfg.use_pos:
            feat_arr = (
                df_norm[self._feat_cols]
                .apply(lambda col: 2.0 * np.arctan(col).astype(np.float32))
                .values.astype(np.float32)
            )
 
        def _make(s: int, e: int):
            x, y = self._make_windows(
                feat_arr[s:e], target_arr[s:e],
                cfg.window_size, cfg.use_pos,
            )
            return torch.tensor(x), torch.tensor(y)
 
        return _make(0, i1), _make(i1, i2), _make(i2, n)
 
    def get_raw_train_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 z-score 归一化后的完整训练集原始序列（无窗口切分）。
        供统计 ARIMAX 的 fit 接口使用，不做 arctan 压缩。
 
        Returns
        -------
        endog : (T,)     目标变量训练集序列
        exog  : (T, F)   特征变量训练集序列
        """
        df  = self._data_df.copy()
        n   = len(df)
        i1  = math.ceil(n * self.cfg.split[0])
 
        if self.scaler_mean is None:
            train_df         = df.iloc[:i1]
            self.scaler_mean = train_df.mean()
            self.scaler_std  = train_df.std()
 
        df_norm = self._z_normalise(df)
 
        endog = df_norm[self._endog_col].values[:i1].astype(np.float32)
        exog  = df_norm[self._feat_cols].values[:i1].astype(np.float32)
        return endog, exog
 
    def inverse_transform_target(self, y: torch.Tensor) -> torch.Tensor:
        """
        将预测值还原到原始量纲。
        use_pos=True  → 先反 arctan，再反 z-score
        use_pos=False → 只反 z-score
        """
        mu = float(self.scaler_mean[self._endog_col])
        sd = float(self.scaler_std[self._endog_col]) + 1e-8
        if self.cfg.use_pos:
            y_norm = torch.tan(y / 2.0)
        else:
            y_norm = y
        return y_norm * sd + mu
 
    # ── 私有方法 ──────────────────────────────────────────────────────
 
    def _z_normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            mu = self.scaler_mean[col]
            sd = self.scaler_std[col]
            df[col] = (df[col] - mu) / (sd + 1e-8)
        return df
 
    @staticmethod
    def _build_pos(window: int) -> np.ndarray:
        """位置编码：linspace(-0.9π, 0.9π)，shape (W, 1)"""
        return np.linspace(
            -0.9 * np.pi, 0.9 * np.pi, window, dtype=np.float32
        ).reshape(-1, 1)
 
    @staticmethod
    def _make_windows(
        feat:    np.ndarray,
        target:  np.ndarray,
        window:  int,
        use_pos: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        滑动窗口。
        x[i] shape: (W, 1+n_feat) if use_pos else (W, n_feat)
        y[i]      : scalar = target[i+W]
        """
        T   = feat.shape[0]
        pos = DataProcessor._build_pos(window) if use_pos else None
        xs, ys = [], []
        for i in range(T - window):
            win = feat[i : i + window].copy()
            if use_pos:
                win = np.concatenate([pos, win], axis=1)
            xs.append(win)
            ys.append(target[i + window])
        return (
            np.array(xs, dtype=np.float32),
            np.array(ys, dtype=np.float32).reshape(-1, 1),
        )
 