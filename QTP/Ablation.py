# -*- coding: utf-8 -*-
"""
Ablation.py
===========
QTP 消融实验：4 个并列变体

变体定义：
  V1  Full QTP          完整模型：tq（时间流阀门）+ bq（宏观快照）
  V2  QTP w/o Snapshot  切除 bq：保留 tq，删除所有 bq 相关门
                         → 有时间感知，无宏观记忆
  V3  QTP w/o Time-flow 切除 tq：保留 bq 快照，用 RY 直接编码数据
                         → 普通 VQC + 快照记忆，无时间阀门
  V4  Vanilla VQC       基线：tq 和 bq 全切，最简 VQC

电路比较（以 W=12, nf=1, L=1, K=4 为例）：

  V1 Full QTP
    编码段: RY(pos) on tq → CRY(decay*x, tq→fq) → CRZ(θ, tq→fq)
            每K步: CRY(snap_cry, fq→bq) + CRZ(snap_crz, fq→bq)
    网络段: RY(amp) on fq → brick-wall CRY → CRY/CRZ(bq→fq) → RZ on fq
    量子比特: tq + bq + fq  共 2+nf 个

  V2 w/o Snapshot（切 bq）
    编码段: RY(pos) on tq → CRY(decay*x, tq→fq) → CRZ(θ, tq→fq)
            【删除】快照门
    网络段: RY(amp) on fq → brick-wall CRY → 【删除】bq 回注 → RZ on fq
    量子比特: tq + fq      共 1+nf 个

  V3 w/o Time-flow（切 tq）
    编码段: RY(x[t,f]) on fq[f]  直接角度编码，无位置调制
            每K步: CRY(snap_cry, fq→bq) + CRZ(snap_crz, fq→bq)
    网络段: RY(amp) on fq → brick-wall CRY → CRY/CRZ(bq→fq) → RZ on fq
    量子比特: bq + fq      共 1+nf 个
    注：x 输入不含 pos 列（use_pos=False）

  V4 Vanilla VQC（切 tq + bq）
    编码段: RY(x[t,f]) on fq[f]  直接角度编码
            【删除】快照门
    网络段: RY(amp) on fq → brick-wall CRY → 【删除】bq 回注 → RZ on fq
    量子比特: fq 仅        共 nf 个
    注：x 输入不含 pos 列（use_pos=False）

运行：
  python Ablation.py
"""
# from matplotlib.colors import LinearSegmentedColormap
# cmap_custom = LinearSegmentedColormap.from_list(
# "qtp", ["#F3EBE1", "#C87A3E", "#5C3A21"]

import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from DataLoader import DataConfig, DataProcessor, DATASET_CONFIGS
from Trainer    import seed_everything, Trainer, print_metrics, SEED, DEVICE

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  ModelConfig（复用）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    windows:    int = 12
    n_features: int = 1
    layers:     int = 1
    K:          int = 4


# ══════════════════════════════════════════════════════════════════════════════
#  V1：Full QTP  —  完整模型（直接引用 Model.py 的实现）
# ══════════════════════════════════════════════════════════════════════════════

class FullQTP(nn.Module):
    """
    V1：完整 QTP
    tq（时间流阀门）+ bq（宏观快照）均保留。
    与 Model.py 中 QuantumTemporalPredictor 完全一致，
    此处独立复制以保证消融实验的自包含性。

    输入：x (W, 1+nf)  第 0 列为位置编码 pos
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.tq     = "position"
        self.bq     = "baseline"
        self.fq     = [f"feat{i}" for i in range(cfg.n_features)]
        self._wires = [self.tq, self.bq] + self.fq
        self.dev    = qml.device("default.qubit", wires=self._wires)
        self._register_params()
        self.q_net  = self._build_circuit()

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [V1 FullQTP]      qubits:{len(self._wires)}"
              f"(tq+bq+fq)  W:{cfg.windows}  K:{cfg.K}  "
              f"L:{cfg.layers}  nf:{cfg.n_features}  Params:{n_p}")

    def _register_params(self):
        cfg = self.cfg
        nf  = cfg.n_features
        self.time_decay = nn.Parameter(torch.linspace(0.3, 1.0, cfg.windows))
        for t in range(cfg.windows):
            for f in range(nf):
                self.register_parameter(
                    f"enc_crz_{t}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1)))
        for t in range(cfg.windows):
            if t % cfg.K == 0:
                for f in range(nf):
                    self.register_parameter(
                        f"snap_cry_{t}_{f}",
                        nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
                    self.register_parameter(
                        f"snap_crz_{t}_{f}",
                        nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
        for layer in range(cfg.layers):
            for f in range(nf):
                self.register_parameter(
                    f"amp_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(0, nf - 1, 2):
                self.register_parameter(
                    f"ent_even_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(1, nf - 1, 2):
                self.register_parameter(
                    f"ent_odd_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(nf):
                self.register_parameter(
                    f"bas_cry_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(nf):
                self.register_parameter(
                    f"bas_crz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
            for f in range(nf):
                self.register_parameter(
                    f"net_rz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
        self.readout_w = nn.Parameter(torch.ones(nf) * 0.1)
        self.readout_b = nn.Parameter(torch.zeros(1))

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            cfg = self.cfg
            tq, bq, fq = self.tq, self.bq, self.fq
            for t in range(cfg.windows):
                qml.RY(x[t, 0], wires=tq)
                for f in range(cfg.n_features):
                    qml.CRY(w["time_decay"][t] * x[t, f+1],
                            wires=[tq, fq[f]])
                for f in range(cfg.n_features):
                    qml.CRZ(w[f"enc_crz_{t}_{f}"], wires=[tq, fq[f]])
                if t % cfg.K == 0:
                    for f in range(cfg.n_features):
                        qml.CRY(w[f"snap_cry_{t}_{f}"], wires=[fq[f], bq])
                        qml.CRZ(w[f"snap_crz_{t}_{f}"], wires=[fq[f], bq])
            for layer in range(cfg.layers):
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                for f in range(cfg.n_features):
                    qml.CRY(w[f"bas_cry_{layer}_{f}"], wires=[bq, fq[f]])
                    qml.CRZ(w[f"bas_crz_{layer}_{f}"], wires=[bq, fq[f]])
                for f in range(cfg.n_features):
                    qml.RZ(w[f"net_rz_{layer}_{f}"], wires=fq[f])
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, 1+nf)  含 pos 列"""
        w     = dict(self.named_parameters())
        q_out = self.q_net(x, w)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  V2：QTP w/o Snapshot  —  切除 bq（宏观快照）
# ══════════════════════════════════════════════════════════════════════════════

class QTP_NoSnapshot(nn.Module):
    """
    V2：切除宏观快照（bq 完全删除）

    与 V1 的差异：
      删除：编码段中 fq→bq 的 CRY/CRZ 快照门
      删除：网络段中 bq→fq 的 CRY/CRZ 回注门
      删除：snap_cry / snap_crz / bas_cry / bas_crz 所有参数
      删除：bq 量子比特本身
      保留：tq 时间流阀门完整保留

    输入：x (W, 1+nf)  含 pos 列（tq 仍需要 pos）
    退化语义：有时间感知，无宏观记忆
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.tq     = "position"
        self.fq     = [f"feat{i}" for i in range(cfg.n_features)]
        self._wires = [self.tq] + self.fq       # ← 无 bq
        self.dev    = qml.device("default.qubit", wires=self._wires)
        self._register_params()
        self.q_net  = self._build_circuit()

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [V2 NoSnapshot]   qubits:{len(self._wires)}"
              f"(tq+fq, no bq)  W:{cfg.windows}  K:{cfg.K}  "
              f"L:{cfg.layers}  nf:{cfg.n_features}  Params:{n_p}")

    def _register_params(self):
        cfg = self.cfg
        nf  = cfg.n_features
        self.time_decay = nn.Parameter(torch.linspace(0.3, 1.0, cfg.windows))
        for t in range(cfg.windows):
            for f in range(nf):
                self.register_parameter(
                    f"enc_crz_{t}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1)))
        # ← 无 snap_cry / snap_crz
        for layer in range(cfg.layers):
            for f in range(nf):
                self.register_parameter(
                    f"amp_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(0, nf - 1, 2):
                self.register_parameter(
                    f"ent_even_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(1, nf - 1, 2):
                self.register_parameter(
                    f"ent_odd_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            # ← 无 bas_cry / bas_crz
            for f in range(nf):
                self.register_parameter(
                    f"net_rz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
        self.readout_w = nn.Parameter(torch.ones(nf) * 0.1)
        self.readout_b = nn.Parameter(torch.zeros(1))

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            cfg = self.cfg
            tq, fq = self.tq, self.fq
            for t in range(cfg.windows):
                qml.RY(x[t, 0], wires=tq)
                for f in range(cfg.n_features):
                    qml.CRY(w["time_decay"][t] * x[t, f+1],
                            wires=[tq, fq[f]])
                for f in range(cfg.n_features):
                    qml.CRZ(w[f"enc_crz_{t}_{f}"], wires=[tq, fq[f]])
                # ← 无快照门（原 if t%K==0 块完全删除）
            for layer in range(cfg.layers):
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                # ← 无 bq 回注
                for f in range(cfg.n_features):
                    qml.RZ(w[f"net_rz_{layer}_{f}"], wires=fq[f])
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, 1+nf)  含 pos 列"""
        w     = dict(self.named_parameters())
        q_out = self.q_net(x, w)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  V3：QTP w/o Time-flow  —  切除 tq（时间流阀门）
# ══════════════════════════════════════════════════════════════════════════════

class QTP_NoTimeflow(nn.Module):
    """
    V3：切除时间流阀门（tq 完全删除）

    与 V1 的差异：
      删除：tq 量子比特本身
      删除：编码段的 RY(pos) on tq
      删除：CRY(decay*x, tq→fq) 位置调制门
      删除：CRZ(enc_crz, tq→fq) 相位记忆因子
      删除：time_decay / enc_crz 所有参数
      替换：改用 RY(x[t,f]) 直接将数据编码到 fq（简单角度编码）
      保留：bq 宏观快照完整保留

    输入：x (W, nf)  不含 pos 列（tq 已删，pos 无意义）
    退化语义：普通 VQC + 宏观快照记忆，无时间阀门
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.bq     = "baseline"
        self.fq     = [f"feat{i}" for i in range(cfg.n_features)]
        self._wires = [self.bq] + self.fq       # ← 无 tq
        self.dev    = qml.device("default.qubit", wires=self._wires)
        self._register_params()
        self.q_net  = self._build_circuit()

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [V3 NoTimeflow]   qubits:{len(self._wires)}"
              f"(bq+fq, no tq)  W:{cfg.windows}  K:{cfg.K}  "
              f"L:{cfg.layers}  nf:{cfg.n_features}  Params:{n_p}")

    def _register_params(self):
        cfg = self.cfg
        nf  = cfg.n_features
        # ← 无 time_decay / enc_crz
        for t in range(cfg.windows):
            if t % cfg.K == 0:
                for f in range(nf):
                    self.register_parameter(
                        f"snap_cry_{t}_{f}",
                        nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
                    self.register_parameter(
                        f"snap_crz_{t}_{f}",
                        nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
        for layer in range(cfg.layers):
            for f in range(nf):
                self.register_parameter(
                    f"amp_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(0, nf - 1, 2):
                self.register_parameter(
                    f"ent_even_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(1, nf - 1, 2):
                self.register_parameter(
                    f"ent_odd_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(nf):
                self.register_parameter(
                    f"bas_cry_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(nf):
                self.register_parameter(
                    f"bas_crz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
            for f in range(nf):
                self.register_parameter(
                    f"net_rz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
        self.readout_w = nn.Parameter(torch.ones(nf) * 0.1)
        self.readout_b = nn.Parameter(torch.zeros(1))

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            cfg = self.cfg
            bq, fq = self.bq, self.fq
            for t in range(cfg.windows):
                # 直接角度编码：RY(x[t,f]) on fq[f]（无位置调制）
                for f in range(cfg.n_features):
                    qml.RY(x[t, f], wires=fq[f])
                if t % cfg.K == 0:
                    for f in range(cfg.n_features):
                        qml.CRY(w[f"snap_cry_{t}_{f}"], wires=[fq[f], bq])
                        qml.CRZ(w[f"snap_crz_{t}_{f}"], wires=[fq[f], bq])
            for layer in range(cfg.layers):
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                for f in range(cfg.n_features):
                    qml.CRY(w[f"bas_cry_{layer}_{f}"], wires=[bq, fq[f]])
                    qml.CRZ(w[f"bas_crz_{layer}_{f}"], wires=[bq, fq[f]])
                for f in range(cfg.n_features):
                    qml.RZ(w[f"net_rz_{layer}_{f}"], wires=fq[f])
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, nf)  不含 pos 列"""
        w     = dict(self.named_parameters())
        q_out = self.q_net(x, w)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  V4：Vanilla VQC  —  基线模型（tq + bq 全切）
# ══════════════════════════════════════════════════════════════════════════════

class VanillaVQC(nn.Module):
    """
    V4：最简 VQC 基线（tq 和 bq 全部切除）

    与 V1 的差异：
      删除：tq 量子比特及所有相关门（同 V3）
      删除：bq 量子比特及所有相关门（同 V2）
      仅保留：fq 特征量子比特 + 网络段 brick-wall 结构

    输入：x (W, nf)  不含 pos 列
    退化语义：最基础的对照组，验证创新点的贡献下界
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.fq     = [f"feat{i}" for i in range(cfg.n_features)]
        self._wires = list(self.fq)              # ← 仅 fq
        self.dev    = qml.device("default.qubit", wires=self._wires)
        self._register_params()
        self.q_net  = self._build_circuit()

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [V4 VanillaVQC]   qubits:{len(self._wires)}"
              f"(fq only)  W:{cfg.windows}  "
              f"L:{cfg.layers}  nf:{cfg.n_features}  Params:{n_p}")

    def _register_params(self):
        cfg = self.cfg
        nf  = cfg.n_features
        # ← 无 time_decay / enc_crz / snap_* / bas_*
        for layer in range(cfg.layers):
            for f in range(nf):
                self.register_parameter(
                    f"amp_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(0, nf - 1, 2):
                self.register_parameter(
                    f"ent_even_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(1, nf - 1, 2):
                self.register_parameter(
                    f"ent_odd_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            for f in range(nf):
                self.register_parameter(
                    f"net_rz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
        self.readout_w = nn.Parameter(torch.ones(nf) * 0.1)
        self.readout_b = nn.Parameter(torch.zeros(1))

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            cfg = self.cfg
            fq  = self.fq
            for t in range(cfg.windows):
                # 直接角度编码，无任何调制
                for f in range(cfg.n_features):
                    qml.RY(x[t, f], wires=fq[f])
            for layer in range(cfg.layers):
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"],
                            wires=[fq[f], fq[f+1]])
                # ← 无 bq 回注
                for f in range(cfg.n_features):
                    qml.RZ(w[f"net_rz_{layer}_{f}"], wires=fq[f])
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, nf)  不含 pos 列"""
        w     = dict(self.named_parameters())
        q_out = self.q_net(x, w)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  变体注册表
# ══════════════════════════════════════════════════════════════════════════════

VARIANTS = {
    "V1_FullQTP":       {"cls": FullQTP,        "use_pos": True},
    "V2_NoSnapshot":    {"cls": QTP_NoSnapshot,  "use_pos": True},
    "V3_NoTimeflow":    {"cls": QTP_NoTimeflow,  "use_pos": False},
    "V4_VanillaVQC":    {"cls": VanillaVQC,      "use_pos": False},
}

VARIANT_LABELS = {
    "V1_FullQTP":    "Full QTP\n(tq+bq)",
    "V2_NoSnapshot": "w/o Snapshot\n(tq only)",
    "V3_NoTimeflow": "w/o Time-flow\n(bq only)",
    "V4_VanillaVQC": "Vanilla VQC\n(no tq/bq)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  消融实验主函数
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(
    datasets:   List[str] = None,
    window:     int       = 12,
    k_ratio:    float     = 0.33,
    layers:     int       = 1,
    epochs:     int       = 50,
    lr:         float     = 5e-2,
    batch_size: int       = 16,
    patience:   int       = 10,
    result_root:str       = "./result/ablation",
    seed:       int       = SEED,
):
    """
    在指定数据集上跑 4 个消融变体，收集指标并绘图。

    Parameters
    ----------
    datasets : 数据集列表，默认 ["mackey", "lorenz"]
    """
    if datasets is None:
        datasets = ["mackey", "lorenz"]

    K = max(1, round(window * k_ratio))
    os.makedirs(result_root, exist_ok=True)

    print("=" * 65)
    print("  QTP 消融实验：4 个并列变体")
    print(f"  数据集: {datasets}")
    print(f"  W={window}  K={K}  L={layers}  epochs={epochs}")
    print("=" * 65)
    print()
    print("  变体对比：")
    print("  V1 Full QTP      tq ✅  bq ✅  完整模型")
    print("  V2 w/o Snapshot  tq ✅  bq ❌  切除宏观快照")
    print("  V3 w/o Time-flow tq ❌  bq ✅  切除时间流阀门")
    print("  V4 Vanilla VQC   tq ❌  bq ❌  最简基线")

    # {dataset: {variant_name: metrics}}
    all_results: Dict[str, Dict[str, dict]] = {}

    for dataset in datasets:
        print(f"\n{'█'*55}")
        print(f"  Dataset: {dataset}")
        print(f"{'█'*55}")
        all_results[dataset] = {}

        for v_name, v_info in VARIANTS.items():
            print(f"\n  ── {v_name} ──")
            seed_everything(seed)

            # 数据（use_pos 由变体决定）
            data_cfg  = DataConfig(
                dataname    = dataset,
                window_size = window,
                use_pos     = v_info["use_pos"],
            )
            processor = DataProcessor(data_cfg)
            (x_tr,y_tr),(x_va,y_va),(x_te,y_te) = processor.get_dataset()
            nf = processor.n_features

            # 构建模型
            cfg   = ModelConfig(windows=window, n_features=nf,
                                layers=layers, K=K)
            model = v_info["cls"](cfg).to(DEVICE)
            n_p   = sum(p.numel() for p in model.parameters())

            save_dir = os.path.join(
                result_root,
                f"{v_name}_{dataset}_W{window}_K{K}_L{layers}_P{n_p}"
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            trainer   = Trainer(
                model      = model,
                optimizer  = optimizer,
                batch_size = batch_size,
                patience   = patience,
                save_path  = save_dir,
            )

            t0 = time.time()
            tr_losses, va_losses = trainer.train(
                x_tr, y_tr, x_va, y_va, epochs=epochs)
            elapsed = time.time() - t0

            _, preds = trainer.evaluate(x_te, y_te)
            metrics  = print_metrics(v_name, dataset, y_te, preds,
                                     n_params=n_p)
            metrics.update({
                "n_params":  n_p,
                "elapsed":   elapsed,
                "tr_losses": tr_losses,
                "va_losses": va_losses,
                "preds":     preds.numpy(),
                "y_te":      y_te.numpy(),
            })
            all_results[dataset][v_name] = metrics

            np.save(os.path.join(save_dir, "preds.npy"), preds.numpy())
            np.save(os.path.join(save_dir, "y_te.npy"),  y_te.numpy())

    # ── 汇总打印 ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  消融实验汇总")
    print(f"{'═'*70}")
    v_names = list(VARIANTS.keys())
    header  = f"  {'Variant':<22}" + "".join(
        f"{'R²_'+d:>14}" for d in datasets) + f"  {'Params':>8}"
    print(header)
    print("  " + "─" * (22 + 14*len(datasets) + 10))
    for v_name in v_names:
        row = f"  {v_name:<22}"
        for d in datasets:
            m = all_results[d].get(v_name)
            row += f"{'—':>14}" if m is None else f"{m['r2']:>14.6f}"
        m0 = all_results[datasets[0]].get(v_name)
        row += f"  {m0['n_params']:>8}" if m0 else ""
        print(row)

    # ── 绘图 ──────────────────────────────────────────────────────────
    _plot_ablation(all_results, datasets, result_root)

    np.save(os.path.join(result_root, "ablation_results.npy"), all_results)
    print(f"\n  消融实验完成 → {result_root}/")
    return all_results


def _plot_ablation(
    all_results: dict,
    datasets:    List[str],
    result_root: str,
) -> None:
    """绘制消融实验结果图：R² 柱状图 + 训练曲线。"""
    v_names = list(VARIANTS.keys())
    labels  = [VARIANT_LABELS[v] for v in v_names]
    colors  = ["#378ADD", "#D85A30", "#1D9E75", "#888888"]

    n_ds  = len(datasets)
    fig, axes = plt.subplots(n_ds, 2, figsize=(14, 4.5 * n_ds))
    if n_ds == 1:
        axes = axes[np.newaxis, :]

    for row, dataset in enumerate(datasets):

        # 左图：R² 柱状图
        ax  = axes[row, 0]
        r2s = []
        for v_name in v_names:
            m = all_results[dataset].get(v_name)
            r2s.append(m["r2"] if m else 0.0)

        bars = ax.bar(range(len(v_names)), r2s,
                      color=colors, edgecolor="black", linewidth=0.5)
        # 标注数值
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    r2 + 0.005, f"{r2:.4f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(v_names)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(f"Ablation R² — {dataset}", fontsize=10)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        ax.set_ylim(bottom=min(min(r2s) - 0.05, -0.05))
        ax.grid(True, alpha=0.3, axis="y")

        # V1 基准线
        v1_r2 = all_results[dataset].get("V1_FullQTP", {}).get("r2", None)
        if v1_r2 is not None:
            ax.axhline(v1_r2, color="#378ADD", lw=1.2,
                       linestyle=":", alpha=0.7, label=f"V1 baseline={v1_r2:.4f}")
            ax.legend(fontsize=7)

        # 右图：训练曲线对比
        ax = axes[row, 1]
        for i, v_name in enumerate(v_names):
            m = all_results[dataset].get(v_name)
            if m is None or not m.get("tr_losses"):
                continue
            ax.plot(m["tr_losses"], color=colors[i],
                    lw=1.5, label=f"{v_name} train")
            ax.plot(m["va_losses"], color=colors[i],
                    lw=1.0, linestyle="--", alpha=0.6,
                    label=f"{v_name} valid")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (normalised)")
        ax.set_title(f"Training curves — {dataset}", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle("QTP Ablation Study: 4 Parallel Variants", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(result_root, "ablation_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figure saved → {save_path}")

    # 贡献拆解图：各组件对 V1 的 R² 贡献
    if len(datasets) >= 1:
        _plot_contribution(all_results, datasets, result_root,
                           v_names, labels, colors)


def _plot_contribution(
    all_results: dict,
    datasets:    List[str],
    result_root: str,
    v_names:     List[str],
    labels:      List[str],
    colors:      List[str],
) -> None:
    """
    绘制各变体相对于 V4 Vanilla VQC 的 R² 提升量。
    量化每个组件的贡献：
      tq 贡献  = V2_NoSnapshot - V4_VanillaVQC
      bq 贡献  = V3_NoTimeflow - V4_VanillaVQC
      协同贡献 = V1_FullQTP - V2_NoSnapshot - V3_NoTimeflow + V4_VanillaVQC
    """
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        res = all_results[dataset]

        def _r2(v): return res.get(v, {}).get("r2", 0.0) or 0.0

        v1 = _r2("V1_FullQTP")
        v2 = _r2("V2_NoSnapshot")
        v3 = _r2("V3_NoTimeflow")
        v4 = _r2("V4_VanillaVQC")

        contrib_tq   = v2 - v4          # tq 单独贡献
        contrib_bq   = v3 - v4          # bq 单独贡献
        contrib_syn  = v1 - v2 - v3 + v4 # 协同效应

        comp_names = ["tq\n(Time-flow)", "bq\n(Snapshot)", "Synergy\n(tq×bq)"]
        comp_vals  = [contrib_tq, contrib_bq, contrib_syn]
        comp_colors= ["#378ADD", "#D85A30", "#1D9E75"]

        bars = ax.bar(comp_names, comp_vals,
                      color=comp_colors, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=0.8, linestyle="--")
        for bar, val in zip(bars, comp_vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (0.002 if val >= 0 else -0.01),
                    f"{val:+.4f}", ha="center",
                    fontsize=9, fontweight="bold")
        ax.set_ylabel("ΔR² vs V4 Vanilla VQC")
        ax.set_title(f"Component contribution — {dataset}\n"
                     f"V1={v1:.4f}  V4={v4:.4f}  "
                     f"Total gain={v1-v4:+.4f}")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("QTP Component Contribution Analysis", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(result_root, "ablation_contribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Figure saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════════

def re_plot():
    # -*- coding: utf-8 -*-
    """
    plot_ablation_waterfall.py
    ==========================
    以数据集为单位，绘制消融实验瀑布图。
    
    瀑布图逻辑（每个数据集一列）：
      起点  → V4 Vanilla VQC 的 R²（基线）
      +tq   → 加入时间流阀门的增益  (V2 - V4)
      +bq   → 加入快照记忆的增益    (V3 - V4)
      +syn  → 协同效应              (V1 - V2 - V3 + V4)
      终点  → V1 Full QTP 的 R²（验证：= V4 + tq + bq + syn）
    
    运行：python plot_ablation_waterfall.py
    """
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as ticker
    
    # ── 配置 ──────────────────────────────────────────────────────────────────────
    NPY_PATH = "./result/ablation/ablation_results.npy"
    SAVE_DIR = "./result/ablation"
    DATASETS = ["mackey", "lorenz"]
    
    DS_LABELS = {"mackey": "Mackey-Glass", "lorenz": "Lorenz"}
    
    COLOR_BASE    = "#D1C7BD"   # V4 基线柱
    COLOR_POS     = "#C87A3E"   # 正增益
    COLOR_NEG     = "#8A7968"   # 负增益（协同可能为负）
    COLOR_TOTAL   = "#4A3B32"   # V1 终点柱
    COLOR_TQ      = "#378ADD"
    COLOR_BQ      = "#1D9E75"
    COLOR_SYN     = "#1D4E5C"
    COLOR_CONNECT = "#999999"   # 连接线
    
    
    # ── 加载 ───────────────────────────────────────────────────────────────────────
    print(f"Loading: {NPY_PATH}")
    all_results = np.load(NPY_PATH, allow_pickle=True).item()
    
    
    def get_r2(dataset, variant):
        m = all_results.get(dataset, {}).get(variant)
        return m["r2"] if m else 0.0
    
    
    # ── 计算每个数据集的瀑布分量 ─────────────────────────────────────────────────
    segments = {}
    for d in DATASETS:
        v1 = get_r2(d, "V1_FullQTP")
        v2 = get_r2(d, "V2_NoSnapshot")
        v3 = get_r2(d, "V3_NoTimeflow")
        v4 = get_r2(d, "V4_VanillaVQC")
        segments[d] = {
            "base":    v4,
            "tq":      v2 - v4,
            "bq":      v3 - v4,
            "synergy": v1 - v2 - v3 + v4,
            "total":   v1,
        }
        print(f"[{d}]  base={v4:.4f}  "
              f"+tq={segments[d]['tq']:+.4f}  "
              f"+bq={segments[d]['bq']:+.4f}  "
              f"+syn={segments[d]['synergy']:+.4f}  "
              f"→ total={v1:.4f}")
    
    
    # ══════════════════════════════════════════════════════════════════════════════
    #  绘图
    # ══════════════════════════════════════════════════════════════════════════════
    
    n_ds    = len(DATASETS)
    # 每个数据集有 5 个柱子：base / +tq / +bq / +syn / total
    # 柱子之间加间隔，数据集之间加更大间隔
    STEPS      = ["Vanilla VQC\n(base)", "+Time qubit\n(Time-flow)",
                  "+History qubit\n(Snapshot)", "+Synergy", "Full QTP\n(total)"]
    N_STEPS    = len(STEPS)
    BAR_W      = 0.55
    DS_GAP     = 1.2    # 数据集之间额外间距
    
    # 计算每组柱子的 x 中心
    ds_centers = []
    x_cursor   = 0.0
    for i in range(n_ds):
        start = x_cursor
        end   = x_cursor + N_STEPS - 1
        ds_centers.append((start + end) / 2)
        x_cursor = end + DS_GAP + 1
    
    fig, ax = plt.subplots(figsize=(5 * n_ds + 2, 7))
    fig.suptitle("QTP Ablation — Waterfall Chart by Dataset\n"
                 "Decomposing R² gain: Vanilla VQC → Full QTP",
                 fontsize=12, fontweight="bold")
    
    # step_colors = [COLOR_BASE, COLOR_TQ, COLOR_BQ, COLOR_SYN, COLOR_TOTAL]
    # COLOR_BASE    = "#D1C7BD"   # V4 基线柱
    # # COLOR_POS     = "#C87A3E"   # 正增益
    # COLOR_NEG     = "#8A7968"   # 负增益（协同可能为负）
    # COLOR_TOTAL   = "#4A3B32"   # V1 终点柱
    # COLOR_TQ      = "#C87A3E"
    # COLOR_BQ      = "#8A7968"
    # COLOR_SYN     = "#7B9095"#"#1D4E5C"
    # COLOR_CONNECT = "#999999"   # 连接线
    
    
    COLOR_BASE    = "#4A4A4A"   # V4 基线柱
    # COLOR_POS     = "#C87A3E"   # 正增益
    COLOR_NEG     = "#8A7968"   # 负增益（协同可能为负）
    COLOR_TOTAL   = "#2F5D8C"   # V1 终点柱
    COLOR_TQ      = "#C87A3E"
    COLOR_BQ      = "#759095"
    COLOR_SYN     = "#D6CDC5"#"#1D4E5C"
    COLOR_CONNECT = "#999999"   # 连接线
    
    step_colors = [COLOR_BASE, COLOR_TQ, COLOR_BQ, COLOR_SYN, COLOR_TOTAL]
    for di, d in enumerate(DATASETS):
        seg      = segments[d]
        base_val = seg["base"]
        deltas   = [0, seg["tq"], seg["bq"], seg["synergy"], 0]
    
        # 计算每个柱子的 bottom 和 height
        bottoms = []
        heights = []
    
        # 柱0：base（从 0 到 base_val）
        bottoms.append(0.0)
        heights.append(base_val)
    
        # 柱1~3：增量（bottom = 上一步的顶部）
        running = base_val
        for k in range(1, 4):
            delta = deltas[k]
            if delta >= 0:
                bottoms.append(running)
                heights.append(delta)
            else:
                bottoms.append(running + delta)
                heights.append(-delta)
            running += delta
    
        # 柱4：total（从 0 到 total）
        bottoms.append(0.0)
        heights.append(seg["total"])
    
        # 确定每个柱子的 x 位置
        x_start = di * (N_STEPS + DS_GAP)
        xs = [x_start + j for j in range(N_STEPS)]
    
        for j, (x, bot, h) in enumerate(zip(xs, bottoms, heights)):
            # 颜色：增量柱按正负区分
            if j == 0:
                color = COLOR_BASE
            elif j == 4:
                color = COLOR_TOTAL
            else:
                delta = deltas[j]
                if j == 1:   color = COLOR_TQ
                elif j == 2: color = COLOR_BQ
                else:        color = COLOR_SYN if delta >= 0 else COLOR_SYN
    
            bar = ax.bar(x, h, bottom=bot, width=BAR_W,
                         color=color, edgecolor="white",
                         linewidth=0.8, alpha=0.88, zorder=3)
    
            # 数值标注
            val_text = (f"{bot + h:.4f}" if j in (0, 4)
                        else f"{deltas[j]:+.4f}")
            label_y  = bot + h + 0.003
            ax.text(x, label_y, val_text,
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                    color="#222222", zorder=5)
    
        # 连接线（浮动柱之间画虚线表示延续）
        running2 = base_val
        for j in range(1, 4):
            top_prev = running2
            ax.plot([xs[j-1] + BAR_W/2 + 0.02,
                     xs[j]   - BAR_W/2 - 0.02],
                    [top_prev, top_prev],
                    color=COLOR_CONNECT, lw=1.0,
                    linestyle="--", alpha=0.7, zorder=2)
            running2 += deltas[j]
    
        # 数据集标题（x 轴分组标签）
        ax.text(0.01 + di * 0.5,   # 多个数据集时水平错开
            0.97,
            DS_LABELS.get(d, d),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", alpha=0.75,
                      edgecolor="#cccccc"))
        
        
        
        # 分隔线
        if di < n_ds - 1:
            sep_x = xs[-1] + BAR_W/2 + DS_GAP/2
            ax.axvline(sep_x, color="#DDDDDD", lw=1.2,
                       linestyle="-", zorder=1)
    
    
    # x 轴刻度：显示 step 名称
    all_xs = []
    all_labels = []
    for di in range(n_ds):
        x_start = di * (N_STEPS + DS_GAP)
        for j, label in enumerate(STEPS):
            all_xs.append(x_start + j)
            all_labels.append(label)
    
    ax.set_xticks(all_xs)
    ax.set_xticklabels(all_labels, fontsize=7.5, rotation=0)
    ax.set_ylabel("$R^2$", fontsize=11)
    ax.set_title("")
    ax.axhline(0, color="black", lw=0.8, linestyle="-", alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.2, axis="y", lw=0.6)
    ax.set_xlim(-0.6, all_xs[-1] + 0.6)
    
    # 图例
    legend_patches = [
        mpatches.Patch(color=COLOR_BASE,  label="Vanilla VQC (base R²)"),
        mpatches.Patch(color=COLOR_TQ,    label="+tq  Time-flow gate"),
        mpatches.Patch(color=COLOR_BQ,    label="+bq  Snapshot memory"),
        mpatches.Patch(color=COLOR_SYN,   label="+Synergy  (tq × bq)"),
        mpatches.Patch(color=COLOR_TOTAL, label="Full QTP (total R²)"),
    ]
    # ax.legend(handles=legend_patches, fontsize=8.5,
    #           loc="lower right", framealpha=0.9)
    
    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "ablation_waterfall.png")
    # plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved → {save_path}")





if __name__ == "__main__":
    # 日志同时写入文件
    # _log = open("log_ablation.log", "a", encoding="utf-8")

    # class _Tee:
    #     def __init__(self, *s): self.s = s
    #     def write(self, d):
    #         for x in self.s: x.write(d); x.flush()
    #     def flush(self):
    #         for x in self.s: x.flush()

    # sys.stdout = _Tee(sys.__stdout__, _log)
    # sys.stderr = _Tee(sys.__stderr__, _log)

    # run_ablation(
    #     datasets    = ["mackey", "lorenz"],   # 按需修改
    #     window      = 12,
    #     k_ratio     = 0.33,
    #     layers      = 1,
    #     epochs      = 50,
    #     lr          = 5e-2,
    #     batch_size  = 16,
    #     patience    = 10,
    #     result_root = "./result/ablation",
    #     seed        = SEED,
    # )
    
    re_plot()