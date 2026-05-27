# -*- coding: utf-8 -*-
"""
Model.py
========
所有模型定义，包含：
  量子模型：
    QuantumTemporalPredictor (QTP)   本文提出
    VQCPredictor             (VQC)   变分量子电路
    QLSTMPredictor           (QLSTM) 量子 LSTM
    QRNNPredictor            (QRNN)  量子 RNN
  经典模型（对比基线）：
    LSTMPredictor
    TransformerPredictor
    MLPPredictor
    ARIMAXPredictor          统计 ARIMAX（statsmodels）

所有模型统一接口：
  forward(x: Tensor) → scalar
  x shape:
    QTP          → (W, 1+n_features)  含 pos 列
    其余模型     → (W, n_features)    不含 pos 列
"""

import math
import warnings
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# warnings.filterwarnings("ignore")

DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  ModelConfig  —  QTP 专用超参
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    windows:    int = 12
    n_features: int = 1
    layers:     int = 1
    K:          int = 4


# ══════════════════════════════════════════════════════════════════════════════
#  1. QTP  —  量子时序预测器（本文提出）
# ══════════════════════════════════════════════════════════════════════════════
class QuantumTemporalPredictor_last(nn.Module):
    """
    量子时序预测器 (QTP)

    创新点：
      1. 时间流编码：position qubit 作为时间阀门，CRY(x[t,f], tq→fq) 实现位置调制
      2. 延迟分步记忆：baseline qubit 每 K 步做相位快照，网络段反馈给 feature qubit

    电路结构：
      编码段  for t in 0..W-1:
                RY(pos[t])                     on tq
                CRY(decay[t]*x[t,f], tq→fq)   位置调制 + 时间衰减
                CRZ(enc_crz[t,f],    tq→fq)   相位记忆因子
                if t%K==0:
                  CRY(snap_cry, fq→bq)         振幅快照
                  CRZ(snap_crz, fq→bq)         相位快照
      网络段  for layer:
                RY(amp) on fq; brick-wall CRY; CRY/CRZ(bq→fq); RZ on fq
      读出段  y = w·⟨Z_fq⟩ + b  （线性）
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.tq     = "position"
        self.bq     = "baseline"
        self.fq     = [f"feat{i}" for i in range(cfg.n_features)]
        self._wires = [self.tq, self.bq] + self.fq
        self.dev    = qml.device("default.qubit", wires=self._wires)
        self._register_parameters()
        self.q_net  = self._build_circuit()

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [QTP] Qubits:{len(self._wires)}  "
              f"W:{cfg.windows}  K:{cfg.K}  L:{cfg.layers}  "
              f"nf:{cfg.n_features}  Params:{n_p}")

    def _register_parameters(self):
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
            #重置Position位
            self.register_parameter(
                f"phi_{layer}",
                nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            self.register_parameter(
                f"theta{layer}",
                nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            self.register_parameter(
                f"omega{layer}",
                nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            
            #记忆系统编码后最后状态
            for f in range(nf):
                self.register_parameter(
                    f"last_rem_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
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
                self.register_parameter(
                    f"bas_crz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))

            for f in range(nf):
                self.register_parameter(
                    f"net_rz_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.3, 0.3)))
            #最后状态回注
            for f in range(nf):
                self.register_parameter(
                    f"last_inj_{layer}_{f}",
                    nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5)))
            
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
                    qml.CRY(w["time_decay"][t] * x[t, f+1], wires=[tq, fq[f]])
                for f in range(cfg.n_features):
                    qml.CRZ(w[f"enc_crz_{t}_{f}"], wires=[tq, fq[f]])
                if t % cfg.K == 0:
                    for f in range(cfg.n_features):
                        qml.CRY(w[f"snap_cry_{t}_{f}"], wires=[fq[f], bq])
                        qml.CRZ(w[f"snap_crz_{t}_{f}"], wires=[fq[f], bq])
            qml.Barrier(wires=self._wires)
            
            
            for layer in range(cfg.layers):
                
                qml.Rot(w[f"phi_{layer}"],w[f"phi_{layer}"],w[f"omega{layer}"], wires=tq)
                
                for f in range(cfg.n_features):
                    qml.CRY(w[f"last_rem_{layer}_{f}"], wires=[fq[f],tq])
                    
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                for f in range(cfg.n_features):
                    qml.CRY(w[f"bas_cry_{layer}_{f}"], wires=[bq, fq[f]])
                    qml.CRZ(w[f"bas_crz_{layer}_{f}"], wires=[bq, fq[f]])
                for f in range(cfg.n_features):
                    qml.RZ(w[f"net_rz_{layer}_{f}"], wires=fq[f])
                    
                for f in range(cfg.n_features):
                    qml.CRY(w[f"last_inj_{layer}_{f}"], wires=[tq,fq[f]])
            qml.Barrier(wires=[self.bq] + self.fq)
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, 1+n_features)  含 pos 列"""
        w     = dict(self.named_parameters())
        q_out = self.q_net(x, w)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)

    def _build_noisy_circuit(self):
        """噪声推理专用电路（default.mixed，无 Barrier）"""
        noisy_dev = qml.device("default.mixed", wires=self._wires)

        @qml.qnode(noisy_dev, interface="torch", diff_method="best")
        def circuit(x, w, noise_type=None, noise_p=0.0):
            def apply_noise(wire):
                if noise_type is None or noise_p == 0.0:
                    return
                if noise_type == "depolarizing":
                    qml.DepolarizingChannel(noise_p, wires=wire)
                elif noise_type == "bit_flip":
                    qml.BitFlip(noise_p, wires=wire)
                elif noise_type == "phase_flip":
                    qml.PhaseFlip(noise_p, wires=wire)
                elif noise_type == "amplitude_damp":
                    qml.AmplitudeDamping(noise_p, wires=wire)

            zero = torch.zeros(1)

            def noisy_RZ(angle, wire):
                qml.Rot(angle, zero, zero, wires=wire)

            def noisy_CRZ(angle, ctrl, tgt):
                qml.CRot(angle, zero, zero, wires=[ctrl, tgt])
            

            cfg = self.cfg
            tq, bq, fq = self.tq, self.bq, self.fq
            for t in range(cfg.windows):
                qml.RY(x[t, 0], wires=tq)
                apply_noise(tq)
                for f in range(cfg.n_features):
                    qml.CRY(w["time_decay"][t] * x[t, f+1], wires=[tq, fq[f]])
                    apply_noise(fq[f])
                for f in range(cfg.n_features):
                    noisy_CRZ(w[f"enc_crz_{t}_{f}"], tq, fq[f])
                    apply_noise(fq[f])
                if t % cfg.K == 0:
                    for f in range(cfg.n_features):
                        qml.CRY(w[f"snap_cry_{t}_{f}"], wires=[fq[f], bq])
                        apply_noise(bq)
                        noisy_CRZ(w[f"snap_crz_{t}_{f}"], fq[f], bq)
                        apply_noise(bq)
            for layer in range(cfg.layers):                
                qml.Rot(w[f"phi_{layer}"],w[f"phi_{layer}"],w[f"omega{layer}"], wires=tq)
                apply_noise(tq)
                for f in range(cfg.n_features):
                    qml.CRY(w[f"last_rem_{layer}_{f}"], wires=[fq[f],tq])
                    apply_noise(tq)
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                    apply_noise(fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                    apply_noise(fq[f+1])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                    apply_noise(fq[f+1])
                for f in range(cfg.n_features):
                    qml.CRY(w[f"bas_cry_{layer}_{f}"], wires=[bq, fq[f]])
                    apply_noise(fq[f])
                    noisy_CRZ(w[f"bas_crz_{layer}_{f}"], bq, fq[f])
                    apply_noise(fq[f])
                for f in range(cfg.n_features):
                    noisy_RZ(w[f"net_rz_{layer}_{f}"], fq[f])
                    apply_noise(fq[f])
                    
                for f in range(cfg.n_features):
                    qml.CRY(w[f"last_inj_{layer}_{f}"], wires=[tq,fq[f]])
                    apply_noise(fq[f])
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    @torch.no_grad()
    def forward_noisy(self, x: torch.Tensor,
                      noise_type: str = None,
                      noise_p: float = 0.0) -> torch.Tensor:
        """噪声推理接口，不参与训练。"""
        w         = dict(self.named_parameters())
        noisy_net = self._build_noisy_circuit()
        q_out     = noisy_net(x, w, noise_type=noise_type, noise_p=noise_p)
        q_out     = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)



class QuantumTemporalPredictor(nn.Module):
    """
    量子时序预测器 (QTP)

    创新点：
      1. 时间流编码：position qubit 作为时间阀门，CRY(x[t,f], tq→fq) 实现位置调制
      2. 延迟分步记忆：baseline qubit 每 K 步做相位快照，网络段反馈给 feature qubit

    电路结构：
      编码段  for t in 0..W-1:
                RY(pos[t])                     on tq
                CRY(decay[t]*x[t,f], tq→fq)   位置调制 + 时间衰减
                CRZ(enc_crz[t,f],    tq→fq)   相位记忆因子
                if t%K==0:
                  CRY(snap_cry, fq→bq)         振幅快照
                  CRZ(snap_crz, fq→bq)         相位快照
      网络段  for layer:
                RY(amp) on fq; brick-wall CRY; CRY/CRZ(bq→fq); RZ on fq
      读出段  y = w·⟨Z_fq⟩ + b  （线性）
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg    = cfg
        self.tq     = "position"
        self.bq     = "baseline"
        self.fq     = [f"feat{i}" for i in range(cfg.n_features)]
        self._wires = [self.tq, self.bq] + self.fq
        self.dev    = qml.device("default.qubit", wires=self._wires)
        self._register_parameters()
        self.q_net  = self._build_circuit()

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [QTP] Qubits:{len(self._wires)}  "
              f"W:{cfg.windows}  K:{cfg.K}  L:{cfg.layers}  "
              f"nf:{cfg.n_features}  Params:{n_p}")

    def _register_parameters(self):
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
                    qml.CRY(w["time_decay"][t] * x[t, f+1], wires=[tq, fq[f]])
                for f in range(cfg.n_features):
                    qml.CRZ(w[f"enc_crz_{t}_{f}"], wires=[tq, fq[f]])
                if t % cfg.K == 0:
                    for f in range(cfg.n_features):
                        qml.CRY(w[f"snap_cry_{t}_{f}"], wires=[fq[f], bq])
                        qml.CRZ(w[f"snap_crz_{t}_{f}"], wires=[fq[f], bq])
            qml.Barrier(wires=self._wires)
            for layer in range(cfg.layers):
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                for f in range(cfg.n_features):
                    qml.CRY(w[f"bas_cry_{layer}_{f}"], wires=[bq, fq[f]])
                    qml.CRZ(w[f"bas_crz_{layer}_{f}"], wires=[bq, fq[f]])
                for f in range(cfg.n_features):
                    qml.RZ(w[f"net_rz_{layer}_{f}"], wires=fq[f])
            qml.Barrier(wires=[self.bq] + self.fq)
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, 1+n_features)  含 pos 列"""
        w     = dict(self.named_parameters())
        q_out = self.q_net(x, w)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)

    def _build_noisy_circuit(self):
        """噪声推理专用电路（default.mixed，无 Barrier）"""
        noisy_dev = qml.device("default.mixed", wires=self._wires)

        @qml.qnode(noisy_dev, interface="torch", diff_method="best")
        def circuit(x, w, noise_type=None, noise_p=0.0):
            def apply_noise(wire):
                if noise_type is None or noise_p == 0.0:
                    return
                if noise_type == "depolarizing":
                    qml.DepolarizingChannel(noise_p, wires=wire)
                elif noise_type == "bit_flip":
                    qml.BitFlip(noise_p, wires=wire)
                elif noise_type == "phase_flip":
                    qml.PhaseFlip(noise_p, wires=wire)
                elif noise_type == "amplitude_damp":
                    qml.AmplitudeDamping(noise_p, wires=wire)

            zero = torch.zeros(1)

            def noisy_RZ(angle, wire):
                qml.Rot(angle, zero, zero, wires=wire)

            def noisy_CRZ(angle, ctrl, tgt):
                qml.CRot(angle, zero, zero, wires=[ctrl, tgt])

            cfg = self.cfg
            tq, bq, fq = self.tq, self.bq, self.fq
            for t in range(cfg.windows):
                qml.RY(x[t, 0], wires=tq)
                apply_noise(tq)
                for f in range(cfg.n_features):
                    qml.CRY(w["time_decay"][t] * x[t, f+1], wires=[tq, fq[f]])
                    apply_noise(fq[f])
                for f in range(cfg.n_features):
                    noisy_CRZ(w[f"enc_crz_{t}_{f}"], tq, fq[f])
                    apply_noise(fq[f])
                if t % cfg.K == 0:
                    for f in range(cfg.n_features):
                        qml.CRY(w[f"snap_cry_{t}_{f}"], wires=[fq[f], bq])
                        apply_noise(bq)
                        noisy_CRZ(w[f"snap_crz_{t}_{f}"], fq[f], bq)
                        apply_noise(bq)
            for layer in range(cfg.layers):
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                    apply_noise(fq[f])
                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                    apply_noise(fq[f+1])
                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"], wires=[fq[f], fq[f+1]])
                    apply_noise(fq[f+1])
                for f in range(cfg.n_features):
                    qml.CRY(w[f"bas_cry_{layer}_{f}"], wires=[bq, fq[f]])
                    apply_noise(fq[f])
                    noisy_CRZ(w[f"bas_crz_{layer}_{f}"], bq, fq[f])
                    apply_noise(fq[f])
                for f in range(cfg.n_features):
                    noisy_RZ(w[f"net_rz_{layer}_{f}"], fq[f])
                    apply_noise(fq[f])
            return [qml.expval(qml.PauliZ(wire)) for wire in fq]
        return circuit

    @torch.no_grad()
    def forward_noisy(self, x: torch.Tensor,
                      noise_type: str = None,
                      noise_p: float = 0.0) -> torch.Tensor:
        """噪声推理接口，不参与训练。"""
        w         = dict(self.named_parameters())
        noisy_net = self._build_noisy_circuit()
        q_out     = noisy_net(x, w, noise_type=noise_type, noise_p=noise_p)
        q_out     = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return (torch.sum(self.readout_w * q_out) + self.readout_b
                ).squeeze().to(dtype=torch.float32, device=DEVICE)






# ══════════════════════════════════════════════════════════════════════════════
#  2. VQCPredictor
# ══════════════════════════════════════════════════════════════════════════════

class VQCPredictor(nn.Module):
    """
    变分量子电路预测器
    编码：经典线性层 → RY(π·x_i) 角度编码 → L 层 RX/RY/RZ + 环形 CNOT
    参数：quantum:{num_qubits*num_layers*3}  classical:{input_layer+output_layer}
    """

    def __init__(self, window: int, n_features: int,
                 num_qubits: int = 4, num_layers: int = 1,
                 diff_method: str = "backprop") -> None:
        super().__init__()
        self.window     = window
        self.n_features = n_features
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        self.input_layer  = nn.Linear(window * n_features, num_qubits)
        self.output_layer = nn.Linear(num_qubits, 1)

        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for i in range(num_qubits):
                qml.RY(torch.pi * inputs[i], wires=i)
            for j in range(num_layers):
                for i in range(num_qubits):
                    qml.RX(weights[i, j, 0], wires=i)
                    qml.RY(weights[i, j, 1], wires=i)
                    qml.RZ(weights[i, j, 2], wires=i)
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[num_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.q_weights = nn.Parameter(
            torch.empty(num_qubits, num_layers, 3).uniform_(0, 2 * math.pi))
        self.circuit = circuit

        n_q = self.q_weights.numel()
        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [VQC] W:{window}  nf:{n_features}  "
              f"qubits:{num_qubits}  layers:{num_layers}  "
              f"Params:{n_p} (quantum:{n_q} + classical:{n_p-n_q})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, n_features)"""
        x_enc = torch.tanh(self.input_layer(x.flatten()))
        q_out = self.circuit(x_enc, self.q_weights)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return self.output_layer(q_out.to(torch.float32)
                                 ).squeeze().to(dtype=torch.float32,
                                                device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  3. QLSTMPredictor
# ══════════════════════════════════════════════════════════════════════════════

class QLSTMPredictor(nn.Module):
    """
    量子 LSTM 预测器（线性增强版）
    四个量子门各有独立参数集，经典线性层做投影
    参数：quantum:{4*num_qubits*num_layers*3}  classical:{linear layers}
    """

    def __init__(self, window: int, n_features: int,
                 num_qubits: int = 4, hidden_size: int = 4,
                 num_layers: int = 1,
                 diff_method: str = "backprop") -> None:
        super().__init__()
        self.window      = window
        self.n_features  = n_features
        self.num_qubits  = num_qubits
        self.hidden_size = hidden_size

        dev = qml.device("default.qubit", wires=num_qubits)

        def make_circuit():
            @qml.qnode(dev, interface="torch", diff_method=diff_method)
            def circuit(inputs, weights):
                for i in range(num_qubits):
                    qml.Hadamard(wires=i)
                    qml.RY(torch.arctan(inputs[i]),      wires=i)
                    qml.RZ(torch.arctan(inputs[i] ** 2), wires=i)
                for j in range(num_layers):
                    for i in range(num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[num_qubits - 1, 0])
                    for i in range(num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[num_qubits - 1, 0])
                    for i in range(num_qubits):
                        qml.Rot(weights[i, j, 0],
                                weights[i, j, 1],
                                weights[i, j, 2], wires=i)
                return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
            return circuit

        self.vqc1 = make_circuit()
        self.vqc2 = make_circuit()
        self.vqc3 = make_circuit()
        self.vqc4 = make_circuit()

        w_shape = (num_qubits, num_layers, 3)
        self.w1 = nn.Parameter(torch.empty(*w_shape).uniform_(0, 2*math.pi))
        self.w2 = nn.Parameter(torch.empty(*w_shape).uniform_(0, 2*math.pi))
        self.w3 = nn.Parameter(torch.empty(*w_shape).uniform_(0, 2*math.pi))
        self.w4 = nn.Parameter(torch.empty(*w_shape).uniform_(0, 2*math.pi))

        self.linear_in    = nn.Linear(n_features + hidden_size, num_qubits)
        self.linear_f     = nn.Linear(num_qubits, hidden_size)
        self.linear_i     = nn.Linear(num_qubits, hidden_size)
        self.linear_g     = nn.Linear(num_qubits, hidden_size)
        self.linear_o     = nn.Linear(num_qubits, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        n_q = (self.w1.numel() + self.w2.numel() +
               self.w3.numel() + self.w4.numel())
        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [QLSTM] W:{window}  nf:{n_features}  "
              f"qubits:{num_qubits}  hidden:{hidden_size}  "
              f"Params:{n_p} (quantum:{n_q} + classical:{n_p-n_q})")

    def _vqc(self, circuit, weights, x):
        out = circuit(x, weights)
        out = torch.stack(out) if isinstance(out, list) else out
        return out.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, n_features)"""
        h = torch.zeros(self.hidden_size, device=x.device)
        c = torch.zeros(self.hidden_size, device=x.device)
        for t in range(self.window):
            v = torch.tanh(self.linear_in(
                torch.cat([x[t], h], dim=0)))
            f = torch.sigmoid(self.linear_f(self._vqc(self.vqc1, self.w1, v)))
            i = torch.sigmoid(self.linear_i(self._vqc(self.vqc2, self.w2, v)))
            g = torch.tanh   (self.linear_g(self._vqc(self.vqc3, self.w3, v)))
            o = torch.sigmoid(self.linear_o(self._vqc(self.vqc4, self.w4, v)))
            c = f * c + i * g
            h = o * torch.tanh(c)
        return self.output_layer(h).squeeze().to(dtype=torch.float32,
                                                  device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  4. QRNNPredictor
# ══════════════════════════════════════════════════════════════════════════════

class QRNNPredictor(nn.Module):
    """
    量子 RNN 预测器
    data qubit + hidden qubit 联合演化，只测量 data qubit
    参数：quantum:{num_qubits*4}  classical:{output_layer}
    """

    def __init__(self, window: int, n_features: int,
                 num_qubits: int = 4, num_qubits_hidden: int = 2,
                 diff_method: str = "backprop") -> None:
        super().__init__()
        self.window            = window
        self.n_features        = n_features
        self.num_qubits        = num_qubits
        self.num_qubits_hidden = num_qubits_hidden
        self.num_qubits_data   = num_qubits - num_qubits_hidden

        assert self.num_qubits_data >= 1, \
            "num_qubits_data = num_qubits - num_qubits_hidden 须 >= 1"

        dev = qml.device("default.qubit", wires=num_qubits)
        nqd = self.num_qubits_data
        nf  = n_features

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for t in range(window):
                feat_mean = inputs[t*nf:(t+1)*nf].mean()
                for j in range(nqd):
                    qml.RY(torch.arccos(
                        feat_mean.clamp(-1+1e-6, 1-1e-6)), wires=j)
                for j in range(num_qubits):
                    qml.RX(weights[j, 0], wires=j)
                    qml.RZ(weights[j, 1], wires=j)
                    qml.RX(weights[j, 2], wires=j)
                for j in range(num_qubits - 1):
                    qml.CNOT(wires=[j, j+1])
                    qml.RZ(weights[j+1, 3], wires=j+1)
                    qml.CNOT(wires=[j, j+1])
                qml.CNOT(wires=[num_qubits-1, 0])
                qml.RZ(weights[0, 3], wires=0)
                qml.CNOT(wires=[num_qubits-1, 0])
            return [qml.expval(qml.PauliZ(j)) for j in range(nqd)]

        self.circuit      = circuit
        self.q_weights    = nn.Parameter(
            torch.empty(num_qubits, 4).uniform_(0, 2*math.pi))
        self.output_layer = nn.Linear(self.num_qubits_data, 1)

        n_q = self.q_weights.numel()
        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [QRNN] W:{window}  nf:{n_features}  "
              f"qubits:{num_qubits}(data:{nqd}+hidden:{num_qubits_hidden})  "
              f"Params:{n_p} (quantum:{n_q} + classical:{n_p-n_q})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (W, n_features)"""
        q_out = self.circuit(x.flatten(), self.q_weights)
        q_out = torch.stack(q_out) if isinstance(q_out, list) else q_out
        return self.output_layer(q_out.to(torch.float32)
                                 ).squeeze().to(dtype=torch.float32,
                                                device=DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  5. LSTMPredictor
# ══════════════════════════════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    def __init__(self, window: int, n_features: int,
                 hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm   = nn.LSTM(n_features, hidden_dim, num_layers,
                              batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [LSTM] nf:{n_features}  hidden:{hidden_dim}  "
              f"layers:{num_layers}  Params:{n_p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(0))
        return self.linear(out[0, -1, :]).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
#  6. TransformerPredictor
# ══════════════════════════════════════════════════════════════════════════════

class TransformerPredictor(nn.Module):
    def __init__(self, window: int, n_features: int,
                 d_model: int = 16, nhead: int = 2,
                 num_layers: int = 1, dim_ff: int = 32):
        super().__init__()
        d_model = max(d_model, nhead * math.ceil(n_features / nhead))
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)
        self.linear  = nn.Linear(d_model, 1)
        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [Transformer] d_model:{d_model}  nhead:{nhead}  "
              f"layers:{num_layers}  Params:{n_p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h   = self.input_proj(x).unsqueeze(0)
        out = self.encoder(h)
        return self.linear(out[0, -1, :]).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
#  7. MLPPredictor
# ══════════════════════════════════════════════════════════════════════════════

class MLPPredictor(nn.Module):
    def __init__(self, window: int, n_features: int, hidden_dim: int = 64):
        super().__init__()
        in_dim   = window * n_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        n_p = sum(p.numel() for p in self.parameters())
        print(f"  [MLP] in_dim:{in_dim}  hidden:{hidden_dim}  Params:{n_p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten()).squeeze()




# ══════════════════════════════════════════════════════════════════════════════
#  模型工厂
# ══════════════════════════════════════════════════════════════════════════════

def build_model(name: str, window: int, n_features: int,
                model_cfg: Optional[ModelConfig] = None) -> nn.Module:
    """
    Parameters
    ----------
    name        : 模型名称
    window      : 滑动窗口长度
    n_features  : 特征列数（不含 pos 列）
    model_cfg   : QTP 专用配置，其他模型传 None

    Returns
    -------
    nn.Module
    """
    if name == "QTP":
        assert model_cfg is not None, "QTP 需要传入 ModelConfig"
        return QuantumTemporalPredictor(model_cfg).to(DEVICE)
    
    elif name == "QTP_last":
        assert model_cfg is not None, "QTP 需要传入 ModelConfig"
        return QuantumTemporalPredictor_last(model_cfg).to(DEVICE)
    
    elif name == "VQC":
        return VQCPredictor(window, n_features,
                            num_qubits=4, num_layers=1).to(DEVICE)
    elif name == "QLSTM":
        return QLSTMPredictor(window, n_features,
                              num_qubits=4, hidden_size=4,
                              num_layers=1).to(DEVICE)
    elif name == "QRNN":
        return QRNNPredictor(window, n_features,
                             num_qubits=4,
                             num_qubits_hidden=2).to(DEVICE)
    elif name == "LSTM":
        return LSTMPredictor(window, n_features,
                             hidden_dim=32).to(DEVICE)
    elif name == "Transformer":
        return TransformerPredictor(window, n_features,
                                    d_model=16, nhead=2,
                                    num_layers=1, dim_ff=32).to(DEVICE)
    elif name == "MLP":
        return MLPPredictor(window, n_features,
                            hidden_dim=64).to(DEVICE)
   
    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: QTP/VQC/QLSTM/QRNN/LSTM/Transformer/MLP/ARIMAX"
        )