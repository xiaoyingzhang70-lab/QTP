# -*- coding: utf-8 -*-
"""
noise_model_overrides.py
=======================
Noise-only model overrides used by dedicated noise experiments.

This file does NOT modify the original training model in Model.py.
Instead, it provides a compatible subclass with a stricter gate-noise
injection rule for inference:

- single-qubit gate  -> apply noise to that wire
- two-qubit gate     -> apply noise to BOTH participating wires

This is intended for controlled gate-noise analysis only.
"""

from __future__ import annotations

import torch
import pennylane as qml

from Model import QuantumTemporalPredictor


class SymmetricGateNoiseQTP(QuantumTemporalPredictor):
    """
    QTP variant used only for noise studies.

    It keeps the original parameterization and clean forward pass intact,
    but overrides the noisy circuit so that two-qubit gates inject channel
    noise symmetrically on both involved wires.
    """

    def _build_noisy_circuit(self, shots=None, meas_seed=None):
        noisy_dev = qml.device(
            "default.mixed",
            wires=self._wires,
            shots=shots,
            seed=meas_seed,
        )

        @qml.qnode(noisy_dev, interface="torch", diff_method="best")
        def circuit(x, w, noise_type=None, noise_p=0.0):
            def apply_noise_1q(wire):
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

            def apply_noise_2q(ctrl, tgt):
                apply_noise_1q(ctrl)
                apply_noise_1q(tgt)

            zero = torch.zeros(1, dtype=x.dtype)

            def noisy_rz(angle, wire):
                qml.Rot(angle, zero, zero, wires=wire)
                apply_noise_1q(wire)

            def noisy_crz(angle, ctrl, tgt):
                qml.CRot(angle, zero, zero, wires=[ctrl, tgt])
                apply_noise_2q(ctrl, tgt)

            cfg = self.cfg
            tq, bq, fq = self.tq, self.bq, self.fq

            for t in range(cfg.windows):
                qml.RY(x[t, 0], wires=tq)
                apply_noise_1q(tq)

                for f in range(cfg.n_features):
                    qml.CRY(w["time_decay"][t] * x[t, f + 1], wires=[tq, fq[f]])
                    apply_noise_2q(tq, fq[f])

                for f in range(cfg.n_features):
                    noisy_crz(w[f"enc_crz_{t}_{f}"], tq, fq[f])

                if t % cfg.K == 0:
                    for f in range(cfg.n_features):
                        qml.CRY(w[f"snap_cry_{t}_{f}"], wires=[fq[f], bq])
                        apply_noise_2q(fq[f], bq)
                        noisy_crz(w[f"snap_crz_{t}_{f}"], fq[f], bq)

            for layer in range(cfg.layers):
                for f in range(cfg.n_features):
                    qml.RY(w[f"amp_{layer}_{f}"], wires=fq[f])
                    apply_noise_1q(fq[f])

                for f in range(0, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_even_{layer}_{f}"], wires=[fq[f], fq[f + 1]])
                    apply_noise_2q(fq[f], fq[f + 1])

                for f in range(1, cfg.n_features - 1, 2):
                    qml.CRY(w[f"ent_odd_{layer}_{f}"], wires=[fq[f], fq[f + 1]])
                    apply_noise_2q(fq[f], fq[f + 1])

                for f in range(cfg.n_features):
                    qml.CRY(w[f"bas_cry_{layer}_{f}"], wires=[bq, fq[f]])
                    apply_noise_2q(bq, fq[f])
                    noisy_crz(w[f"bas_crz_{layer}_{f}"], bq, fq[f])

                for f in range(cfg.n_features):
                    noisy_rz(w[f"net_rz_{layer}_{f}"], fq[f])

            return [qml.expval(qml.PauliZ(wire)) for wire in fq]

        return circuit
