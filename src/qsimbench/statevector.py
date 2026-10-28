"""NumPy による状態ベクトルシミュレータ(CPU, complex128)。

各ゲートは tensordot で該当軸に作用させる。1ゲートごとに状態と同サイズの一時配列が
できるため、ピークメモリは理論状態サイズ(16·2^n バイト)の約2〜3倍になる。
"""
from __future__ import annotations

import cmath
import math

import numpy as np

from .circuits import Circuit, Gate

BYTES_PER_AMPLITUDE = 16  # complex128

_S2 = 1 / math.sqrt(2)
_FIXED = {
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "t": np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex),
    "cx": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "cz": np.diag([1, 1, 1, -1]).astype(complex),
    "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
}


def gate_matrix(gate: Gate) -> np.ndarray:
    """ゲートのユニタリ行列。2量子ビットゲートは gate.qubits[0] が上位ビット。"""
    name, p = gate.name, gate.params
    if name in _FIXED:
        return _FIXED[name]
    if name == "rx":
        c, s = math.cos(p[0] / 2), math.sin(p[0] / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        c, s = math.cos(p[0] / 2), math.sin(p[0] / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.diag([cmath.exp(-0.5j * p[0]), cmath.exp(0.5j * p[0])])
    if name == "p":
        return np.diag([1, cmath.exp(1j * p[0])])
    if name == "cp":
        return np.diag([1, 1, 1, cmath.exp(1j * p[0])])
    raise ValueError(f"unknown gate: {name!r}")


def statevector_bytes(n: int) -> int:
    """理論値: n 量子ビットの状態ベクトル1本のサイズ(バイト)。"""
    return BYTES_PER_AMPLITUDE * 2**n


def initial_state(n: int, basis_index: int = 0) -> np.ndarray:
    if not 0 <= basis_index < 2**n:
        raise ValueError("basis_index out of range")
    psi = np.zeros((2,) * n, dtype=complex)
    psi.reshape(-1)[basis_index] = 1.0
    return psi


def apply_gate(psi: np.ndarray, gate: Gate) -> np.ndarray:
    k = len(gate.qubits)
    u = gate_matrix(gate).reshape((2,) * (2 * k))
    out = np.tensordot(u, psi, axes=(list(range(k, 2 * k)), list(gate.qubits)))
    return np.moveaxis(out, list(range(k)), list(gate.qubits))


def simulate(circuit: Circuit, basis_index: int = 0) -> np.ndarray:
    """回路を実行し、長さ 2^n のフラットな状態ベクトルを返す。"""
    psi = initial_state(circuit.n_qubits, basis_index)
    for g in circuit.gates:
        psi = apply_gate(psi, g)
    return np.ascontiguousarray(psi).reshape(-1)


def probabilities(state: np.ndarray) -> np.ndarray:
    return np.abs(state) ** 2


def sample_counts(probs: np.ndarray, shots: int, seed: int) -> np.ndarray:
    """測定 shots 回の各基底状態のカウント(多項分布)。"""
    rng = np.random.default_rng(seed)
    p = probs / probs.sum()
    return rng.multinomial(shots, p)


def bitstring(index: int, n: int) -> str:
    """qubit 0 が左端のビット列。"""
    return format(index, f"0{n}b")
