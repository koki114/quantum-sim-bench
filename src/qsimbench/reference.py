"""正当性検証用の参照計算。シミュレータ本体(tensordot)とは別の方法で求める。

- ``dense_reference``: 2^n×2^n の全体ユニタリを基底状態ごとの添字演算で組み立てる(n ≤ DENSE_MAX_QUBITS)。
- ``analytic_reference``: GHZ / QFT(基底状態入力)の閉形式解。
"""
from __future__ import annotations

import numpy as np

from .circuits import Circuit
from .statevector import gate_matrix

DENSE_MAX_QUBITS = 9


def _embed(mat: np.ndarray, qubits: tuple[int, ...], n: int) -> np.ndarray:
    big = 2**n
    k = len(qubits)
    shifts = [n - 1 - q for q in qubits]
    mask = sum(1 << s for s in shifts)
    full = np.zeros((big, big), dtype=complex)
    for col in range(big):
        sub_in = sum(((col >> s) & 1) << (k - 1 - i) for i, s in enumerate(shifts))
        base = col & ~mask
        for sub_out in range(2**k):
            idx = base
            for i, s in enumerate(shifts):
                idx |= ((sub_out >> (k - 1 - i)) & 1) << s
            full[idx, col] += mat[sub_out, sub_in]
    return full


def dense_reference(circuit: Circuit, basis_index: int = 0) -> np.ndarray:
    n = circuit.n_qubits
    if n > DENSE_MAX_QUBITS:
        raise ValueError(f"dense reference supports n <= {DENSE_MAX_QUBITS}")
    psi = np.zeros(2**n, dtype=complex)
    psi[basis_index] = 1.0
    for g in circuit.gates:
        psi = _embed(gate_matrix(g), g.qubits, n) @ psi
    return psi


def analytic_reference(kind: str, n: int, basis_index: int = 0) -> np.ndarray | None:
    """閉形式の状態ベクトル。該当しない場合は None。"""
    big = 2**n
    if kind == "ghz" and basis_index == 0:
        psi = np.zeros(big, dtype=complex)
        psi[0] = psi[-1] = 2**-0.5
        return psi
    if kind == "qft":
        j = np.arange(big)
        return np.exp(2j * np.pi * j * basis_index / big) / np.sqrt(big)
    return None


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))
