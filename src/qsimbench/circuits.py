"""回路表現とビルダー。

回路は ``Gate`` のリスト。量子ビット q は状態ベクトルの軸 q に対応し、
基底状態のインデックスでは qubit 0 が最上位ビット(MSB)。
ビット列表記も qubit 0 が左端。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    gates: tuple[Gate, ...]

    def __len__(self) -> int:
        return len(self.gates)


def ghz(n: int) -> Circuit:
    """GHZ状態 (|0..0> + |1..1>)/√2。"""
    gates = [Gate("h", (0,))] + [Gate("cx", (i, i + 1)) for i in range(n - 1)]
    return Circuit(n, tuple(gates))


def qft(n: int) -> Circuit:
    """量子フーリエ変換(最後にビット反転SWAPを含む)。"""
    gates: list[Gate] = []
    for q in range(n):
        gates.append(Gate("h", (q,)))
        for m in range(q + 1, n):
            gates.append(Gate("cp", (m, q), (2 * math.pi / 2 ** (m - q + 1),)))
    for i in range(n // 2):
        gates.append(Gate("swap", (i, n - 1 - i)))
    return Circuit(n, tuple(gates))


def random_layered(n: int, depth: int, seed: int) -> Circuit:
    """RY/RZ の乱数角 + 隣接CX(偶奇交互)を depth 層重ねた回路。seed で再現可能。"""
    rng = np.random.default_rng(seed)
    gates: list[Gate] = []
    for layer in range(depth):
        for q in range(n):
            gates.append(Gate("ry", (q,), (float(rng.uniform(0, 2 * math.pi)),)))
            gates.append(Gate("rz", (q,), (float(rng.uniform(0, 2 * math.pi)),)))
        for q in range(layer % 2, n - 1, 2):
            gates.append(Gate("cx", (q, q + 1)))
    return Circuit(n, tuple(gates))


CIRCUIT_KINDS = ("ghz", "qft", "random")


def build(kind: str, n: int, depth: int = 4, seed: int = 0) -> Circuit:
    if n < 1:
        raise ValueError("qubits must be >= 1")
    if kind == "ghz":
        return ghz(n)
    if kind == "qft":
        return qft(n)
    if kind == "random":
        return random_layered(n, depth, seed)
    raise ValueError(f"unknown circuit kind: {kind!r} (choose from {CIRCUIT_KINDS})")
