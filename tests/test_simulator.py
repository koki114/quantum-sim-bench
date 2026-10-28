import numpy as np
import pytest

from qsimbench import circuits, reference, statevector as sv
from qsimbench.circuits import Circuit, Gate


def test_bell_state():
    c = Circuit(2, (Gate("h", (0,)), Gate("cx", (0, 1))))
    p = sv.probabilities(sv.simulate(c))
    assert np.allclose(p, [0.5, 0, 0, 0.5])


def test_cx_control_target_order():
    # qubit0 を 1 にして cx(0,1) → |11>、cx(1,0) では変化しない
    x0 = Gate("x", (0,))
    assert sv.probabilities(sv.simulate(Circuit(2, (x0, Gate("cx", (0, 1))))))[0b11] == pytest.approx(1)
    assert sv.probabilities(sv.simulate(Circuit(2, (x0, Gate("cx", (1, 0))))))[0b10] == pytest.approx(1)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_ghz_analytic(n):
    state = sv.simulate(circuits.ghz(n))
    assert reference.max_abs_error(state, reference.analytic_reference("ghz", n)) < 1e-12


@pytest.mark.parametrize("n,k", [(2, 1), (3, 5), (5, 11), (8, 200)])
def test_qft_analytic(n, k):
    state = sv.simulate(circuits.qft(n), k)
    assert reference.max_abs_error(state, reference.analytic_reference("qft", n, k)) < 1e-10


@pytest.mark.parametrize("n,depth,seed", [(2, 3, 0), (4, 4, 1), (6, 5, 2), (7, 3, 3)])
def test_random_matches_dense_reference(n, depth, seed):
    c = circuits.random_layered(n, depth, seed)
    err = reference.max_abs_error(sv.simulate(c), reference.dense_reference(c))
    assert err < 1e-12


def test_dense_reference_nonadjacent_reversed_qubits():
    c = Circuit(4, (Gate("h", (3,)), Gate("cx", (3, 0)), Gate("cp", (2, 0), (0.7,)), Gate("swap", (1, 3))))
    assert reference.max_abs_error(sv.simulate(c), reference.dense_reference(c)) < 1e-12


def test_norm_preserved_large_random():
    p = sv.probabilities(sv.simulate(circuits.random_layered(12, 6, 5)))
    assert abs(p.sum() - 1) < 1e-12


def test_random_circuit_reproducible():
    assert circuits.random_layered(5, 3, 7) == circuits.random_layered(5, 3, 7)
    assert circuits.random_layered(5, 3, 7) != circuits.random_layered(5, 3, 8)


def test_against_qiskit():
    """独立実装(Qiskit)との照合。qubit q ↔ Qiskit qubit n-1-q で添字規約を合わせる。"""
    qk = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    for c in (circuits.random_layered(5, 4, 11), circuits.qft(4), circuits.ghz(5)):
        n = c.n_qubits
        qc = qk.QuantumCircuit(n)
        for g in c.gates:
            qs = [n - 1 - q for q in g.qubits]
            getattr(qc, g.name)(*g.params, *qs)
        expected = Statevector(qc).data
        assert reference.max_abs_error(sv.simulate(c), expected) < 1e-12


def test_sampling_statistics():
    probs = np.array([0.5, 0.0, 0.0, 0.5])
    counts = sv.sample_counts(probs, 100_000, seed=1)
    assert counts.sum() == 100_000 and counts[1] == counts[2] == 0
    assert reference.total_variation_distance(counts / 100_000, probs) < 0.01


def test_sampling_reproducible():
    p = np.full(8, 1 / 8)
    assert (sv.sample_counts(p, 1000, 3) == sv.sample_counts(p, 1000, 3)).all()


def test_theoretical_memory():
    assert sv.statevector_bytes(10) == 16 * 1024
    assert sv.statevector_bytes(30) == 16 * 2**30


def test_bad_inputs():
    with pytest.raises(ValueError):
        circuits.build("nope", 3)
    with pytest.raises(ValueError):
        circuits.build("ghz", 0)
    with pytest.raises(ValueError):
        sv.initial_state(2, 4)
