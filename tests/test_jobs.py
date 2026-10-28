import json
import math

import pandas as pd
import pytest

from qsimbench import benchmark
from qsimbench.cli import main
from qsimbench.jobs import ExperimentConfig, config_to_toml, execute, load_config

SMALL = dict(name="t", qubits=[3, 5], shots=2000, warmup=1, repeats=3)


def test_measure_time_counts_and_excludes_warmup():
    calls = []
    r = benchmark.measure_time(lambda: calls.append(1), warmup=2, repeats=4)
    assert len(calls) == 6 and len(r["times"]) == 4
    assert r["min"] <= r["median"] <= max(r["times"])


def test_measure_peak_memory_sees_numpy_allocation():
    import numpy as np

    peak = benchmark.measure_peak_memory(lambda: np.ones(2**20, dtype=complex))
    assert peak >= 16 * 2**20


def test_execute_success_and_files(tmp_path):
    rec = execute(ExperimentConfig(**SMALL, circuit="qft", input_value=2), tmp_path)
    assert rec.status == "success" and rec.exit_code == 0
    df = pd.read_csv(f"{rec.out_dir}/results.csv")
    assert list(df.qubits) == [3, 5]
    assert (df.max_abs_error_vs_reference < 1e-10).all()
    assert (df.peak_mem_bytes_measured >= df.statevector_bytes_theory).all()
    assert len(json.loads(df.times_s[0])) == 3
    meta = json.loads(open(f"{rec.out_dir}/job.json", encoding="utf-8").read())
    assert meta["environment"]["gpu_used"] is False and meta["status"] == "success"
    assert "finish status=success" in open(f"{rec.out_dir}/run.log", encoding="utf-8").read()


def test_config_roundtrip_toml(tmp_path):
    cfg = ExperimentConfig(**SMALL, circuit="random", depth=2)
    p = tmp_path / "c.toml"
    p.write_text(config_to_toml(cfg), encoding="utf-8")
    assert load_config(p) == cfg


def test_rerun_same_config_same_results(tmp_path):
    cfg = ExperimentConfig(**SMALL, circuit="random")
    a, b = execute(cfg, tmp_path), execute(cfg, tmp_path)
    assert a.out_dir != b.out_dir  # 上書きしない
    assert a.config_digest == b.config_digest
    for ra, rb in zip(a.rows, b.rows):
        assert ra["tvd_sampling_vs_sim"] == rb["tvd_sampling_vs_sim"]


def test_failure_is_recorded(tmp_path):
    rec = execute(ExperimentConfig(name="bad", qubits=[3], backend="cuda-gpu"), tmp_path)
    assert rec.status == "failed" and rec.exit_code != 0
    assert "not implemented" in rec.error
    assert (tmp_path / rec.out_dir.split("\\")[-1].split("/")[-1] / "job.json").exists()


def test_qubit_limit_rejected(tmp_path):
    rec = execute(ExperimentConfig(name="big", qubits=[40]), tmp_path)
    assert rec.status == "failed" and not rec.rows


def test_unknown_config_key(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('nme = "typo"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_cli_batch_exit_codes(tmp_path, capsys):
    good = tmp_path / "good.toml"
    good.write_text(config_to_toml(ExperimentConfig(**SMALL)), encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text('backend = "gpu"\n', encoding="utf-8")
    out = tmp_path / "out"
    assert main(["run", str(good), "--out", str(out)]) == 0
    assert main(["run", str(good), str(bad), "--out", str(out)]) != 0
    assert len(list(out.iterdir())) == 2  # 成功ジョブのみ追加(不正設定は実行されない)


def test_tvd_close_to_theory_scale(tmp_path):
    rec = execute(ExperimentConfig(name="s", circuit="ghz", qubits=[6], shots=50_000, repeats=1), tmp_path)
    r = rec.rows[0]
    assert r["tvd_sampling_vs_sim"] < 5 * r["tvd_expected_scale_theory"] + 1e-3
    assert math.isclose(r["statevector_bytes_theory"], 16 * 64)
