import json
import statistics

import pandas as pd
import pytest

from qsimbench import jobs
from qsimbench.analysis import scaling_summary
from qsimbench.benchmark import process_resources, summarize
from qsimbench.jobs import ExperimentConfig, execute

SMALL = dict(name="a", qubits=[3, 4], shots=500, warmup=0, repeats=4, seed=7)


def test_summarize_matches_statistics_module():
    t = [1.0, 2.0, 4.0, 5.0]
    s = summarize(t)
    assert s["mean"] == 3.0 and s["min"] == 1.0 and s["max"] == 5.0 and s["median"] == 3.0
    assert s["std"] == pytest.approx(statistics.stdev(t))  # 標本標準偏差(n-1)
    assert s["cv"] == pytest.approx(s["std"] / 3.0)
    assert s["sem"] == pytest.approx(s["std"] / 2.0)


def test_summarize_single_sample_has_zero_std():
    assert summarize([0.5])["std"] == 0.0


def test_csv_contains_conditions_and_stats(tmp_path):
    rec = execute(ExperimentConfig(**SMALL), tmp_path)
    df = pd.read_csv(f"{rec.out_dir}/results.csv")
    for col in ("circuit", "seed", "shots", "warmup", "repeats", "qubits", "status",
                "time_mean_s", "time_std_s", "time_min_s", "time_max_s", "time_median_s",
                "peak_mem_bytes_measured", "statevector_bytes_theory"):
        assert col in df
    assert (df.seed == 7).all() and (df.repeats == 4).all() and (df.status == "ok").all()
    assert (df.time_min_s <= df.time_mean_s).all() and (df.time_mean_s <= df.time_max_s).all()


def test_seed_and_config_saved(tmp_path):
    rec = execute(ExperimentConfig(**SMALL), tmp_path)
    meta = json.loads(open(f"{rec.out_dir}/job.json", encoding="utf-8").read())
    assert meta["config"]["seed"] == 7 and meta["config"]["repeats"] == 4
    assert "tracemalloc" in meta["environment"]["memory_method"]
    assert {"python", "numpy", "platform"} <= set(meta["environment"])


def test_same_seed_same_sampling_result(tmp_path):
    cfg = ExperimentConfig(**{**SMALL, "circuit": "random"})
    a, b = execute(cfg, tmp_path), execute(cfg, tmp_path)
    assert [r["tvd_sampling_vs_sim"] for r in a.rows] == [r["tvd_sampling_vs_sim"] for r in b.rows]
    c = execute(ExperimentConfig(**{**SMALL, "circuit": "random", "seed": 8}), tmp_path)
    assert [r["tvd_sampling_vs_sim"] for r in a.rows] != [r["tvd_sampling_vs_sim"] for r in c.rows]


def test_run_ids_unique_even_within_same_second(tmp_path):
    ids = {execute(ExperimentConfig(**{**SMALL, "qubits": [2], "repeats": 1}), tmp_path).run_id for _ in range(5)}
    assert len(ids) == 5


def test_no_overwrite_existing_run_dir(tmp_path):
    rec = execute(ExperimentConfig(**SMALL), tmp_path)
    marker = open(f"{rec.out_dir}/results.csv", "rb").read()
    rec2 = jobs.save(rec, tmp_path)  # 同じ run_id で保存し直しても別ディレクトリへ
    assert str(rec2) != rec.out_dir and open(f"{rec.out_dir}/results.csv", "rb").read() == marker


def test_failure_midrun_records_error_and_failed_row(tmp_path, monkeypatch):
    real = jobs.run_point

    def flaky(cfg, n):
        if n == 4:
            raise RuntimeError("boom")
        return real(cfg, n)

    monkeypatch.setattr(jobs, "run_point", flaky)
    rec = execute(ExperimentConfig(**SMALL), tmp_path)
    assert rec.status == "failed" and rec.exit_code == 1 and "boom" in rec.error
    df = pd.read_csv(f"{rec.out_dir}/results.csv")
    assert list(df.status) == ["ok", "failed"] and "boom" in df.error.iloc[1]
    assert "FAILED" in open(f"{rec.out_dir}/run.log", encoding="utf-8").read()


def test_scaling_summary_ignores_failed_and_small_n():
    df = pd.DataFrame({
        "qubits": [4, 12, 13, 14], "status": ["ok", "ok", "ok", "failed"],
        "time_mean_s": [1e-4, 1.0, 2.0, None], "time_cv": [0.1, 0.02, 0.03, None],
        "peak_mem_bytes_measured": [10, 300, 600, None], "statevector_bytes_theory": [10, 100, 200, None],
    })
    s = scaling_summary(df)
    assert s["points_used"] == 2 and s["slope_log2_time_per_qubit"] == pytest.approx(1.0)
    assert s["measured_peak_over_theory_max_n"] == pytest.approx(3.0)
    assert scaling_summary(df, min_qubits=13)["slope_log2_time_per_qubit"] is None


def test_process_resources_shape():
    r = process_resources()
    assert r["cpu_time_s"] > 0
    assert r["peak_rss_bytes"] is None or r["peak_rss_bytes"] > 0
