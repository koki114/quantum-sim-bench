import json

import pandas as pd
import psutil
import pytest

from qsimbench import benchmark, jobqueue as jq, statevector as sv, sweep, threads as thr
from qsimbench.cli import main
from qsimbench.jobs import ExperimentConfig, config_to_toml

tp = pytest.importorskip("threadpoolctl")

TINY = ExperimentConfig(name="s", qubits=[3, 4], shots=200, warmup=0, repeats=2, measure_memory=False)


def blas_threads():
    return [d["num_threads"] for d in tp.threadpool_info() if d["user_api"] == "blas"]


# ---- 分類: 「検出対象なし」と「設定値の不一致」を区別する -------------------------
def test_classify_distinguishes_cases():
    blas = lambda n: [{"user_api": "blas", "internal_api": "openblas", "version": "1", "num_threads": n}]  # noqa: E731
    omp = [{"user_api": "openmp", "internal_api": "openmp", "version": None, "num_threads": 8}]
    assert thr.classify(2, None) == "threadpoolctl_unavailable"
    assert thr.classify(2, []) == "no_library_detected"
    assert thr.classify(2, omp) == "no_blas_detected"
    assert thr.classify(None, blas(8)) == "no_request"
    assert thr.classify(2, blas(2)) == "ok"
    assert thr.classify(2, blas(8)) == "mismatch"


def test_detect_reports_library_name_version_threads():
    d = thr.detect()
    assert d and {"internal_api", "version", "num_threads", "user_api"} <= set(d[0])
    assert any(x["user_api"] == "blas" and x["num_threads"] >= 1 for x in d)


def test_settings_apply_and_restore():
    original = blas_threads()
    with thr.ThreadSettings(1) as s:
        assert blas_threads() == [1] * len(original)
    assert s.info["status"] == "ok" and s.info["requested"] == 1
    assert blas_threads() == original  # 同一プロセスに設定を残さない


def test_settings_record_mismatch(monkeypatch):
    fake = [{"user_api": "blas", "internal_api": "openblas", "version": "1", "num_threads": 99,
             "threading_layer": None}]
    monkeypatch.setattr(thr, "detect", lambda: fake)
    s = thr.ThreadSettings(2)
    s.start()
    assert s.finish()["status"] == "mismatch"


def test_threadpoolctl_unavailable_gives_null_not_guess(monkeypatch):
    monkeypatch.setattr(thr, "threadpoolctl", None)
    s = thr.ThreadSettings(2)
    s.start()
    info = s.finish()
    assert info["applied_at_start"] is None and info["status"] == "threadpoolctl_unavailable"
    assert info["requested"] == 2 and info["threadpoolctl_version"] is None
    assert thr.blas_threads_summary(info) is None


def test_invalid_thread_request():
    with pytest.raises(ValueError):
        thr.ThreadSettings(0)


def test_environment_info_has_blas_and_env():
    env = benchmark.environment_info()
    assert env["blas"] and set(env["blas_env"]) == set(thr.ENV_KEYS)


# ---- 順序: スレッド設定は計算の前に適用されている --------------------------------
def test_threads_applied_before_compute(tmp_path, monkeypatch):
    jid = jq.submit(TINY, tmp_path)
    seen = []
    real = sv.simulate

    def spy(circ, basis_index=0):
        seen.append(blas_threads())  # 計算(simulate)が呼ばれた時点の実スレッド数
        return real(circ, basis_index)

    monkeypatch.setattr(sv, "simulate", spy)
    assert jq.worker_main(tmp_path / jid, threads=1) == 0
    assert seen and all(t == [1] * len(t) for t in seen)
    info = json.loads((tmp_path / jid / "worker_result.json").read_text(encoding="utf-8"))["thread_info"]
    assert info["status"] == "ok" and info["applied_at_start"] == info["applied_at_end"]


def test_worker_without_request_only_detects(tmp_path):
    jid = jq.submit(TINY, tmp_path)
    assert jq.worker_main(tmp_path / jid) == 0
    info = json.loads((tmp_path / jid / "worker_result.json").read_text(encoding="utf-8"))["thread_info"]
    assert info["requested"] is None and info["status"] == "no_request"


# ---- 子プロセスへの伝達と記録 -------------------------------------------------
def test_requested_env_and_applied_recorded_per_job(tmp_path):
    jid = jq.submit(TINY, tmp_path)
    jq.run_queue(tmp_path, workers=1, threads_per_job=2)
    j = jq.load_job(tmp_path, jid)
    ti = j["thread_info"]
    assert j["state"] == "succeeded" and j["threads_per_job"] == 2
    assert ti["requested"] == 2 and ti["env"]["OPENBLAS_NUM_THREADS"] == "2" and ti["env"]["OMP_NUM_THREADS"] == "2"
    assert ti["status"] == "ok"
    assert all(d["num_threads"] == 2 for d in ti["applied_at_start"] if d["user_api"] == "blas")
    df = jq.jobs_dataframe(tmp_path)
    assert df.thread_status.iloc[0] == "ok" and "openblas:2" in df.blas_threads_applied.iloc[0]


def test_workers_and_threads_independent(tmp_path):
    ids = [jq.submit(TINY, tmp_path) for _ in range(2)]
    jq.run_queue(tmp_path, workers=2, threads_per_job=1)  # workers=2 でも threads=1(連動しない)
    for i in ids:
        j = jq.load_job(tmp_path, i)
        assert j["workers"] == 2 and j["threads_per_job"] == 1 and j["thread_info"]["requested"] == 1


# ---- sweep の計画 ----------------------------------------------------------------
def test_plan_skips_oversubscribed_by_default():
    p = sweep.plan([1, 2, 4], [1, 2, 4], rounds=2, seed=0, logical_cores=8)
    assert (4, 4) not in p["conditions"] and len(p["conditions"]) == 8
    assert p["skipped"] == [{"workers": 4, "threads": 4, "reason": "workers*threads=16 > logical cores 8"}]
    assert (2, 4) in p["conditions"] and (4, 2) in p["conditions"]  # ちょうど8は許容


def test_plan_allow_oversubscribe():
    p = sweep.plan([4], [4], 1, 0, logical_cores=8, allow_oversubscribe=True)
    assert p["conditions"] == [(4, 4)] and p["oversubscribed"] == [(4, 4)] and not p["skipped"]


def test_plan_order_reproducible_and_shuffled_per_round():
    a = sweep.plan([1, 2, 4], [1, 2], 3, seed=7, logical_cores=8)
    b = sweep.plan([1, 2, 4], [1, 2], 3, seed=7, logical_cores=8)
    c = sweep.plan([1, 2, 4], [1, 2], 3, seed=8, logical_cores=8)
    assert a["orders"] == b["orders"] and a["orders"] != c["orders"]
    assert all(sorted(o) == sorted(a["conditions"]) for o in a["orders"])  # 全ラウンドで全条件を1回ずつ


def test_plan_validation():
    with pytest.raises(ValueError):
        sweep.plan([], [1], 1, 0, 8)
    with pytest.raises(ValueError):
        sweep.plan([1], [0], 1, 0, 8)
    with pytest.raises(ValueError):
        sweep.plan([4], [4], 1, 0, logical_cores=8)  # 全条件が過負荷 → 実行条件なし


# ---- sweep の実行と出力 ------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_sweep(tmp_path_factory):
    out_root = tmp_path_factory.mktemp("sw")
    out = sweep.run_sweep(TINY, [1, 2], [1], jobs_per_condition=2, rounds=2, seed=3, out_root=out_root)
    return out_root, out


def test_sweep_outputs_and_metadata(tiny_sweep):
    _, out = tiny_sweep
    meta = json.loads((out / "sweep.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished" and meta["sweep_seed"] == 3 and meta["config"]["seed"] == 0
    assert meta["environment"]["blas"] and meta["logical_cores"] >= 1 and len(meta["order_per_round"]) == 2
    assert (out / "config.toml").exists() and (out / "jobs").is_dir()
    r = pd.read_csv(out / "rounds.csv")
    assert len(r) == 4 and (r.failed == 0).all() and set(r.thread_status) == {"ok"}
    assert {"wall_s", "job_process_s_median", "job_in_execute_s_median", "circuit_nmax_s_median"} <= set(r)
    # 経過時間 ≥ ジョブ(プロセス込み) ≥ execute 内 ≥ 回路時間: 区別された量が整合している
    assert (r.job_in_execute_s_median <= r.job_process_s_median).all()
    assert (r.circuit_nmax_s_median <= r.job_in_execute_s_median).all()
    assert (r.job_process_s_median <= r.wall_s + 1e-6).all()
    assert set(r.blas_threads_applied) == {"openblas:1"}


def test_sweep_summary_has_no_optimum_claim_and_one_baseline(tiny_sweep):
    _, out = tiny_sweep
    s = pd.read_csv(out / "summary.csv")
    assert len(s) == 2 and s.baseline.sum() == 1 and s.rounds.eq(2).all()
    assert {"wall_s_median", "wall_s_min", "wall_s_max", "wall_s_std", "wall_s_cv",
            "wall_range_overlaps_baseline", "wall_speedup_median_vs_baseline"} <= set(s)
    assert not any("best" in c or "optimal" in c for c in s.columns)


def test_sweep_output_dir_unique_and_not_overwritten(tiny_sweep):
    out_root, out = tiny_sweep
    before = (out / "rounds.csv").read_bytes()
    out2 = sweep.run_sweep(TINY, [1], [1], jobs_per_condition=1, rounds=1, seed=0, out_root=out_root)
    assert out2 != out and (out / "rounds.csv").read_bytes() == before


def test_sweep_continues_after_condition_failure(tmp_path, monkeypatch):
    real = jq.run_queue

    def flaky(root, workers=1, threads_per_job=None, **kw):
        if workers == 2:
            raise RuntimeError("boom")
        return real(root, workers, threads_per_job, **kw)

    monkeypatch.setattr(jq, "run_queue", flaky)
    out = sweep.run_sweep(TINY, [1, 2], [1], jobs_per_condition=1, rounds=1, seed=0, out_root=tmp_path)
    r = pd.read_csv(out / "rounds.csv").sort_values("workers")
    assert r.failed.tolist() == [0, 1] and "boom" in r.error.iloc[1]
    assert json.loads((out / "sweep.json").read_text(encoding="utf-8"))["status"] == "finished"
    assert pd.read_csv(out / "summary.csv").failed_total.tolist() == [0, 1]


def test_summarize_when_everything_failed():
    df = pd.DataFrame([{"round": 1, "workers": 1, "threads": 1, "oversubscribed": False, "jobs": 1,
                        "failed": 1, "error": "x"}])
    assert sweep.summarize(df).failed_total.tolist() == [1]


def test_sweep_cli_and_old_commands_unchanged(tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text(config_to_toml(TINY), encoding="utf-8")
    assert main(["sweep", str(cfg), "--workers", "1", "--threads", "1", "--jobs-per-condition", "1",
                 "--rounds", "1", "--out", str(tmp_path / "sw")]) == 0
    assert "sweep written to" in capsys.readouterr().out
    # 既存コマンドの引数は変わらない(threads 省略時の既定は 物理コア数÷workers のまま)
    root = str(tmp_path / "jobs")
    assert main(["submit", str(cfg), "--root", root]) == 0
    assert main(["work", "--root", root, "--workers", "1"]) == 0
    assert jq.list_jobs(root)[0]["threads_per_job"] == max(1, (psutil.cpu_count(logical=False) or 1))
