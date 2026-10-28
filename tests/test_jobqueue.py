import json
from pathlib import Path

import pandas as pd
import pytest

from qsimbench import jobqueue as jq
from qsimbench.cli import main
from qsimbench.jobs import ExperimentConfig, config_to_toml

CFG = ExperimentConfig(name="q", qubits=[3, 5], shots=500, warmup=0, repeats=2)


def make_failing_job(root):
    """設定は有効なまま、出力先 runs/ をファイルにしてワーカー側の保存で必ず失敗するジョブを作る。
    (再実行は新しいジョブフォルダで行われるので、同じ障害は再現しない)"""
    jid = jq.submit(CFG, root)
    (Path(root) / jid / "runs").write_text("blocked", encoding="utf-8")
    return jid


def test_submit_creates_queued_job(tmp_path):
    jid = jq.submit(CFG, tmp_path)
    j = jq.load_job(tmp_path, jid)
    assert j["state"] == "queued" and j["submitted_at"] and j["started_at"] is None
    assert (tmp_path / jid / "config.toml").exists()


def test_job_ids_unique(tmp_path):
    assert len({jq.submit(CFG, tmp_path) for _ in range(20)}) == 20


def test_submit_rejects_invalid_config(tmp_path):
    with pytest.raises(ValueError):
        jq.submit(ExperimentConfig(qubits=[99]), tmp_path)
    assert jq.list_jobs(tmp_path) == []


def test_sequential_run_success_and_metadata(tmp_path):
    jid = jq.submit(CFG, tmp_path)
    s = jq.run_queue(tmp_path, workers=1)
    j = jq.load_job(tmp_path, jid)
    assert (s["jobs_run"], s["succeeded"], s["failed"]) == (1, 1, 0)
    assert j["state"] == "succeeded" and j["exit_code"] == 0
    assert j["started_at"] and j["finished_at"] and j["duration_s"] > 0
    assert j["duration_s"] >= j["worker_run_duration_s"]  # プロセス起動分だけ長い
    assert j["concurrency_at_start"] == 1 and j["workers"] == 1
    assert j["resources"]["cpu_time_s"] > 0
    assert (tmp_path / j["log_path"]).exists()
    assert (Path(j["run_dir"]) / "results.csv").exists() and (Path(j["run_dir"]) / "job.json").exists()
    assert list((tmp_path / "batches").glob("*.json"))


def test_parallel_run_all_succeed_and_overlap(tmp_path):
    ids = [jq.submit(CFG, tmp_path) for _ in range(4)]
    s = jq.run_queue(tmp_path, workers=2)
    jobs = [jq.load_job(tmp_path, i) for i in ids]
    assert s["succeeded"] == 4 and all(j["state"] == "succeeded" for j in jobs)
    assert max(j["concurrency_at_start"] for j in jobs) == 2  # 実際に同時実行された
    assert all(j["concurrency_at_start"] <= 2 for j in jobs)  # 上限を超えない


def test_workers_limit_validated(tmp_path):
    with pytest.raises(ValueError):
        jq.run_queue(tmp_path, workers=0)
    with pytest.raises(ValueError):
        jq.run_queue(tmp_path, workers=10_000)


def test_failure_recorded_and_others_still_run(tmp_path):
    bad = make_failing_job(tmp_path)
    good = jq.submit(CFG, tmp_path)
    s = jq.run_queue(tmp_path, workers=1)
    jb, jg = jq.load_job(tmp_path, bad), jq.load_job(tmp_path, good)
    assert jb["state"] == "failed" and jb["exit_code"] != 0 and jb["error"]
    assert jg["state"] == "succeeded"
    assert (s["failed"], s["succeeded"]) == (1, 1)


def test_fail_fast_leaves_rest_queued(tmp_path):
    bad = make_failing_job(tmp_path)
    later = jq.submit(CFG, tmp_path)
    s = jq.run_queue(tmp_path, workers=1, fail_fast=True)
    assert jq.load_job(tmp_path, bad)["state"] == "failed"
    assert jq.load_job(tmp_path, later)["state"] == "queued" and s["left_queued"] == 1


def test_retry_runs_as_new_job_and_keeps_old(tmp_path):
    bad = make_failing_job(tmp_path)
    jq.run_queue(tmp_path)
    before = (tmp_path / bad / "job.json").read_bytes()
    new = jq.retry(tmp_path, bad)
    assert new != bad and jq.load_job(tmp_path, new)["retry_of"] == bad
    jq.run_queue(tmp_path)
    assert jq.load_job(tmp_path, new)["state"] == "succeeded"
    assert (tmp_path / bad / "job.json").read_bytes() == before  # 元の失敗記録は不変
    assert jq.load_job(tmp_path, bad)["state"] == "failed"


def test_retry_only_failed(tmp_path):
    jid = jq.submit(CFG, tmp_path)
    with pytest.raises(ValueError):
        jq.retry(tmp_path, jid)  # queued は再実行不可
    jq.run_queue(tmp_path)
    with pytest.raises(ValueError):
        jq.retry(tmp_path, jid)  # succeeded も不可


def test_cancel_queued_only(tmp_path):
    a, b = jq.submit(CFG, tmp_path), jq.submit(CFG, tmp_path)
    jq.cancel(tmp_path, a)
    s = jq.run_queue(tmp_path)
    assert jq.load_job(tmp_path, a)["state"] == "cancelled" and s["jobs_run"] == 1
    with pytest.raises(ValueError):
        jq.cancel(tmp_path, b)  # 完了済みは取り消せない


def test_invalid_transition_rejected(tmp_path):
    jid = jq.submit(CFG, tmp_path)
    with pytest.raises(ValueError):
        jq._update(tmp_path, jq.load_job(tmp_path, jid), "succeeded")  # queued → succeeded は不可


def test_worker_crash_without_result_is_failed(tmp_path):
    jid = jq.submit(CFG, tmp_path)
    job = jq._update(tmp_path, jq.load_job(tmp_path, jid), "running", started_at="x")
    (tmp_path / jid / "worker.log").write_text("Killed\n", encoding="utf-8")
    out = jq._finalize(tmp_path, job, returncode=137, wall=1.0)
    assert out["state"] == "failed" and out["exit_code"] == 137 and "Killed" in out["error"]


def test_export_csv_and_no_overwrite(tmp_path):
    jq.submit(CFG, tmp_path)
    jq.run_queue(tmp_path)
    jobs_csv, res_csv = tmp_path / "jobs.csv", tmp_path / "res.csv"
    jq.export_csv(jq.jobs_dataframe(tmp_path), jobs_csv)
    jq.export_csv(jq.results_dataframe(tmp_path), res_csv)
    jdf, rdf = pd.read_csv(jobs_csv), pd.read_csv(res_csv)
    assert jdf.state.iloc[0] == "succeeded" and jdf.peak_rss_bytes.iloc[0] > 0
    assert list(rdf.qubits) == [3, 5] and rdf.job_id.nunique() == 1
    with pytest.raises(FileExistsError):
        jq.export_csv(jdf, jobs_csv)


def test_cli_roundtrip(tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text(config_to_toml(CFG), encoding="utf-8")
    root = str(tmp_path / "jobs")
    assert main(["submit", str(cfg), "--root", root]) == 0
    assert main(["work", "--root", root]) == 0
    assert main(["list", "--root", root]) == 0
    out = capsys.readouterr().out
    assert "succeeded" in out
    jid = jq.list_jobs(root)[0]["job_id"]
    assert main(["show", jid, "--root", root]) == 0
    assert main(["show", "nope", "--root", root]) == 2
    assert main(["export", "--jobs", str(tmp_path / "j.csv"), "--root", root]) == 0
    assert main(["export", "--jobs", str(tmp_path / "j.csv"), "--root", root]) == 2  # 上書き拒否


def test_cli_failure_exit_code_and_retry(tmp_path):
    root = str(tmp_path / "jobs")
    bad = make_failing_job(root)
    assert main(["work", "--root", root]) == 1
    assert main(["retry", "--root", root]) == 0  # 未再実行の失敗ジョブを自動で選ぶ
    assert main(["retry", "--root", root]) == 0  # 既に再登録済みなので何もしない
    assert sum(j["retry_of"] == bad for j in jq.list_jobs(root)) == 1


def test_legacy_run_command_still_works(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text(config_to_toml(CFG), encoding="utf-8")
    assert main(["run", str(cfg), "--out", str(tmp_path / "out")]) == 0
    assert json.loads(next((tmp_path / "out").glob("*/job.json")).read_text(encoding="utf-8"))["status"] == "success"
