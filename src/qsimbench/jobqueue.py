"""ローカルCPU用のジョブキュー(ファイルベース)。

``root/<job_id>/`` に job.json・config.toml・worker.log・worker_result.json・runs/ を置く。
各ジョブは子プロセス ``python -m qsimbench worker <job_dir>`` として実行され、
実際の計算は既存の ``jobs.execute`` が行う(ここではシミュレータに触れない)。
job.json を書くのはスケジューラ(``run_queue``)と ``cancel`` だけ。

状態: queued → running → succeeded | failed、queued → cancelled。
実行中ジョブの中断は実装していない(running の cancelled は存在しない)。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

from . import statevector as sv
from . import threads as thr
from .benchmark import process_resources
from .jobs import ExperimentConfig, config_to_toml, execute, load_config

STATES = ("queued", "running", "succeeded", "failed", "cancelled")
TRANSITIONS = {"queued": {"running", "cancelled"}, "running": {"succeeded", "failed"}}
MEM_FRACTION = 0.6  # 並列起動時、推定ピークの合計が利用可能RAMのこの割合を超えないようにする
PEAK_FACTOR = 3  # 理論状態サイズに対する推定ピーク倍率(README の実測 約3倍 に基づく経験則)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _job_dir(root: Path, job_id: str) -> Path:
    return Path(root) / job_id


def load_job(root: str | Path, job_id: str) -> dict:
    p = _job_dir(Path(root), job_id) / "job.json"
    if not p.exists():
        raise KeyError(f"job not found: {job_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def list_jobs(root: str | Path) -> list[dict]:
    root = Path(root)
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("*/job.json")] if root.exists() else []
    return sorted(jobs, key=lambda j: (j["seq"], j["job_id"]))  # seq = 登録順(同一秒でも順序を保つ)


def _update(root: Path, job: dict, new_state: str | None = None, **fields) -> dict:
    if new_state is not None:
        if new_state not in TRANSITIONS.get(job["state"], ()):
            raise ValueError(f"invalid transition {job['state']} -> {new_state} for {job['job_id']}")
        job["state"] = new_state
    job.update(fields)
    _write_json(_job_dir(root, job["job_id"]) / "job.json", job)
    return job


def submit(cfg: ExperimentConfig | str | Path, root: str | Path = "jobs", retry_of: str | None = None) -> str:
    """設定を queued ジョブとして登録し job_id を返す。"""
    if not isinstance(cfg, ExperimentConfig):
        cfg = load_config(cfg)
    cfg.validate()
    root = Path(root)
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    d = _job_dir(root, job_id)
    seq = max((j["seq"] for j in list_jobs(root)), default=0) + 1
    d.mkdir(parents=True)  # 既存なら例外(上書きしない)
    (d / "config.toml").write_text(config_to_toml(cfg), encoding="utf-8")
    _write_json(d / "job.json", {
        "job_id": job_id, "seq": seq, "state": "queued", "name": cfg.name, "circuit": cfg.circuit,
        "qubits": cfg.qubits, "shots": cfg.shots, "seed": cfg.seed,
        "config_digest": cfg.digest(), "retry_of": retry_of,
        "submitted_at": _now(), "started_at": None, "finished_at": None,
        "exit_code": None, "duration_s": None, "worker_run_duration_s": None,
        "log_path": f"{job_id}/worker.log", "run_dir": None, "error": None,
        "workers": None, "threads_per_job": None, "concurrency_at_start": None,
        "resources": None, "thread_info": None,
    })
    return job_id


def cancel(root: str | Path, job_id: str) -> dict:
    """queued ジョブのみ取り消せる。実行中ジョブの中断は未実装なので拒否する。"""
    root = Path(root)
    job = load_job(root, job_id)
    if job["state"] != "queued":
        raise ValueError(f"cannot cancel job in state {job['state']!r} (only queued jobs; running jobs cannot be stopped)")
    return _update(root, job, "cancelled", finished_at=_now())


def retry(root: str | Path, job_id: str) -> str:
    """failed ジョブの設定から新しい queued ジョブを作る。元のジョブ・結果は変更しない。"""
    root = Path(root)
    job = load_job(root, job_id)
    if job["state"] != "failed":
        raise ValueError(f"only failed jobs can be retried (state={job['state']!r})")
    return submit(_job_dir(root, job_id) / "config.toml", root, retry_of=job_id)


def _estimated_peak_bytes(job: dict) -> int:
    return PEAK_FACTOR * sv.statevector_bytes(max(job["qubits"]))


def _spawn(root: Path, job: dict, threads: int) -> subprocess.Popen:
    d = _job_dir(root, job["job_id"])
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(threads)
    pkg_parent = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
    log = open(d / "worker.log", "ab")
    try:
        return subprocess.Popen([sys.executable, "-m", "qsimbench", "worker", str(d), "--threads", str(threads)],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
    finally:
        log.close()  # 子プロセス側のハンドルは継承済み


def _finalize(root: Path, job: dict, returncode: int, wall: float) -> dict:
    d = _job_dir(root, job["job_id"])
    res_path = d / "worker_result.json"
    if res_path.exists():
        res = json.loads(res_path.read_text(encoding="utf-8"))
        ok = res["exit_code"] == 0 and returncode == 0
        err = res.get("error")
        res_fields = dict(exit_code=res["exit_code"], worker_run_duration_s=res.get("run_duration_s"),
                          run_dir=res.get("run_dir"), thread_info=res.get("thread_info"), resources=_with_util(res.get("resources"), res.get("process_wall_s")))
    else:  # ワーカーが結果を書けずに終了(クラッシュ・強制終了・OOM kill など)
        tail = ""
        lp = d / "worker.log"
        if lp.exists():
            tail = "\n".join(lp.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:])
        ok, err = False, f"worker exited with code {returncode} without a result file. log tail:\n{tail}"
        res_fields = dict(exit_code=returncode, resources=None)
    return _update(root, job, "succeeded" if ok else "failed", finished_at=_now(),
                   duration_s=wall, error=None if ok else (err or "failed"), **res_fields)


def _with_util(resources: dict | None, wall: float | None) -> dict | None:
    if not resources:
        return None
    out = dict(resources)
    cpu = resources.get("cpu_time_s")
    # 平均CPU使用コア数(プロセス生存期間の平均)。瞬間値ではない。
    out["cpu_util_cores_avg"] = cpu / wall if cpu is not None and wall else None
    return out


def run_queue(root: str | Path = "jobs", workers: int = 1, threads_per_job: int | None = None,
              fail_fast: bool = False, poll_s: float = 0.1) -> dict:
    """queued ジョブを最大 workers 個まで同時に実行し、バッチ要約を返す(batches/ にも保存)。

    fail_fast=False(既定): 失敗しても残りを実行する。True: 失敗後は新規起動せず、残りは queued のまま。
    """
    root = Path(root).resolve()
    logical = os.cpu_count() or 1
    if not 1 <= workers <= logical:
        raise ValueError(f"workers must be 1..{logical} (logical cores)")
    physical = psutil.cpu_count(logical=False) or logical
    threads = threads_per_job or max(1, physical // workers)

    pending = [j for j in list_jobs(root) if j["state"] == "queued"]
    running: dict[str, tuple[subprocess.Popen, float, int]] = {}
    t_batch = time.perf_counter()
    started_at = _now()
    done: list[dict] = []
    stop = False
    while (pending and not stop) or running:
        while pending and not stop and len(running) < workers:
            job = load_job(root, pending[0]["job_id"])  # cancel 反映のため再読込
            pending.pop(0)
            if job["state"] != "queued":
                continue
            est = _estimated_peak_bytes(job)
            in_flight = sum(e for _, _, e in running.values())
            if running and in_flight + est > MEM_FRACTION * psutil.virtual_memory().available:
                pending.insert(0, job)  # メモリ不足の見込み: 実行中が終わるまで待つ
                break
            job = _update(root, job, "running", started_at=_now(), workers=workers,
                          threads_per_job=threads, concurrency_at_start=len(running) + 1)
            running[job["job_id"]] = (_spawn(root, job, threads), time.perf_counter(), est)
        for jid in list(running):
            proc, t0, _ = running[jid]
            rc = proc.poll()
            if rc is not None:
                done.append(_finalize(root, load_job(root, jid), rc, time.perf_counter() - t0))
                del running[jid]
                if fail_fast and done[-1]["state"] == "failed":
                    stop = True
        if running:
            time.sleep(poll_s)
    wall = time.perf_counter() - t_batch
    summary = {
        "batch_id": f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        "started_at": started_at, "workers": workers, "threads_per_job": threads,
        "jobs_run": len(done), "succeeded": sum(j["state"] == "succeeded" for j in done),
        "failed": sum(j["state"] == "failed" for j in done),
        "left_queued": len(pending), "wall_time_s": wall,
        "sum_job_durations_s": sum(j["duration_s"] for j in done),
        "job_ids": [j["job_id"] for j in done],
    }
    (root / "batches").mkdir(exist_ok=True)
    _write_json(root / "batches" / f"{summary['batch_id']}.json", summary)
    return summary


def worker_main(job_dir: str | Path, threads: int | None = None) -> int:
    """子プロセス側: スレッド設定を適用してから、config.toml を既存の execute で実行する。

    順序: (1) スレッド適用+検出 → (2) execute(計算) → (3) 再検出して変化の有無を記録。
    """
    d = Path(job_dir).resolve()
    t0 = time.perf_counter()
    result: dict = {"exit_code": 2, "error": None, "run_dir": None, "run_duration_s": None}
    settings = thr.ThreadSettings(threads)
    try:
        settings.start()
        cfg = load_config(d / "config.toml")
        rec = execute(cfg, d / "runs")
        result.update(exit_code=rec.exit_code, error=rec.error, run_dir=rec.out_dir,
                      run_duration_s=rec.duration_s)
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    result["thread_info"] = settings.finish()
    result["process_wall_s"] = time.perf_counter() - t0
    result["resources"] = process_resources()
    _write_json(d / "worker_result.json", result)
    return int(result["exit_code"])


def jobs_dataframe(root: str | Path) -> pd.DataFrame:
    rows = []
    for j in list_jobs(root):
        r = j.get("resources") or {}
        rows.append({
            "job_id": j["job_id"], "state": j["state"], "retry_of": j["retry_of"], "name": j["name"],
            "circuit": j["circuit"], "qubits": " ".join(map(str, j["qubits"])), "shots": j["shots"],
            "seed": j["seed"], "config_digest": j["config_digest"],
            "submitted_at": j["submitted_at"], "started_at": j["started_at"], "finished_at": j["finished_at"],
            "exit_code": j["exit_code"], "duration_s": j["duration_s"],
            "worker_run_duration_s": j["worker_run_duration_s"],
            "workers": j["workers"], "threads_per_job": j["threads_per_job"],
            "concurrency_at_start": j["concurrency_at_start"],
            "peak_rss_bytes": r.get("peak_rss_bytes"), "cpu_time_s": r.get("cpu_time_s"),
            "cpu_util_cores_avg": r.get("cpu_util_cores_avg"),
            "log_path": j["log_path"],
            "blas_threads_applied": thr.blas_threads_summary(j.get("thread_info")),
            "thread_status": (j.get("thread_info") or {}).get("status"),
            "error": (j["error"] or "").strip().splitlines()[-1] if j["error"] else None,
        })
    return pd.DataFrame(rows)


def results_dataframe(root: str | Path) -> pd.DataFrame:
    """全ジョブの results.csv を job_id 付きで連結する。"""
    frames = []
    for j in list_jobs(root):
        if j.get("run_dir") and (Path(j["run_dir"]) / "results.csv").exists():
            df = pd.read_csv(Path(j["run_dir"]) / "results.csv")
            df.insert(0, "job_id", j["job_id"])
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def export_csv(df: pd.DataFrame, out: str | Path) -> Path:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists (refusing to overwrite)")
    df.to_csv(out, index=False)
    return out
