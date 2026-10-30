"""workers(同時実行ジョブ数)と BLAS スレッド数を独立に振る比較実験。

条件の格子 = workers × threads。各ラウンドで全条件を seed 付きでシャッフルした順に実行し、
時間経過(発熱・他プロセス)の影響が特定条件に偏らないようにする。
結果は ``out/<日時>-<hex>/`` に新規作成する(既存は上書きしない):
  sweep.json   計画・seed・実行順・スキップ条件・環境・設定
  rounds.csv   条件×ラウンドごとの1行(経過時間とジョブ内時間を別列で記録)
  summary.csv  条件ごとの集計(中央値・最小・最大・標準偏差・変動係数)。「最適」判定は行わない
  jobs/        各ジョブの記録(通常のジョブキューと同じ形式)
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil

from . import jobqueue as jq
from . import threads as thr
from .benchmark import environment_info
from .jobs import ExperimentConfig, config_to_toml, load_config


def _stats(xs: list[float], prefix: str) -> dict:
    xs = [x for x in xs if x is not None and x == x]
    if not xs:
        return {f"{prefix}_{k}": None for k in ("median", "min", "max", "std")}
    return {f"{prefix}_median": statistics.median(xs), f"{prefix}_min": min(xs), f"{prefix}_max": max(xs),
            f"{prefix}_std": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def plan(workers: list[int], threads: list[int], rounds: int, seed: int, logical_cores: int,
         allow_oversubscribe: bool = False) -> dict:
    """条件の格子、スキップ条件、各ラウンドの実行順を返す(seed で再現可能)。"""
    if rounds < 1 or not workers or not threads or min(workers) < 1 or min(threads) < 1:
        raise ValueError("workers/threads must be >= 1 and rounds >= 1")
    grid = [(w, t) for w in sorted(set(workers)) for t in sorted(set(threads))]
    over = lambda w, t: w * t > logical_cores  # noqa: E731
    run = [c for c in grid if allow_oversubscribe or not over(*c)]
    skipped = [{"workers": w, "threads": t, "reason": f"workers*threads={w * t} > logical cores {logical_cores}"}
               for w, t in grid if not allow_oversubscribe and over(w, t)]
    if not run:
        raise ValueError("no runnable conditions (all exceed logical cores; use --allow-oversubscribe)")
    orders = []
    for r in range(rounds):
        o = list(run)
        random.Random(seed * 1000 + r).shuffle(o)
        orders.append(o)
    return {"conditions": run, "skipped": skipped, "orders": orders,
            "oversubscribed": [c for c in run if over(*c)]}


def run_sweep(cfg: ExperimentConfig | str | Path, workers: list[int], threads: list[int],
              jobs_per_condition: int = 4, rounds: int = 3, seed: int = 0, out_root: str | Path = "sweeps",
              allow_oversubscribe: bool = False, logical_cores: int | None = None) -> Path:
    if not isinstance(cfg, ExperimentConfig):
        cfg = load_config(cfg)
    cfg.validate()
    if jobs_per_condition < 1:
        raise ValueError("jobs_per_condition must be >= 1")
    logical = logical_cores or os.cpu_count() or 1
    pl = plan(workers, threads, rounds, seed, logical, allow_oversubscribe)

    out = Path(out_root) / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True)  # 既存なら例外(上書きしない)
    jobs_root = out / "jobs"
    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"), "status": "running",
        "config": asdict(cfg), "config_digest": cfg.digest(), "sweep_seed": seed,
        "jobs_per_condition": jobs_per_condition, "rounds": rounds,
        "logical_cores": logical, "physical_cores": psutil.cpu_count(logical=False),
        "conditions": [{"workers": w, "threads": t} for w, t in pl["conditions"]],
        "skipped_conditions": pl["skipped"],
        "oversubscribed_conditions": [{"workers": w, "threads": t} for w, t in pl["oversubscribed"]],
        "order_per_round": [[{"workers": w, "threads": t} for w, t in o] for o in pl["orders"]],
        "environment": environment_info(),
        "note": "workers*threads の制限は今回の測定における保守的な設定であり、超過が常に遅いことを意味しない",
    }
    (out / "sweep.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "config.toml").write_text(config_to_toml(cfg), encoding="utf-8")

    rows: list[dict] = []
    t_all = time.perf_counter()
    for r, order in enumerate(pl["orders"], start=1):
        for idx, (w, t) in enumerate(order, start=1):
            row = {"round": r, "order_index": idx, "workers": w, "threads": t,
                   "oversubscribed": (w, t) in pl["oversubscribed"], "jobs": jobs_per_condition}
            try:
                ids = [jq.submit(cfg, jobs_root) for _ in range(jobs_per_condition)]
                s = jq.run_queue(jobs_root, workers=w, threads_per_job=t)
                jobs = [jq.load_job(jobs_root, i) for i in ids]
                row.update(_condition_row(jobs_root, jobs, s, cfg))
            except Exception as e:  # noqa: BLE001 - 1条件の失敗で sweep 全体を止めない
                row.update(failed=jobs_per_condition, error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc().strip().splitlines()[-1])
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "rounds.csv", index=False)  # 途中で中断しても残す
    meta["status"] = "finished"
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["total_wall_s"] = time.perf_counter() - t_all
    (out / "sweep.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    summarize(pd.DataFrame(rows)).to_csv(out / "summary.csv", index=False)
    return out


def _condition_row(jobs_root: Path, jobs: list[dict], s: dict, cfg: ExperimentConfig) -> dict:
    ok = [j for j in jobs if j["state"] == "succeeded"]
    nmax = max(cfg.qubits)
    inrun: list[float] = []
    for j in ok:
        csv = Path(j["run_dir"]) / "results.csv"
        df = pd.read_csv(csv)
        inrun.append(float(df.loc[df["qubits"] == nmax, "time_mean_s"].iloc[0]))
    infos = [j.get("thread_info") or {} for j in ok]
    statuses = sorted({i.get("status", "missing") for i in infos}) or ["no_successful_job"]
    applied = sorted({thr.blas_threads_summary(i) or "null" for i in infos})
    return {
        "failed": len(jobs) - len(ok),
        # 経過時間: バッチ全体(ジョブ登録後〜全ジョブ終了)。ジョブ間の並列を含む
        "wall_s": s["wall_time_s"],
        # プロセス起動(Python/NumPy の import 等)を含むジョブ1件の時間
        **_stats([j["duration_s"] for j in ok], "job_process_s"),
        # execute() 内の時間(プロセス起動を除く)
        **_stats([j["worker_run_duration_s"] for j in ok], "job_in_execute_s"),
        # 最大nの回路実行時間(job 内の time_mean_s; 計測用の反復平均)
        "nmax": nmax, **_stats(inrun, "circuit_nmax_s"),
        "thread_status": ";".join(statuses), "blas_threads_applied": ";".join(applied),
    }


def summarize(rounds: pd.DataFrame) -> pd.DataFrame:
    """条件ごとの集計。優劣・最適の判定はしない。

    wall_range_overlaps_baseline: 基準条件(workers・threads が最小の条件)と
    [min, max] の範囲が重なるか。重なるなら差を断定できない。
    """
    rounds = rounds.copy()
    for col in ("wall_s", "failed", "job_process_s_median", "job_in_execute_s_median",
                "circuit_nmax_s_median", "thread_status"):
        if col not in rounds:  # 全条件が失敗した場合などで列が無くても集計できるようにする
            rounds[col] = None
    rows = []
    for (w, t), g in rounds.groupby(["workers", "threads"], sort=True):
        wall = g["wall_s"].dropna().tolist()
        row = {"workers": w, "threads": t, "rounds": len(g), "jobs_per_round": int(g["jobs"].iloc[0]),
               "failed_total": int(g["failed"].fillna(0).sum()), "oversubscribed": bool(g["oversubscribed"].iloc[0])}
        row.update(_stats(wall, "wall_s"))
        row["wall_s_cv"] = (row["wall_s_std"] / statistics.fmean(wall)) if len(wall) > 1 else None
        for col in ("job_process_s_median", "job_in_execute_s_median", "circuit_nmax_s_median"):
            row[f"{col}_of_rounds"] = statistics.median(g[col].dropna().tolist()) if g[col].notna().any() else None
        row["thread_status"] = ";".join(sorted({str(x) for x in g.get("thread_status", pd.Series(dtype=str)).dropna()}))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    base = df.iloc[0]  # groupby は (workers, threads) 昇順 → 先頭が基準
    df["baseline"] = (df["workers"] == base["workers"]) & (df["threads"] == base["threads"])
    bmed = base["wall_s_median"]
    df["wall_speedup_median_vs_baseline"] = [bmed / m if m else None for m in df["wall_s_median"]]
    df["wall_range_overlaps_baseline"] = [
        None if pd.isna(lo) or pd.isna(base["wall_s_min"]) else bool(lo <= base["wall_s_max"] and base["wall_s_min"] <= hi)
        for lo, hi in zip(df["wall_s_min"], df["wall_s_max"])]
    return df
