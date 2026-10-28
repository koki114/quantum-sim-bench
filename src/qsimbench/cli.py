"""CLI。

``run``: 設定ファイルを(このプロセスで)直接実行する。複数指定でバッチ、1つでも失敗なら非0。
``submit/work/list/show/retry/cancel/export/compare-workers``: ローカルジョブキュー(jobqueue.py)。
``analyze``: 結果CSVのスケーリング要約。
``sweep``: workers × BLASスレッド数 の比較実験(sweeps/ に新規保存)。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import jobqueue as jq
from .jobs import execute, load_config


def _cmd_run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    worst = 0
    for path in args.configs:
        try:
            cfg = load_config(path)
        except Exception as e:  # noqa: BLE001
            print(f"[{path}] config error: {e}", file=sys.stderr)
            worst = worst or 2
            continue
        rec = execute(cfg, args.out)
        print(f"[{path}] {rec.status} exit={rec.exit_code} {rec.duration_s:.2f}s -> {rec.out_dir}")
        if rec.rows:
            cols = ["qubits", "status", "time_mean_s", "time_std_s", "peak_mem_bytes_measured",
                    "max_abs_error_vs_reference"]
            print(rec.dataframe().reindex(columns=cols).to_string(index=False))
        worst = worst or rec.exit_code
    return worst


def _cmd_submit(args) -> int:
    code = 0
    for path in args.configs:
        try:
            print(f"{jq.submit(path, args.root)}  queued  <- {path}")
        except Exception as e:  # noqa: BLE001
            print(f"[{path}] not submitted: {e}", file=sys.stderr)
            code = 2
    return code


def _print_summary(s: dict) -> None:
    print(f"batch {s['batch_id']}: workers={s['workers']} threads/job={s['threads_per_job']} "
          f"jobs={s['jobs_run']} succeeded={s['succeeded']} failed={s['failed']} "
          f"left_queued={s['left_queued']} wall={s['wall_time_s']:.2f}s "
          f"(sum of job durations {s['sum_job_durations_s']:.2f}s)")


def _cmd_work(args) -> int:
    s = jq.run_queue(args.root, args.workers, args.threads_per_job, args.fail_fast)
    _print_summary(s)
    return 1 if s["failed"] else 0


def _cmd_list(args) -> int:
    df = jq.jobs_dataframe(args.root)
    if df.empty:
        print("(no jobs)")
        return 0
    cols = ["job_id", "state", "name", "circuit", "qubits", "exit_code", "duration_s", "retry_of"]
    print(df[cols].to_string(index=False))
    return 0


def _cmd_show(args) -> int:
    print(json.dumps(jq.load_job(args.root, args.job_id), indent=2, ensure_ascii=False))
    return 0


def _cmd_retry(args) -> int:
    ids = args.job_ids or [j["job_id"] for j in jq.list_jobs(args.root) if j["state"] == "failed"
                           and not any(k["retry_of"] == j["job_id"] for k in jq.list_jobs(args.root))]
    if not ids:
        print("no failed jobs to retry")
        return 0
    for i in ids:
        print(f"{jq.retry(args.root, i)}  queued  (retry of {i})")
    return 0


def _cmd_cancel(args) -> int:
    jq.cancel(args.root, args.job_id)
    print(f"{args.job_id} cancelled")
    return 0


def _cmd_export(args) -> int:
    if not (args.jobs or args.results):
        print("specify --jobs and/or --results", file=sys.stderr)
        return 2
    if args.jobs:
        print("wrote", jq.export_csv(jq.jobs_dataframe(args.root), args.jobs))
    if args.results:
        print("wrote", jq.export_csv(jq.results_dataframe(args.root), args.results))
    return 0


def _cmd_compare(args) -> int:
    rows = []
    for w in args.workers:
        for path in args.configs:
            jq.submit(path, args.root)
        s = jq.run_queue(args.root, w, args.threads_per_job)
        rows.append(s)
        _print_summary(s)
    base = rows[0]["wall_time_s"]
    table = pd.DataFrame([{"workers": r["workers"], "threads_per_job": r["threads_per_job"],
                           "jobs": r["jobs_run"], "failed": r["failed"], "wall_time_s": round(r["wall_time_s"], 3),
                           "speedup_vs_first": round(base / r["wall_time_s"], 2)} for r in rows])
    print(table.to_string(index=False))
    return 1 if any(r["failed"] for r in rows) else 0


def _cmd_sweep(args) -> int:
    from .sweep import run_sweep

    out = run_sweep(args.config, args.workers, args.threads, args.jobs_per_condition, args.rounds,
                    args.seed, args.out, args.allow_oversubscribe)
    print("sweep written to", out)
    summ = pd.read_csv(out / "summary.csv")
    cols = ["workers", "threads", "rounds", "failed_total", "wall_s_median", "wall_s_min", "wall_s_max",
            "circuit_nmax_s_median_of_rounds", "wall_range_overlaps_baseline", "thread_status"]
    print(summ.reindex(columns=cols).to_string(index=False))
    return 1 if summ["failed_total"].sum() else 0


def _cmd_analyze(args) -> int:
    from .analysis import scaling_summary

    print(json.dumps(scaling_summary(pd.read_csv(args.csv)), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qsimbench", description="CPU状態ベクトルシミュレーションとベンチマーク")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, root=True):
        p = sub.add_parser(name, help=help_)
        if root:
            p.add_argument("--root", default="jobs", help="ジョブ保存先ディレクトリ(既定: jobs)")
        p.set_defaults(fn=fn)
        return p

    p = add("run", _cmd_run, "設定ファイル(.toml/.json)を直接実行する", root=False)
    p.add_argument("configs", nargs="+")
    p.add_argument("--out", default="results", help="結果の出力先(既存の run は上書きしない)")

    p = add("submit", _cmd_submit, "設定をジョブとして登録する(実行はしない)")
    p.add_argument("configs", nargs="+")

    p = add("work", _cmd_work, "queued ジョブを実行する")
    p.add_argument("--workers", type=int, default=1, help="同時実行数(既定 1 = 逐次)")
    p.add_argument("--threads-per-job", type=int, default=None, help="各ジョブのBLASスレッド数(既定: 物理コア数÷workers)")
    p.add_argument("--fail-fast", action="store_true", help="失敗後は新しいジョブを開始しない")

    add("list", _cmd_list, "ジョブ一覧")
    p = add("show", _cmd_show, "ジョブの詳細(JSON)")
    p.add_argument("job_id")
    p = add("retry", _cmd_retry, "失敗ジョブを新しいジョブとして再登録(未指定なら未再実行の全失敗ジョブ)")
    p.add_argument("job_ids", nargs="*")
    p = add("cancel", _cmd_cancel, "queued ジョブを取り消す")
    p.add_argument("job_id")
    p = add("export", _cmd_export, "ジョブ一覧/結果をCSV出力(既存ファイルは上書きしない)")
    p.add_argument("--jobs", help="ジョブ表の出力先CSV")
    p.add_argument("--results", help="全ジョブの結果行の出力先CSV")
    p = add("compare-workers", _cmd_compare, "同じ設定を異なる同時実行数で実行して比較する")
    p.add_argument("configs", nargs="+")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--threads-per-job", type=int, default=None)

    p = add("sweep", _cmd_sweep, "workers と BLAS スレッド数を独立に振る比較実験(新規ディレクトリに保存)", root=False)
    p.add_argument("config")
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4], help="同時実行ジョブ数の候補")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="各ジョブのBLASスレッド数の候補")
    p.add_argument("--jobs-per-condition", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3, help="全条件を繰り返す回数(順序は毎回シャッフル)")
    p.add_argument("--seed", type=int, default=0, help="条件の実行順を決める seed")
    p.add_argument("--out", default="sweeps")
    p.add_argument("--allow-oversubscribe", action="store_true", help="workers×threads が論理コア数を超える条件も実行")

    p = add("analyze", _cmd_analyze, "結果CSVのスケーリング要約", root=False)
    p.add_argument("csv")

    p = sub.add_parser("worker")  # 内部用: ジョブの子プロセス
    p.add_argument("job_dir")
    p.add_argument("--threads", type=int, default=None)
    p.set_defaults(fn=lambda a: jq.worker_main(a.job_dir, a.threads))

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (KeyError, ValueError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
