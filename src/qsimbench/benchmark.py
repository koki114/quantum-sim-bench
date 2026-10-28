"""計測ユーティリティ(実行時間・ピークメモリ・実行環境)。

時間とメモリは別の実行で測る(tracemalloc は実行を遅くするため)。
"""
from __future__ import annotations

import os
import platform
import statistics
import time
import tracemalloc
from typing import Callable

import numpy as np
import psutil

from . import threads


def measure_time(fn: Callable[[], object], warmup: int, repeats: int) -> dict:
    """warmup 回は捨て、repeats 回の壁時計時間(秒)を記録する。"""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"times": times, **summarize(times)}


def summarize(times: list[float]) -> dict:
    """平均・標本標準偏差(n-1)・最小・最大・中央値・変動係数・平均の標準誤差。"""
    n = len(times)
    mean = statistics.fmean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times),
        "cv": std / mean if mean > 0 else 0.0,
        "sem": std / n**0.5,
    }


def measure_peak_memory(fn: Callable[[], object]) -> int:
    """fn 実行中の Python/NumPy 確保メモリのピーク(バイト)。tracemalloc による実測値。

    プロセス全体のRSSではなく、NumPy配列を含む Python 側の確保量のみ。
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def environment_info() -> dict:
    vm = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": platform.processor() or "unknown",
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gib": round(vm.total / 2**30, 2),
        "ram_available_gib_at_start": round(vm.available / 2**30, 2),
        "execution_device": "CPU (NumPy)",
        "gpu_used": False,
        "blas": threads.detect(),  # 検出できたBLAS(名前・バージョン・スレッド数)。検出不可なら null
        "blas_env": threads.read_env(),
        "memory_method": "tracemalloc peak (Python/NumPy allocations only; not process RSS)",
    }


def process_resources() -> dict:
    """自プロセスのOS視点の資源使用量。取得できない項目は None(未計測)。

    peak_rss_bytes: Windows は peak_wset、Linux は ru_maxrss。どちらも取れなければ None。
    cpu_time_s: user+system の累積CPU秒(壁時計時間ではない)。
    """
    p = psutil.Process()
    ct = p.cpu_times()
    peak = getattr(p.memory_info(), "peak_wset", None)
    if peak is None:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: KiB
        except ImportError:
            peak = None
    return {"peak_rss_bytes": peak, "cpu_time_s": ct.user + ct.system}
