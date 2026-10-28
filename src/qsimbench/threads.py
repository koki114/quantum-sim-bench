"""BLASスレッド数の設定と、実際に適用された値の検出。

3つの値を区別して記録する(取得できない値は推測せず None):
- requested: こちらが要求した値
- env:       プロセスが見ている OMP/OPENBLAS/MKL_NUM_THREADS(未設定なら None)
- applied:   threadpoolctl.threadpool_info() が検出したライブラリごとの実スレッド数

検出は「そのライブラリがロード済み」のときだけ可能。NumPy を import した後に呼ぶこと。
"""
from __future__ import annotations

import os

import numpy as np  # noqa: F401  (BLAS をロードさせてから検出する)

try:
    import threadpoolctl
except ImportError:  # 任意依存として扱う: 無ければ applied は None(未確認)
    threadpoolctl = None

ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def read_env() -> dict:
    return {k: os.environ.get(k) for k in ENV_KEYS}


def detect() -> list[dict] | None:
    """検出したライブラリのリスト。threadpoolctl が無ければ None、検出対象なしなら []。"""
    if threadpoolctl is None:
        return None
    keep = ("user_api", "internal_api", "version", "num_threads", "threading_layer")
    return [{k: d.get(k) for k in keep} for d in threadpoolctl.threadpool_info()]


def classify(requested: int | None, applied: list[dict] | None) -> str:
    """検出結果の分類。

    threadpoolctl_unavailable: 検出手段が無い(applied=None)
    no_library_detected:       検出手段はあるが対象ライブラリが見つからない
    no_blas_detected:          BLAS 以外しか見つからない
    no_request:                要求値なし(検出のみ)
    ok / mismatch:             BLAS の実スレッド数が要求値と一致 / 不一致
    """
    if applied is None:
        return "threadpoolctl_unavailable"
    if not applied:
        return "no_library_detected"
    blas = [d for d in applied if d["user_api"] == "blas"]
    if not blas:
        return "no_blas_detected"
    if requested is None:
        return "no_request"
    return "ok" if all(d["num_threads"] == requested for d in blas) else "mismatch"


class ThreadSettings:
    """要求スレッド数を適用し(計算の前)、適用状況を記録する。

    start() で設定+検出、finish() で計算後に再検出して変化がないか確認する。
    """

    def __init__(self, requested: int | None):
        if requested is not None and requested < 1:
            raise ValueError("threads must be >= 1")
        self.requested = requested
        self._limiter = None
        self._start: list[dict] | None = None
        self._start_status = ""

    def start(self) -> None:
        if self.requested is not None and threadpoolctl is not None:
            self._limiter = threadpoolctl.threadpool_limits(limits=self.requested, user_api="blas")
        self._start = detect()
        self._start_status = classify(self.requested, self._start)

    def finish(self) -> dict:
        end = detect()
        end_status = classify(self.requested, end)
        if self._limiter is not None:
            self._limiter.restore_original_limits()  # 同一プロセス内の呼び出し(テスト等)に設定を残さない
        return {
            "requested": self.requested,
            "env": read_env(),
            "threadpoolctl_version": getattr(threadpoolctl, "__version__", None),
            "applied_at_start": self._start,
            "applied_at_end": end,
            "status": self._start_status if self._start_status == end_status else "changed_during_run",
        }

    def __enter__(self) -> "ThreadSettings":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.info = self.finish()


def blas_threads_summary(info: dict | None) -> str | None:
    """検出された BLAS の実スレッド数を 'openblas:4' 形式の文字列に(計算開始時点の値)。"""
    if not info or not info.get("applied_at_start"):
        return None
    return ",".join(f"{d['internal_api']}:{d['num_threads']}" for d in info["applied_at_start"]
                    if d["user_api"] == "blas") or None
