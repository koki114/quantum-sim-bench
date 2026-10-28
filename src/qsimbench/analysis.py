"""結果CSVの性能分析。入力は results.csv の status=="ok" の行。

出力はすべて「実測値へのフィット/比」であり、測定範囲外への外挿はしない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scaling_summary(df: pd.DataFrame, min_qubits: int = 12) -> dict:
    """n ≥ min_qubits の点で log2(時間) ~ a·n + b をフィットし、メモリ比と変動係数も返す。

    傾き a は「n が 1 増えたときの時間の倍率(2^a)」。状態ベクトル法の理想は a≈1。
    点が 2 未満なら slope は None(未計算)。
    """
    ok = df[df["status"] == "ok"] if "status" in df else df
    big = ok[ok["qubits"] >= min_qubits]
    out: dict = {"points_used": int(len(big)), "min_qubits": min_qubits, "slope_log2_time_per_qubit": None}
    if len(big) >= 2:
        out["slope_log2_time_per_qubit"] = float(np.polyfit(big["qubits"], np.log2(big["time_mean_s"]), 1)[0])
    if "peak_mem_bytes_measured" in ok and len(ok):
        ratio = ok["peak_mem_bytes_measured"] / ok["statevector_bytes_theory"]
        out["measured_peak_over_theory_max_n"] = float(ratio.iloc[-1])
    out["max_time_cv"] = float(ok["time_cv"].max()) if len(ok) else None
    return out
