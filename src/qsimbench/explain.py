"""UI の説明表示に使う純粋関数(整形・コスト見積もり・入力検証・結果の読み方)。

Streamlit に依存しない。数値はすべて既存の計算(statevector / jobs / jobqueue)から取得し、
ここでは新しい物理量を作らない。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from . import statevector as sv
from .jobqueue import PEAK_FACTOR
from .jobs import MAX_QUBITS

# 本測定(sweep)の保存先。リポジトリ内のコピー(元の sweeps/ は変更しない)
MEASURED_SWEEP_DIR = (Path(__file__).resolve().parents[2] / "docs" / "measurements"
                      / "sweep_20261028-235609-09c218")

LEVEL_LABEL = {"light": "🟢 軽い", "caution": "🟡 注意", "heavy": "🔴 重い", "blocked": "⛔ 実行不可"}


def format_bytes(b: float) -> str:
    """1024 基準(KiB, MiB, GiB...)。"""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"):
        if b < 1024:
            return f"{b:.4g} {unit}"
        b /= 1024
    return f"{b:.4g} ZiB"


def memory_table(ns: list[int]) -> pd.DataFrame:
    """量子ビット数ごとの振幅数と、状態ベクトル本体の理論メモリ(16·2^n バイト)。"""
    return pd.DataFrame({
        "qubits": ns,
        "amplitudes": [2**n for n in ns],
        "state_bytes_theory": [sv.statevector_bytes(n) for n in ns],
        "state_text": [format_bytes(sv.statevector_bytes(n)) for n in ns],
    })


def estimate_load(n_max: int, available_bytes: int | None = None) -> dict:
    """実行前の負荷見積もり。ピークは「理論サイズ×PEAK_FACTOR」の経験則による**推定**。"""
    avail = psutil.virtual_memory().available if available_bytes is None else available_bytes
    theory = sv.statevector_bytes(n_max)
    peak = PEAK_FACTOR * theory
    if n_max > MAX_QUBITS:
        level = "blocked"
    elif peak > 0.5 * avail:
        level = "heavy"
    elif n_max >= 20:
        level = "caution"
    else:
        level = "light"
    return {"n_max": n_max, "amplitudes": 2**n_max, "theory_bytes": theory, "peak_estimate_bytes": peak,
            "available_bytes": avail, "level": level, "label": LEVEL_LABEL[level]}


def validate_circuit_inputs(kind: str, n: int, input_value: int, depth: int) -> tuple[list[str], list[str]]:
    """(エラー, 警告)。エラーがあれば実行しない。"""
    errors, warnings = [], []
    if not 1 <= n <= MAX_QUBITS:
        errors.append(f"量子ビット数は 1〜{MAX_QUBITS} にしてください(現在 {n})。")
    elif not 0 <= input_value < 2**n:
        errors.append(f"入力する基底状態は 0〜{2**n - 1} の整数です(n={n} のとき 2^n 通り)。現在 {input_value}。")
    if kind == "random" and depth < 1:
        errors.append("層数は 1 以上にしてください。")
    if kind == "ghz" and input_value != 0:
        warnings.append("GHZ の解析解(閉形式)は入力が 0 のときだけ用意しています。この場合の検証は密行列計算が使える n≤9 に限られます。")
    return errors, warnings


def top_outcomes(probs: np.ndarray, k: int = 32) -> np.ndarray:
    """確率が高い順の基底状態インデックス(確率がほぼ 0 のものは除く)。同じ確率なら番号の小さい順。"""
    idx = np.lexsort((np.arange(len(probs)), -np.round(probs, 12)))[:k]
    return idx[probs[idx] > 1e-12]


def reading_guide(kind: str, n: int, probs: np.ndarray, input_value: int = 0) -> list[str]:
    """実際の結果の数値を使った「結果の読み方」(Markdown の箇条書き)。"""
    top = top_outcomes(probs, 3)
    if kind == "ghz" and input_value == 0:
        two = top_outcomes(probs, 2)
        names = " と ".join(f"`{sv.bitstring(int(i), n)}`" for i in sorted(two))
        return [f"確率が高いのは {names} の 2 つで、それぞれ {probs[two[0]]:.4f}(理論値 0.5)です。",
                "それ以外のビット列の確率は 0 です。「全部 0」か「全部 1」のどちらかしか出ないのが GHZ 状態の特徴です。",
                f"確率の合計は {probs.sum():.6f}(1 になるはず)です。"]
    if kind == "qft":
        N = 2**n
        return [f"入力 |{input_value}⟩ に QFT をかけると、全 {N} 通りの確率が一様に 1/{N} = {1 / N:.6f} になります"
                f"(実測の最小 {probs.min():.6f}、最大 {probs.max():.6f})。",
                "確率が一様なので、測定確率のグラフだけでは入力の違いは見えません。"
                "入力の情報は、複素振幅の**位相**(角度)に入っています。下の「位相のグラフ」で確認できます。"]
    lines = ["上位の結果: " + "、".join(f"`{sv.bitstring(int(i), n)}`={probs[i]:.4f}" for i in top) + "。",
             "ランダム回路は、乱数で決めた回転ゲートを使うため、確率が特定のビット列に偏ったり、広く散らばったりします。"
             "同じシード・層数なら同じ回路になります。"]
    return lines


def tabulate_gates(circuit, limit: int = 60) -> pd.DataFrame:
    rows = [{"番号": i + 1, "ゲート": g.name, "作用する量子ビット": ", ".join(map(str, g.qubits)),
             "角度(ラジアン)": ", ".join(f"{p:.4f}" for p in g.params) or "-"}
            for i, g in enumerate(circuit.gates[:limit])]
    return pd.DataFrame(rows)


def load_measured_sweep(base: Path | None = None) -> dict | None:
    """同梱の本測定データ(実測)を読む。無ければ None。"""
    base = Path(base) if base else MEASURED_SWEEP_DIR
    try:
        return {"meta": json.loads((base / "sweep.json").read_text(encoding="utf-8")),
                "summary": pd.read_csv(base / "summary.csv"),
                "rounds": pd.read_csv(base / "rounds.csv")}
    except (OSError, ValueError):
        return None


def benchmark_reading(ok: pd.DataFrame) -> list[str]:
    """性能測定の結果(status=ok の行)から、数値入りの「読み方」を作る。過度な一般化はしない。"""
    ok = ok.sort_values("qubits")
    lines: list[str] = []
    if len(ok) >= 2:
        a, b = ok.iloc[-2], ok.iloc[-1]
        r = (b["time_mean_s"] / a["time_mean_s"]) ** (1 / (b["qubits"] - a["qubits"]))
        lines.append(f"最大の 2 点(n={int(a['qubits'])}→{int(b['qubits'])})の平均時間から、n が 1 増えるごとに時間は"
                     f"約 {r:.2f} 倍になっています。1 ゲートの計算量が 2 倍になるうえ、回路によってはゲート数も増えるため、"
                     "2 倍前後かそれ以上になるのが普通です。小さい n では Python の準備時間が支配的で、この値は参考です。")
    if "peak_mem_bytes_measured" in ok and ok["peak_mem_bytes_measured"].notna().any():
        last = ok.iloc[-1]
        ratio = last["peak_mem_bytes_measured"] / last["statevector_bytes_theory"]
        lines.append(f"最大の n={int(last['qubits'])} では、🟢 実測のピークメモリ {format_bytes(last['peak_mem_bytes_measured'])} は"
                     f"🔵 理論値 {format_bytes(last['statevector_bytes_theory'])} の {ratio:.2f} 倍です"
                     "(理論値は状態ベクトル本体のみで、ゲートごとに一時配列ができるため実測のほうが大きくなります)。")
    cv = ok["time_cv"].max()
    where = int(ok.loc[ok["time_cv"].idxmax(), "qubits"])
    if cv <= 0.05:
        verdict = "測定は比較的安定しています"
    elif cv <= 0.2:
        verdict = "多少のばらつきがあります"
    else:
        verdict = "ばらつきが大きく、他の処理の影響を受けた可能性があります(反復回数を増やす、他のアプリを閉じる)"
    lines.append(f"最大の変動係数は n={where} で {cv * 100:.1f}% です。{verdict}(5% / 20% は目安で、厳密な基準ではありません)。")
    if "max_abs_error_vs_reference" in ok and ok["max_abs_error_vs_reference"].notna().any():
        lines.append(f"参照結果との最大誤差は {ok['max_abs_error_vs_reference'].max():.1e} です"
                     "(1e-15 程度なら数値誤差の範囲で一致)。参照のない点は確率の合計だけを確認しています。")
    return lines


def sweep_reading(s: pd.DataFrame, nmax: int) -> list[str]:
    """並列実行の実測要約(summary.csv)から、数値入りの読み方を作る。"""
    lines: list[str] = []
    t1 = s[s["threads"] == 1].sort_values("workers")
    if len(t1) >= 2:
        base = t1.iloc[0]
        for _, row in t1.iloc[1:].iterrows():
            lines.append(
                f"スレッド数 1 で、ワーカーを {int(base['workers'])}→{int(row['workers'])} にすると、壁時計時間(中央値)は "
                f"{base['wall_s_median']:.1f}→{row['wall_s_median']:.1f} 秒(約 {base['wall_s_median'] / row['wall_s_median']:.2f} 倍速)。"
                f"一方、1 ジョブあたりの回路時間(n={nmax})は {base['circuit_nmax_s_median_of_rounds']:.2f}→"
                f"{row['circuit_nmax_s_median_of_rounds']:.2f} 秒(約 "
                f"{row['circuit_nmax_s_median_of_rounds'] / base['circuit_nmax_s_median_of_rounds']:.2f} 倍に増加)です。")
    w0 = s[s["workers"] == s["workers"].min()].sort_values("threads")
    if len(w0) >= 2:
        lo, hi = w0["wall_s_min"].max(), w0["wall_s_max"].min()
        overlap = "すべての範囲が重なっており、差を断定できません" if lo <= hi else "範囲が重ならない組がありました"
        lines.append(f"ワーカー {int(w0['workers'].iloc[0])} でスレッド数を {int(w0['threads'].iloc[0])}→{int(w0['threads'].iloc[-1])} と"
                     f"変えたときの壁時計時間(中央値)は {w0['wall_s_median'].min():.1f}〜{w0['wall_s_median'].max():.1f} 秒で、{overlap}。")
    worst = s.loc[s["wall_s_cv"].idxmax()]
    lines.append(f"壁時計時間のばらつきが最も大きかったのは、ワーカー {int(worst['workers'])}・スレッド {int(worst['threads'])} の条件"
                 f"(変動係数 {worst['wall_s_cv'] * 100:.0f}%)です。外れ値を含む 3 ラウンドの結果なので、この条件が不安定だとは断定できません。")
    return lines
