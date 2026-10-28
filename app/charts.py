"""グラフ生成。入力の DataFrame/配列の値をそのままプロットする(加工しない)。

色だけに頼らないよう、系列ごとにマーカー形状・線種・バーの模様も変えている。
"""
import numpy as np
import plotly.graph_objects as go

BLUE, ORANGE, GRAY, GREEN = "#2a6fdb", "#c25e00", "#666666", "#1b7f5c"
SYMBOLS = ["circle", "diamond", "square", "triangle-up"]


def _layout(fig, title, xt, yt, **kw):
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, legend=dict(orientation="h", y=-0.25),
                      font=dict(size=14), margin=dict(l=10, r=10, t=60, b=10), **kw)
    return fig


def probability_chart(labels, theory, sampled, shots):
    fig = go.Figure()
    fig.add_bar(x=labels, y=theory, name="理論確率(シミュレーション結果 |α|²)", marker_color=BLUE)
    fig.add_bar(x=labels, y=sampled, name=f"測定頻度({shots:,} 回のサンプリング)", marker_color=ORANGE,
                marker_pattern_shape="/")
    # ラベルは "000" のような文字列。軸の型を自動判定にすると数値 0 と解釈され、桁が消えるため category を明示する
    fig = _layout(fig, "測定確率の分布(確率の高い順)", "ビット列(左端が 0 番の量子ビット)", "確率(0〜1)", barmode="group")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(labels))
    return fig


def phase_chart(n, state):
    """複素振幅の位相 arg(α_y) を y ごとに表示(振幅がほぼ 0 の点は除く)。"""
    idx = np.nonzero(np.abs(state) > 1e-9)[0]
    fig = go.Figure(go.Scatter(x=idx, y=np.angle(state[idx]), mode="markers", name="位相 arg(α_y)",
                               marker=dict(color=GREEN, symbol="diamond", size=9)))
    return _layout(fig, "複素振幅の位相", f"基底状態の番号 y(0〜{2**n - 1})", "位相 [rad](−π〜π)")


def time_chart(df):
    fig = go.Figure()
    fig.add_scatter(x=df["qubits"], y=df["time_mean_s"], mode="lines+markers", name="平均(エラーバー=標準偏差)",
                    line_color=BLUE, marker_symbol="circle", error_y=dict(type="data", array=df["time_std_s"]))
    fig.add_scatter(x=df["qubits"], y=df["time_median_s"], mode="markers", name="中央値",
                    marker=dict(color=ORANGE, symbol="diamond", size=9))
    fig = _layout(fig, "回路規模と回路の計算時間(実測)", "量子ビット数 n", "秒(対数軸)", yaxis_type="log")
    fig.update_yaxes(dtick=1, exponentformat="power")  # 目盛りを 10 のべき乗だけにして読みやすくする
    return fig


def memory_chart(df):
    fig = go.Figure()
    fig.add_scatter(x=df["qubits"], y=df["peak_mem_bytes_measured"], mode="lines+markers",
                    name="実測ピーク(tracemalloc)", line_color=ORANGE, marker_symbol="circle")
    fig.add_scatter(x=df["qubits"], y=df["statevector_bytes_theory"], mode="lines+markers",
                    name="理論値 16·2ⁿ バイト(状態ベクトル本体)", line=dict(color=GRAY, dash="dash"),
                    marker_symbol="square")
    fig = _layout(fig, "メモリ使用量: 実測と理論値", "量子ビット数 n", "バイト(対数軸)", yaxis_type="log")
    fig.update_yaxes(dtick=1, exponentformat="power")
    return fig


def cv_chart(df):
    fig = go.Figure(go.Scatter(x=df["qubits"], y=df["time_cv"] * 100, mode="lines+markers", name="変動係数",
                               line_color=BLUE, marker_symbol="diamond"))
    return _layout(fig, "反復間のばらつき(変動係数)", "量子ビット数 n", "標準偏差 ÷ 平均 [%]")


def memory_scaling_chart(ns, bytes_theory, ram_bytes, max_qubits):
    fig = go.Figure()
    fig.add_scatter(x=list(ns), y=list(bytes_theory), mode="lines+markers", name="状態ベクトル本体の理論メモリ",
                    line_color=BLUE, marker_symbol="circle")
    fig.add_hline(y=ram_bytes, line_dash="dash", line_color=GRAY,
                  annotation_text="実行環境の搭載メモリ(参考)", annotation_position="top left")
    fig.add_vline(x=max_qubits, line_dash="dot", line_color=ORANGE,
                  annotation_text=f"このアプリの上限 n={max_qubits}", annotation_position="bottom right")
    fig = _layout(fig, "量子ビット数とメモリ(理論値)", "量子ビット数 n", "バイト(対数軸)", yaxis_type="log")
    fig.update_yaxes(dtick=2, exponentformat="power")
    return fig


def sweep_wall_chart(summary):
    """workers ごとに、threads 別の壁時計時間の中央値と最小〜最大の範囲。"""
    fig = go.Figure()
    for k, (t, g) in enumerate(summary.groupby("threads")):
        g = g.sort_values("workers")
        fig.add_scatter(x=g["workers"], y=g["wall_s_median"], mode="lines+markers", name=f"スレッド数 {t}",
                        marker_symbol=SYMBOLS[k % 4],
                        error_y=dict(type="data", symmetric=False, array=g["wall_s_max"] - g["wall_s_median"],
                                     arrayminus=g["wall_s_median"] - g["wall_s_min"]))
    return _layout(fig, "4 ジョブ全体の壁時計時間(中央値、線=最小〜最大、3 ラウンド)", "ワーカー数", "秒",
                   xaxis=dict(tickvals=sorted(summary["workers"].unique())))


def sweep_job_chart(summary, nmax):
    fig = go.Figure()
    for k, (t, g) in enumerate(summary.groupby("threads")):
        g = g.sort_values("workers")
        fig.add_scatter(x=g["workers"], y=g["circuit_nmax_s_median_of_rounds"], mode="lines+markers",
                        name=f"スレッド数 {t}", marker_symbol=SYMBOLS[k % 4])
    return _layout(fig, f"1 ジョブあたりの回路の計算時間(n={nmax}、中央値)", "ワーカー数", "秒",
                   xaxis=dict(tickvals=sorted(summary["workers"].unique())))
