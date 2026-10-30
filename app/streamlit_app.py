"""Streamlit UI。計算は qsimbench に委譲し、ここでは入力・説明・表示だけを行う。"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import streamlit as st  # noqa: E402

import charts  # noqa: E402
import content as C  # noqa: E402
from qsimbench import circuits, explain, statevector as sv  # noqa: E402
from qsimbench.jobs import MAX_QUBITS, ExperimentConfig, config_to_toml, execute, run_point  # noqa: E402

st.set_page_config(page_title="量子回路シミュレーション・性能評価ツール", layout="wide")

TAB_NAMES = ["はじめに", "① 回路を試す", "② 規模とコスト", "③ 性能測定", "④ 並列実行の実測例", "⑤ 数式としくみ", "⑥ 用語集"]


# ---------------------------------------------------------------- 共通部品
_PLOT_WIDTH_PARAM = "width" in inspect.signature(st.plotly_chart).parameters  # 新しい Streamlit は width="stretch"


def show_plot(fig, key: str, container=None) -> None:
    """グラフ表示。key は必須(同一内容のグラフが複数あっても ID が重複しないように、場所ごとに固定)。"""
    target = container or st
    if _PLOT_WIDTH_PARAM:
        target.plotly_chart(fig, width="stretch", key=key)
    else:
        target.plotly_chart(fig, use_container_width=True, key=key)


def terms_box(tab: str) -> None:
    with st.expander("📖 この画面で出てくる用語"):
        for t in C.TERMS_BY_TAB[tab]:
            st.markdown(f"**{t}** — {C.GLOSSARY[t][0]}")
        st.caption("全ての用語は「⑥ 用語集」タブにあります。")


def precheck(n_values: list[int], runs: int | None = None) -> dict:
    """実行前チェック: 条件と予想される負荷を表示し、見積もりを返す。"""
    est = explain.estimate_load(max(n_values))
    with st.container(border=True):
        st.markdown(f"**実行前チェック** — 負荷の目安: {est['label']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("最大の量子ビット数", est["n_max"])
        c2.metric("振幅の数 (2ⁿ)", f"{est['amplitudes']:,}")
        c3.metric("🔵 状態ベクトルの理論メモリ", explain.format_bytes(est["theory_bytes"]))
        st.caption(
            f"🟠 推定: 計算中のピークメモリは理論値の約 3 倍({explain.format_bytes(est['peak_estimate_bytes'])})。"
            f"いま使えるメモリは約 {explain.format_bytes(est['available_bytes'])}。"
            "この推定は経験則で、保証ではありません。")
        if est["level"] == "heavy":
            st.warning("⚠️ メモリを多く使います。他のアプリを閉じてから実行してください。最悪の場合、PC が重くなったり実行が失敗したりします。")
        elif est["level"] == "caution":
            st.info("ℹ️ n=20 以上は、実行に数秒〜数十秒かかることがあります。")
    return est


def failure_message(e: Exception) -> None:
    st.error(f"**計算に失敗しました。** 原因: `{type(e).__name__}: {e}`\n\n"
             "**次にすること**: 量子ビット数を小さくして再実行してください。直らない場合は、条件(回路・量子ビット数・シード)を"
             "控えて、README の「困ったとき」を確認してください。")


def run_simulation(kind, n, input_value, shots, seed, depth) -> dict:
    cfg = ExperimentConfig(name="ui_sim", circuit=kind, qubits=[n], depth=depth, input_value=input_value,
                           shots=shots, seed=seed, warmup=0, repeats=3, measure_memory=True)
    cfg.validate()
    circ = circuits.build(kind, n, depth, seed)
    state = sv.simulate(circ, input_value)
    probs = sv.probabilities(state)
    counts = sv.sample_counts(probs, shots, seed)
    return {"cfg": cfg, "kind": kind, "n": n, "input_value": input_value, "shots": shots, "seed": seed,
            "circuit": circ, "state": state, "probs": probs, "counts": counts, "row": run_point(cfg, n)}


def show_simulation(res: dict, prefix: str) -> None:
    kind, n, probs, row = res["kind"], res["n"], res["probs"], res["row"]
    st.markdown("#### 1. 実行条件")
    # 値の列は、型が混在すると Arrow 変換(実ブラウザでの表示)で失敗するため、すべて str にそろえる
    st.table(pd.DataFrame({"項目": ["回路", "量子ビット数 n", "入力の基底状態", "測定回数(ショット)", "乱数シード"],
                           "値": [C.KIND_LABEL[kind], str(n), str(res["input_value"]), f"{res['shots']:,}", str(res["seed"])]}))
    st.markdown("#### 2. 結果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ゲート数", len(res["circuit"]))
    c2.metric("🟢 実測: 回路の計算時間(3回平均)", f"{row['time_mean_s'] * 1e3:.3g} ms")
    c3.metric("🔵 理論: 状態ベクトルのメモリ", explain.format_bytes(row["statevector_bytes_theory"]))
    err = row.get("max_abs_error_vs_reference", float("nan"))
    if np.isnan(err):
        c4.metric("参照結果との照合", "参照なし", "確率の合計のみ確認", delta_color="off")
    else:
        c4.metric(f"参照結果との照合({row['reference_kind']})", "一致 ✓" if err < 1e-9 else "不一致 ✗",
                  f"最大誤差 {err:.1e}", delta_color="off")
    top = explain.top_outcomes(probs, 32)
    labels = [sv.bitstring(int(i), n) for i in top]
    show_plot(charts.probability_chart(labels, probs[top], res["counts"][top] / res["counts"].sum(),
                                       res["shots"]), f"{prefix}_prob")
    if kind == "qft":
        show_plot(charts.phase_chart(n, res["state"]), f"{prefix}_phase")
    with st.expander("複素振幅の値(実部・虚部・大きさ・位相)を見る"):
        idx = np.arange(min(2**n, 64))
        a = res["state"][idx]
        st.dataframe(pd.DataFrame({"ビット列": [sv.bitstring(int(i), n) for i in idx], "実部": a.real,
                                   "虚部": a.imag, "大きさ |α|": np.abs(a), "確率 |α|²": np.abs(a) ** 2,
                                   "位相 [rad]": np.angle(a)}), hide_index=True)
        if 2**n > 64:
            st.caption("先頭の 64 件のみ表示しています。")
    with st.expander("回路の中身(ゲートの列)を見る"):
        st.dataframe(explain.tabulate_gates(res["circuit"]), hide_index=True)
        if len(res["circuit"]) > 60:
            st.caption("先頭の 60 ゲートのみ表示しています。")
    st.markdown("#### 3. 結果の読み方")
    for line in explain.reading_guide(kind, n, probs, res["input_value"]):
        st.markdown(f"- {line}")
    st.markdown("#### 4. 注意点")
    st.markdown(
        f"- 「測定頻度」は {res['shots']:,} 回のくじ引きの結果なので、理論確率とは少しずれます"
        f"(全変動距離 TVD = {row['tvd_sampling_vs_sim']:.4f}。有限回数での目安は約 {row['tvd_expected_scale_theory']:.4f})。\n"
        "- 時間は**このアプリを動かしているコンピュータの CPU での実測**です(公開版ではクラウドの共有 CPU)。GPU の性能を示すものではありません。小さい n では時間がほぼ「Python の準備時間」です。")
    st.markdown("#### 5. 次に試すとよい操作")
    st.markdown("- 量子ビット数 n を 1 つ増やして、振幅の数・理論メモリ・時間がどう変わるか見る(→「② 規模とコスト」)。\n"
                "- 別の回路に切り替えて、確率の形を比べる。\n- 性能をきちんと比べたいときは「③ 性能測定」へ。")


# ---------------------------------------------------------------- 画面
st.title("量子回路シミュレーション・性能評価ツール")
st.caption("CPU 上の状態ベクトル方式 / **GPU は使用していません** / このアプリが動いているコンピュータだけで計算します")

tabs = st.tabs(TAB_NAMES)

# ---- はじめに
with tabs[0]:
    st.markdown(f"### {C.INTRO}")
    cols = st.columns(3)
    for col, (title, text) in zip(cols, C.STEPS):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
    terms_box("intro")
    st.markdown("#### まず試す")
    st.write("下のボタンで、3 量子ビットの GHZ 状態を計算します(1 秒もかかりません)。")
    if st.button("▶ 3 量子ビットの GHZ 状態を計算する", type="primary", key="run_demo"):
        try:
            with st.spinner("計算中…"):
                st.session_state["demo"] = run_simulation("ghz", 3, 0, 10000, 0, 4)
            st.success("計算できました。下の結果を上から順に読んでみてください。")
        except Exception as e:  # noqa: BLE001
            failure_message(e)
    if "demo" in st.session_state:
        show_simulation(st.session_state["demo"], "demo")
    with st.expander("このアプリの範囲と制約"):
        st.markdown(C.SCOPE_NOTE)
    with st.expander("色と記号の見方"):
        st.markdown("🟢 **実測** … このアプリを動かしているコンピュータで実際に測った値 / 🔵 **理論値** … 式から計算した値 / "
                    "🟠 **推定** … 実測から見積もった値で、保証はない。グラフは色に加えて、線種・マーカー・模様でも区別しています。")

# ---- ① 回路を試す
with tabs[1]:
    st.header("① 回路を試す")
    st.write("量子回路を 1 つ選んで実行し、測定確率と実行時間を確認します。")
    terms_box("try")
    kind = st.radio("**ステップ 1: 回路を選ぶ**", circuits.CIRCUIT_KINDS, format_func=C.KIND_LABEL.get,
                    key="kind_sim", horizontal=True)
    st.caption(C.KIND_DESC[kind])
    n = st.slider("**ステップ 2: 量子ビット数 n**", 1, 14, 3, key="n_sim",
                  help=C.GLOSSARY["量子ビット（qubit）"][0] + " n が 1 増えると、振幅の数は 2 倍になります。")
    st.caption(f"n={n} → 振幅の数 2ⁿ = {2**n:,} 個、状態ベクトルの理論メモリ = {explain.format_bytes(sv.statevector_bytes(n))}")
    input_value, depth = 0, 4
    with st.expander("詳細設定(測定回数・シード・入力など)"):
        shots = st.number_input("測定回数(ショット数)", 100, 10_000_000, 10_000, step=1000, key="shots_sim",
                                help="回路を何回測定するか。多いほど頻度が理論確率に近づきます。")
        seed = st.number_input("乱数シード", 0, 2**31 - 1, 0, key="seed_sim",
                               help="同じ値なら、測定結果とランダム回路が再現されます。")
        if kind == "qft":
            input_value = st.number_input(f"入力する基底状態 x(0〜{2**n - 1})", 0, 2**n - 1, min(3, 2**n - 1),
                                          key=f"input_sim_{n}", help="QFT の入力 |x⟩ の番号。範囲は n に応じて変わります。")
        if kind == "random":
            depth = st.slider("層数(ランダム回路)", 1, 12, 4, key="depth_sim", help="回転ゲートと CX の層を何回重ねるか。")
    errors, warnings = explain.validate_circuit_inputs(kind, n, int(input_value), int(depth))
    for m in errors:
        st.error(m)
    for m in warnings:
        st.warning(m)
    precheck([n])
    if st.button("**ステップ 3: 計算する**", type="primary", key="run_sim", disabled=bool(errors)):
        with st.status("計算中…", expanded=False) as status:
            try:
                st.session_state["sim"] = run_simulation(kind, n, int(input_value), int(shots), int(seed), int(depth))
                status.update(label="計算が完了しました", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label="計算に失敗しました", state="error")
                st.session_state.pop("sim", None)
                failure_message(e)
    if "sim" in st.session_state:
        st.success("計算が完了しました。下の結果を読んでみましょう。")
        show_simulation(st.session_state["sim"], "sim")

# ---- ② 規模とコスト
with tabs[2]:
    st.header("② 量子ビット数と計算コスト")
    st.write("状態ベクトル方式では、量子ビットを 1 つ増やすたびに、必要なメモリが 2 倍になります。実行せずに、理論値を確認できます。")
    terms_box("cost")
    n_cost = st.slider("量子ビット数 n", 1, 40, 20, key="n_cost")
    c1, c2, c3 = st.columns(3)
    c1.metric("振幅の数 2ⁿ", f"{2**n_cost:,}")
    c2.metric("🔵 理論メモリ 16×2ⁿ バイト", explain.format_bytes(sv.statevector_bytes(n_cost)))
    c3.metric("実行環境の搭載メモリ(参考。公開版ではクラウドのサーバー)", explain.format_bytes(psutil.virtual_memory().total))
    st.latex(C.LATEX["amplitudes"])
    st.latex(C.LATEX["memory"])
    st.markdown("振幅 1 個は倍精度の複素数で 16 バイトです(実部 8 + 虚部 8)。単位は 1024 を基準とする KiB・MiB・GiB です。")
    tbl = explain.memory_table(sorted({10, 20, 25, 30, n_cost}))
    st.dataframe(tbl.rename(columns={"qubits": "量子ビット数", "amplitudes": "振幅の数",
                                     "state_text": "状態ベクトル本体の理論メモリ"})[
        ["量子ビット数", "振幅の数", "状態ベクトル本体の理論メモリ"]], hide_index=True)
    ns = list(range(1, 41))
    show_plot(charts.memory_scaling_chart(ns, [sv.statevector_bytes(k) for k in ns],
                                          psutil.virtual_memory().total, MAX_QUBITS), "cost_scaling")
    st.markdown("#### 注意点")
    st.markdown(
        "- これは**状態ベクトル本体の理論値**で、プロセス全体のメモリ使用量ではありません。ゲート演算の一時配列や Python/NumPy の"
        "管理領域などで追加のメモリが必要です(このアプリの実測では、ピークは理論値の約 3 倍でした。🟢 READMEの測定例: n=20 で 3.00 倍)。\n"
        "- この式が当てはまるのは**状態ベクトル方式**のシミュレーションです。他の方式には一般化できません。\n"
        f"- このアプリは安全のため n={MAX_QUBITS} までに制限しています。時間も n の増加とともに増えます(実測は「③ 性能測定」)。")

# ---- ③ 性能測定
with tabs[3]:
    st.header("③ 性能測定(ベンチマーク)")
    st.write("回路の規模を変えながら実行して、**計算時間・メモリ・測定のばらつき**を測ります。"
             "同じ条件で複数回測り、平均や中央値などで比べます。")
    terms_box("bench")
    with st.expander("各指標の意味(何が分かるか)"):
        for name, text in C.METRICS:
            st.markdown(f"**{name}** — {text}")
        st.latex(C.LATEX["cv"])
    kind_b = st.selectbox("**回路**", circuits.CIRCUIT_KINDS, format_func=C.KIND_LABEL.get, key="kind_bench")
    lo, hi = st.slider("**測定する量子ビット数の範囲**", 2, MAX_QUBITS, (4, 14), key="range_bench",
                       help="この範囲の n を順に測ります。大きい n ほど時間とメモリが必要です。")
    step = st.select_slider("刻み", [1, 2, 4], value=2, key="step_bench")
    repeats = st.number_input("反復回数(多いほどばらつきを評価しやすい)", 1, 50, 3, key="repeats_bench")
    with st.expander("詳細設定(ウォームアップ・シード・メモリ測定など)"):
        warmup = st.number_input("ウォームアップ回数(測定に含めない準備実行)", 0, 20, 1, key="warmup_bench")
        shots_b = st.number_input("測定回数(ショット数)", 100, 1_000_000, 5000, step=1000, key="shots_bench")
        seed_b = st.number_input("乱数シード", 0, 2**31 - 1, 0, key="seed_bench")
        mem = st.checkbox("メモリを測定する(tracemalloc)", True, key="mem_bench")
        depth_b = st.slider("層数(ランダム回路)", 1, 12, 4, key="depth_bench") if kind_b == "random" else 4
        input_b = st.number_input("QFT の入力基底状態 x", 0, 2**lo - 1, min(3, 2**lo - 1), key=f"input_bench_{lo}") \
            if kind_b == "qft" else 0
    sizes = list(range(lo, hi + 1, step))
    errors, warnings = explain.validate_circuit_inputs(kind_b, lo, int(input_b), int(depth_b))
    for m in errors:
        st.error(m)
    for m in warnings:
        st.warning(m)
    est = precheck(sizes)
    st.caption(f"測定する n: {sizes}。1 つの n につき、ウォームアップ {int(warmup)} 回 + 測定 {int(repeats)} 回"
               f"{' + メモリ測定 1 回' if mem else ''} を実行します。")
    confirmed = True
    if est["level"] == "heavy":
        confirmed = st.checkbox("負荷が重いことを理解したうえで実行する", key="confirm_bench")
    if st.button("**性能測定を実行する**", type="primary", key="run_bench", disabled=bool(errors) or not confirmed):
        cfg = ExperimentConfig(name="ui_bench", circuit=kind_b, qubits=sizes, depth=int(depth_b),
                               input_value=int(input_b), shots=int(shots_b), seed=int(seed_b),
                               warmup=int(warmup), repeats=int(repeats), measure_memory=mem)
        with st.status("測定中…(大きな n は時間がかかります)", expanded=False) as status:
            rec = execute(cfg, out_root=None)
            if rec.status == "success":
                status.update(label="測定が完了しました", state="complete")
            else:
                status.update(label="測定に失敗しました", state="error")
        st.session_state["rec"] = rec
    rec = st.session_state.get("rec")
    if rec is not None:
        if rec.status != "success":
            st.error("**測定に失敗しました。** 原因: `" + (rec.error or "不明").strip().splitlines()[-1] +
                     "`\n\n**次にすること**: 量子ビット数の範囲を小さくして再実行してください。成功した点だけは下に表示します。")
        df = rec.dataframe()
        ok = df[df["status"] == "ok"] if "status" in df else df
        if len(ok):
            st.success(f"{len(ok)} 点の測定が完了しました。")
            st.markdown("#### 1. 実行条件")
            st.json({"config": rec.config, "environment": rec.environment}, expanded=False)
            st.markdown("#### 2. 結果")
            show_plot(charts.time_chart(ok), "bench_time")
            g1, g2 = st.columns(2)
            if "peak_mem_bytes_measured" in ok:
                show_plot(charts.memory_chart(ok), "bench_mem", g1)
            show_plot(charts.cv_chart(ok), "bench_cv", g2)
            st.dataframe(ok.rename(columns={
                "qubits": "量子ビット数", "time_mean_s": "平均[s]", "time_median_s": "中央値[s]",
                "time_min_s": "最小[s]", "time_max_s": "最大[s]", "time_std_s": "標準偏差[s]", "time_cv": "変動係数",
                "max_abs_error_vs_reference": "参照との最大誤差"})[
                ["量子ビット数", "平均[s]", "中央値[s]", "最小[s]", "最大[s]", "標準偏差[s]", "変動係数", "参照との最大誤差"]],
                hide_index=True)
            st.markdown("#### 3. 結果の読み方")
            for line in explain.benchmark_reading(ok):
                st.markdown(f"- {line}")
            st.markdown("#### 4. 注意点")
            st.markdown("- 時間は、このアプリを動かしているコンピュータ(公開版ではクラウドの共有 CPU)の、この時点の値です。他のアプリの動作や CPU の発熱で変わります。別の PC や GPU には当てはまりません。\n"
                        "- メモリの実測は `tracemalloc`(Python/NumPy の確保量のピーク)で、OS から見たプロセス全体の使用量ではありません。\n"
                        "- 小さい n では時間が Python の準備時間に支配され、n に対して増えません。")
            st.markdown("#### 5. 次に試すとよい操作")
            st.markdown("- 反復回数を増やして、変動係数が下がるか見る。\n- 回路を QFT に変えて比べる(同じ条件で比べる)。\n"
                        "- 並列実行の影響は「④ 並列実行の実測例」を見る。")
            d1, d2, d3 = st.columns(3)
            d1.download_button("結果 CSV", df.to_csv(index=False).encode("utf-8-sig"), f"{rec.run_id}_results.csv", "text/csv")
            d2.download_button("実験条件 (TOML)", config_to_toml(ExperimentConfig(**rec.config)), f"{rec.run_id}_config.toml")
            d3.download_button("実行環境・ログ (JSON)", pd.Series({"environment": rec.environment, "status": rec.status,
                                                                  "duration_s": rec.duration_s, "log": rec.log}).to_json(indent=2),
                               f"{rec.run_id}_job.json")

# ---- ④ 並列実行の実測例
with tabs[4]:
    st.header("④ 並列実行の実測例")
    st.write("複数のジョブを同時に動かす「ワーカー数」と、ジョブ内の計算を分担する「スレッド数」を変えて測った**実測データ**です"
             "(このアプリを作った PC で事前に測定したもの。この画面では計算しません)。")
    terms_box("parallel")
    st.markdown("**ワーカー数** = 同時に動かすジョブの数 / **スレッド数** = 1 つのジョブ内で計算を分担する数。"
                "別々の設定で、効く場所が違います。**どちらも「増やせば必ず速くなる」わけではありません。**")
    data = explain.load_measured_sweep()
    if data is None:
        st.info("同梱の測定データが見つかりません。`python -m qsimbench sweep configs/example_sweep.toml` で作成できます。")
    else:
        meta, s, r = data["meta"], data["summary"], data["rounds"]
        env = meta["environment"]
        nmax = int(r["nmax"].iloc[0])
        st.caption(
            f"測定日 {meta['started_at'][:10]} / CPU {env.get('cpu', '不明')[:40]}(論理 {meta['logical_cores']} コア・物理 "
            f"{meta['physical_cores']} コア) / 回路 {meta['config']['circuit'].upper()} n={meta['config']['qubits']} / "
            f"各条件 {meta['jobs_per_condition']} ジョブ × {meta['rounds']} ラウンド / 条件の順序の seed={meta['sweep_seed']} / "
            f"実行モード: CPU のみ(GPU 未使用)")
        if meta["skipped_conditions"]:
            sk = meta["skipped_conditions"][0]
            st.caption(f"スキップした条件: ワーカー {sk['workers']} × スレッド {sk['threads']}(合計 {sk['workers'] * sk['threads']} > "
                       f"論理コア数 {meta['logical_cores']}。今回の測定における保守的な制限で、これが常に遅いという意味ではありません)")
        g1, g2 = st.columns(2)
        show_plot(charts.sweep_wall_chart(s), "sweep_wall", g1)
        show_plot(charts.sweep_job_chart(s, nmax), "sweep_job", g2)
        st.markdown("左: **スループット**(まとめて終わるまでの時間)。右: **1 ジョブあたり**の計算時間。この 2 つは別の量です。")
        st.dataframe(s.rename(columns={
            "workers": "ワーカー数", "threads": "スレッド数", "rounds": "ラウンド数", "wall_s_median": "経過時間 中央値[s]",
            "wall_s_min": "経過時間 最小[s]", "wall_s_max": "経過時間 最大[s]", "wall_s_cv": "経過時間 変動係数",
            "circuit_nmax_s_median_of_rounds": f"回路時間 n={nmax} 中央値[s]",
            "wall_speedup_median_vs_baseline": "経過時間の短縮率(対 1×1)",
            "wall_range_overlaps_baseline": "範囲が 1×1 と重なる"})[
            ["ワーカー数", "スレッド数", "ラウンド数", "経過時間 中央値[s]", "経過時間 最小[s]", "経過時間 最大[s]",
             "経過時間 変動係数", f"回路時間 n={nmax} 中央値[s]", "経過時間の短縮率(対 1×1)", "範囲が 1×1 と重なる"]],
            hide_index=True)
        st.markdown("#### 結果の読み方")
        for line in explain.sweep_reading(s, nmax):
            st.markdown(f"- {line}")
        st.markdown("#### 言えること・言えないこと")
        st.markdown(
            "- ✅ 測定した PC・この負荷では、ワーカー数を増やすと経過時間は短くなりました(ただし 1 ジョブあたりの計算時間は伸びています)。\n"
            "- ✅ スレッド数の設定は、全ジョブで実際に適用されたことを確認済みです(要求値と `threadpoolctl` の検出値が一致)。\n"
            "- ❌ スレッド数の効果は、ばらつきの範囲内で、有無を断定できません。\n"
            "- ❌ 遅くなる原因(メモリ帯域・ハイパースレッディング・CPU 周波数など)は切り分けていません。\n"
            "- ❌ 「最適な設定」は断定できません。測定は 1 台の PC・1 種類の負荷・3 ラウンドで、外れ値も含みます。"
            "他の PC・GPU・クラスタに当てはまるとは限りません。")
        with st.expander("自分の PC で再現する(コマンド)"):
            st.code("python -m qsimbench sweep configs/example_sweep.toml --workers 1 2 4 --threads 1 2 4 "
                    "--jobs-per-condition 4 --rounds 3 --seed 0", language="powershell")

# ---- ⑤ 数式としくみ
with tabs[5]:
    st.header("⑤ 数式としくみ")
    st.write("各項目は「直感 → 数式 → 記号の意味 → 具体例 → このアプリとの対応」の順です。具体例の数値は、このアプリで実際に計算した値です。")
    terms_box("math")

    st.subheader("5.1 量子状態と状態ベクトル")
    st.markdown("**直感**: 量子ビットが n 個あると、ビット列は 2ⁿ 通りあります。それぞれに複素数(振幅)を 1 つ割り当てた「複素数の並び」が量子状態です。")
    st.latex(C.LATEX["state"])
    st.markdown("- n: 量子ビット数 / |x⟩: ビット列 x に対応する基底状態 / α_x: その複素振幅")
    st.latex(C.LATEX["prob"])
    st.markdown("- |α_x|²: ビット列 x を測定して得る確率 / 合計が 1 になることを「正規化」という")
    st.latex(C.LATEX["one_qubit"])
    h = sv.simulate(circuits.Circuit(1, (circuits.Gate("h", (0,)),)))
    st.latex(C.LATEX["hadamard"])
    st.markdown(f"**具体例(実際に計算)**: |0⟩ に H ゲートをかけた結果の振幅は α={h[0]:.4f}, β={h[1]:.4f}、"
                f"確率は P(0)={abs(h[0])**2:.4f}, P(1)={abs(h[1])**2:.4f}。")
    st.markdown("**アプリとの対応**: 状態は長さ 2ⁿ の複素数配列(`statevector.py`)。配列の番号 x を 2 進数で書いたものがビット列で、"
                "**量子ビット 0 番が左端(最上位ビット)** です。ゲートは `apply_gate`(tensordot)で配列を更新し、"
                "確率は `probabilities`(絶対値の 2 乗)で求めます。")

    st.subheader("5.2 量子ビット数と計算コスト")
    st.markdown("**直感**: n が 1 増えると、並べる複素数が 2 倍になるので、メモリも計算時間も約 2 倍になります。")
    st.latex(C.LATEX["amplitudes"])
    st.latex(C.LATEX["memory"])
    st.markdown("- 16 バイト = 倍精度複素数 1 個(実部 8 + 虚部 8)/ 単位は 1024 基準")
    st.dataframe(explain.memory_table([10, 20, 25, 30]).rename(columns={
        "qubits": "量子ビット数", "amplitudes": "振幅の数", "state_text": "理論メモリ"})[["量子ビット数", "振幅の数", "理論メモリ"]],
        hide_index=True)
    st.markdown("**注意**: 理論値は状態ベクトル本体のみ。実際のピークは約 3 倍(🟢 実測例)。状態ベクトル方式に限った式です。→ 「② 規模とコスト」")

    st.subheader("5.3 GHZ 状態")
    st.markdown("**直感**: 「全部 0」と「全部 1」が半々の重ね合わせ。測定すると、全部そろった結果しか出ません。")
    st.latex(C.LATEX["ghz"])
    g3 = sv.probabilities(sv.simulate(circuits.ghz(3)))
    st.markdown(f"**具体例(n=3 を実際に計算)**: P(`000`)={g3[0]:.4f}、P(`111`)={g3[7]:.4f}、それ以外は合計 {g3[1:7].sum():.4f}。")
    st.markdown("**アプリとの対応**: `circuits.ghz(n)` は、0 番に H、続けて隣り合う量子ビットに CX を連鎖させる回路です。→ 「① 回路を試す」")

    st.subheader("5.4 量子フーリエ変換(QFT)")
    st.markdown("**直感**: 基底状態 |x⟩ を、位相が x に比例して回転する全ビット列の重ね合わせに変えます。古典のフーリエ変換の量子版です。")
    st.latex(C.LATEX["qft"])
    st.markdown("- x: 入力の番号 / y: 出力側の番号 / N=2ⁿ / e^{2πixy/N}: 角度 2πxy/N の複素位相(大きさ 1)")
    q = sv.simulate(circuits.qft(2), 1)
    st.latex(C.LATEX["qft_example"])
    st.markdown("**具体例(n=2, x=1 を実際に計算)**: 振幅 = " + ", ".join(f"{a.real:+.2f}{a.imag:+.2f}i" for a in q) +
                "(式の通り (1, i, −1, −i)/2)。")
    st.markdown("**古典との関係**: 出力ベクトルは、古典の逆離散フーリエ変換(`numpy.fft.ifft`、符号の規約が e^{+i…})と、"
                "係数 √N を除いて同じです。")
    st.latex(C.LATEX["qft_dft"])
    st.markdown("**アプリとの対応**: `circuits.qft(n)` は H と制御位相 CP の並びの最後にビット順を入れ替える SWAP を置いた回路で、"
                "上の式(量子ビット 0 番が最上位ビット)と一致することをテストで確認しています。")

    st.subheader("5.5 ベンチマークの指標")
    for name, text in C.METRICS:
        st.markdown(f"- **{name}**: {text}")
    st.latex(C.LATEX["cv"])
    st.latex(C.LATEX["tvd"])
    st.markdown("TVD(全変動距離)は、測定頻度 p̂ と理論確率 P の差の大きさ(0 に近いほど一致)。")

    st.subheader("5.6 並列実行とスレッド")
    st.markdown(
        "- **ワーカー数**: 独立したジョブを同時に動かす数。スループットを上げる設定。\n"
        "- **スレッド数**: 1 つのジョブの中で BLAS が計算を分担する数。\n"
        "- **資源競合**: 同時に動くものが CPU のコア・キャッシュ・メモリ帯域を取り合うため、1 つあたりは遅くなることがある。\n"
        "- **オーバーヘッド**: プロセスの起動や管理など、計算以外の時間。\n"
        "- **スループット vs 1 ジョブの時間**: 「まとめて終わるまでの時間」と「1 つの計算の時間」は別の量。\n\n"
        "実測の様子は「④ 並列実行の実測例」を見てください。")

# ---- ⑥ 用語集
with tabs[6]:
    st.header("⑥ 用語集")
    q_ = st.text_input("用語を検索", "", key="glossary_q", placeholder="例: 振幅")
    hits = [t for t, (s_, d_) in C.GLOSSARY.items() if q_ in t or q_ in s_ or q_ in d_]
    if not hits:
        st.info("該当する用語がありません。")
    for t in hits:
        with st.expander(t, expanded=bool(q_)):
            st.markdown(f"**{C.GLOSSARY[t][0]}**")
            st.write(C.GLOSSARY[t][1])
