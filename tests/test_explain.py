"""説明・数式・UI部品が、実際の実装と一致していることのテスト。"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsimbench import circuits, explain, reference, statevector as sv
from qsimbench.jobs import MAX_QUBITS

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))
pytest.importorskip("plotly")
import charts  # noqa: E402
import content  # noqa: E402


# ---- メモリ表(数式 16·2^n)---------------------------------------------------
def test_format_bytes_uses_1024_units():
    assert explain.format_bytes(16 * 2**10) == "16 KiB"
    assert explain.format_bytes(16 * 2**20) == "16 MiB"
    assert explain.format_bytes(16 * 2**25) == "512 MiB"
    assert explain.format_bytes(16 * 2**30) == "16 GiB"
    assert explain.format_bytes(512) == "512 B"


def test_memory_table_matches_statevector_and_spec():
    t = explain.memory_table([10, 20, 25, 30])
    assert t.amplitudes.tolist() == [1024, 1048576, 33554432, 1073741824]
    assert t.state_text.tolist() == ["16 KiB", "16 MiB", "512 MiB", "16 GiB"]
    assert (t.state_bytes_theory == [sv.statevector_bytes(n) for n in t.qubits]).all()
    assert (t.state_bytes_theory == 16 * t.amplitudes).all()


def test_theory_formula_matches_actual_array():
    n = 12
    assert sv.simulate(circuits.ghz(n)).nbytes == sv.statevector_bytes(n)  # complex128 配列の実サイズ


# ---- 負荷見積もり・入力検証 --------------------------------------------------
def test_estimate_load_levels():
    big = 64 * 2**30
    assert explain.estimate_load(10, big)["level"] == "light"
    assert explain.estimate_load(20, big)["level"] == "caution"
    assert explain.estimate_load(24, 2**30)["level"] == "heavy"  # ピーク推定が空きの 50% 超
    assert explain.estimate_load(MAX_QUBITS + 1, big)["level"] == "blocked"
    e = explain.estimate_load(20, big)
    assert e["peak_estimate_bytes"] == 3 * e["theory_bytes"] and e["amplitudes"] == 2**20


def test_validate_circuit_inputs():
    assert explain.validate_circuit_inputs("qft", 3, 7, 4) == ([], [])
    errs, _ = explain.validate_circuit_inputs("qft", 3, 8, 4)
    assert errs and "0〜7" in errs[0]
    assert explain.validate_circuit_inputs("ghz", MAX_QUBITS + 1, 0, 4)[0]
    assert explain.validate_circuit_inputs("random", 3, 0, 0)[0]
    assert explain.validate_circuit_inputs("ghz", 3, 1, 4)[1]  # 警告のみ


# ---- 結果の読み方が実際の数値と一致 -----------------------------------------
def test_reading_guide_ghz_uses_real_probabilities():
    p = sv.probabilities(sv.simulate(circuits.ghz(3)))
    text = " ".join(explain.reading_guide("ghz", 3, p))
    assert "`000`" in text and "`111`" in text and "0.5000" in text


def test_reading_guide_qft_uniform():
    p = sv.probabilities(sv.simulate(circuits.qft(3), 5))
    text = " ".join(explain.reading_guide("qft", 3, p, 5))
    assert "1/8" in text and "位相" in text


def test_top_outcomes_sorted_and_filtered():
    p = np.array([0.1, 0.0, 0.6, 0.3])
    assert explain.top_outcomes(p, 10).tolist() == [2, 3, 0]


# ---- 数式とコードの整合(画面に載せる数式の検証) -------------------------------
@pytest.mark.parametrize("n,x", [(2, 1), (3, 5), (4, 9)])
def test_qft_formula_matches_simulator(n, x):
    """|x> -> (1/√N) Σ_y e^{2πi xy/N}|y> が、実際の回路の出力と一致する。"""
    N = 2**n
    formula = np.array([np.exp(2j * np.pi * x * y / N) for y in range(N)]) / np.sqrt(N)
    assert np.allclose(sv.simulate(circuits.qft(n), x), formula, atol=1e-12)


@pytest.mark.parametrize("n,x", [(3, 2), (5, 7)])
def test_qft_equals_sqrtN_times_ifft(n, x):
    N = 2**n
    e = np.zeros(N)
    e[x] = 1
    assert np.allclose(sv.simulate(circuits.qft(n), x), np.sqrt(N) * np.fft.ifft(e), atol=1e-12)


def test_qft_example_on_screen_n2_x1():
    q = sv.simulate(circuits.qft(2), 1)
    assert np.allclose(q, np.array([1, 1j, -1, -1j]) / 2)  # 画面の具体例 (1/2)(|0>+i|1>-|2>-i|3>)


def test_ghz_formula_and_screen_example():
    for n in (2, 3, 5):
        psi = sv.simulate(circuits.ghz(n))
        expect = np.zeros(2**n, dtype=complex)
        expect[0] = expect[-1] = 1 / np.sqrt(2)
        assert np.allclose(psi, expect)
    assert reference.max_abs_error(sv.simulate(circuits.ghz(4)), reference.analytic_reference("ghz", 4)) < 1e-12


def test_hadamard_one_qubit_example():
    h = sv.simulate(circuits.Circuit(1, (circuits.Gate("h", (0,)),)))
    assert np.allclose(np.abs(h) ** 2, [0.5, 0.5]) and np.isclose((np.abs(h) ** 2).sum(), 1)


def test_bit_order_qubit0_is_leftmost():
    # 0 番を X で反転 → ビット列 '100'(インデックス 4)。用語集・5.1 の説明と一致
    p = sv.probabilities(sv.simulate(circuits.Circuit(3, (circuits.Gate("x", (0,)),))))
    assert p.argmax() == 4 and sv.bitstring(4, 3) == "100"


# ---- 用語集・数式の網羅 -------------------------------------------------------
REQUIRED_TERMS = ["量子ビット", "量子状態", "重ね合わせ", "複素振幅", "測定確率", "量子ゲート", "量子回路", "GHZ状態",
                  "QFT", "状態ベクトルシミュレーション", "ベンチマーク", "ワーカー", "スレッド", "並列実行",
                  "メモリ使用量", "参照結果", "数値誤差", "変動係数"]


def test_glossary_has_all_required_terms_with_text():
    for t in REQUIRED_TERMS:
        hits = [k for k in content.GLOSSARY if t in k]
        assert hits, t
        short, long_ = content.GLOSSARY[hits[0]]
        assert len(short) > 10 and len(long_) > 10


def test_terms_by_tab_reference_existing_terms():
    for names in content.TERMS_BY_TAB.values():
        assert all(n in content.GLOSSARY for n in names)


def test_latex_formulas_present_and_balanced():
    for key in ("state", "prob", "one_qubit", "amplitudes", "memory", "ghz", "qft", "cv"):
        s = content.LATEX[key]
        assert s.count("{") == s.count("}") and s.strip()
    assert "2^n" in content.LATEX["amplitudes"] and "16" in content.LATEX["memory"]
    assert re.search(r"e\^\{2\\pi i", content.LATEX["qft"])


def test_content_does_not_claim_gpu_or_cluster_results():
    text = content.SCOPE_NOTE + content.INTRO
    assert "GPU" in text and "未実施" in text
    for bad in ("GPU で高速", "クラスタで実行済み", "Slurm で実行済み"):
        assert bad not in text


# ---- 読み方の文章が実データに基づく ---------------------------------------------
def make_bench_df():
    return pd.DataFrame({"qubits": [10, 12, 14], "time_mean_s": [0.01, 0.04, 0.17], "time_cv": [0.2, 0.05, 0.03],
                         "time_std_s": [0.002, 0.002, 0.005], "time_median_s": [0.01, 0.04, 0.17],
                         "peak_mem_bytes_measured": [3 * 16384, 3 * 65536, 3 * 262144],
                         "statevector_bytes_theory": [16384, 65536, 262144],
                         "max_abs_error_vs_reference": [1e-16, 2e-16, np.nan]})


def test_benchmark_reading_numbers():
    text = " ".join(explain.benchmark_reading(make_bench_df()))
    assert "3.00 倍" in text  # メモリ実測/理論
    assert "約 2.06 倍" in text  # (0.17/0.04)^(1/2)
    assert "20.0%" in text and "n=10" in text  # 最大CV


def test_sweep_reading_uses_data_and_checks_overlap():
    s = explain.load_measured_sweep()["summary"]
    lines = explain.sweep_reading(s, 18)
    t1 = s[s.threads == 1].set_index("workers")
    assert any(f"{t1.loc[1, 'wall_s_median']:.1f}→{t1.loc[2, 'wall_s_median']:.1f}" in l for l in lines)
    assert any("重なって" in l for l in lines)  # 実データでは threads 間の範囲が重なる


def test_measured_sweep_loads_and_matches_files():
    d = explain.load_measured_sweep()
    assert d is not None and len(d["summary"]) == 8 and len(d["rounds"]) == 24
    pd.testing.assert_frame_equal(d["summary"], pd.read_csv(explain.MEASURED_SWEEP_DIR / "summary.csv"))
    assert d["meta"]["environment"]["gpu_used"] is False
    assert explain.load_measured_sweep(Path("/nonexistent/dir")) is None


# ---- グラフ: プロットされる値が元データと一致 ---------------------------------
def test_time_and_memory_charts_use_dataframe_values():
    df = make_bench_df()
    f = charts.time_chart(df)
    assert list(f.data[0].y) == df.time_mean_s.tolist() and list(f.data[0].x) == df.qubits.tolist()
    assert f.layout.xaxis.title.text and f.layout.yaxis.title.text and f.layout.title.text
    m = charts.memory_chart(df)
    assert list(m.data[0].y) == df.peak_mem_bytes_measured.tolist()
    assert list(m.data[1].y) == df.statevector_bytes_theory.tolist()
    assert len({tr.name for tr in m.data}) == 2  # 凡例の名前がある


def test_sweep_charts_match_summary():
    s = explain.load_measured_sweep()["summary"]
    f = charts.sweep_wall_chart(s)
    for tr in f.data:
        t = int(tr.name.split()[-1])
        g = s[s.threads == t].sort_values("workers")
        assert list(tr.y) == g.wall_s_median.tolist()
        assert np.allclose(np.array(tr.error_y.array) + np.array(g.wall_s_median), g.wall_s_max)
    assert len({tr.marker.symbol for tr in f.data}) == len(f.data)  # 色以外(形)でも区別できる


def test_probability_chart_values():
    f = charts.probability_chart(["00", "11"], [0.5, 0.5], [0.49, 0.51], 100)
    assert list(f.data[0].y) == [0.5, 0.5] and list(f.data[1].y) == [0.49, 0.51]
    assert f.data[1].marker.pattern.shape == "/"  # 模様でも区別


def test_requirements_split_app_vs_dev():
    root = Path(__file__).resolve().parents[1]
    # requirements*.txt は pip がロケールで読むため ASCII のみであること(decode("ascii") が検証を兼ねる)
    names = lambda p: {re.split(r"[<>=]", l)[0].strip() for l in (root / p).read_bytes().decode("ascii").splitlines()  # noqa: E731
                       if l.strip() and not l.startswith(("#", "-r"))}
    app, dev = names("requirements.txt"), names("requirements-dev.txt")
    assert not {"pytest", "qiskit"} & app  # クラウド公開で使う requirements.txt に重いテスト用依存を入れない
    assert {"pytest", "qiskit"} <= dev
    assert "-r requirements.txt" in (root / "requirements-dev.txt").read_text(encoding="ascii")
    assert {"streamlit", "plotly", "numpy", "pandas", "psutil", "threadpoolctl"} <= app


def test_readme_glossary_in_sync_with_app_content():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- GLOSSARY:START -->")[1].split("<!-- GLOSSARY:END -->")[0]
    for term, (short, _) in content.GLOSSARY.items():
        assert f"**{term}** — {short}" in block, term


def test_readme_honesty_statements_present():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "GPU は使用していません" in readme
    import re as _re
    shots = _re.findall(r"\]\((docs/screenshots/[^)]+\.png)\)", readme)
    assert len(shots) >= 5  # 掲載している画像は、すべて実在するファイル(リンク切れなし)
    for rel in shots:
        head = (Path(__file__).resolve().parents[1] / rel).read_bytes()[:4]
        assert head == bytes([137]) + b"PNG", rel  # PNG signature
    assert "https://quantum-sim-bench-rm5zlvzcacm43umvgc9quk.streamlit.app/" in readme and "MIT" in readme
    assert "start_app.bat" in readme and "requirements-dev.txt" in readme


def test_bitstring_axis_is_categorical_so_leading_zeros_are_kept():
    """回帰テスト: 横軸ラベル "000" が数値 0 と解釈されて桁が消えない(画面で「0」と表示された不具合)。"""
    f = charts.probability_chart(["000", "111"], [0.5, 0.5], [0.49, 0.51], 100)
    assert f.layout.xaxis.type == "category"
    assert list(f.data[0].x) == ["000", "111"] and list(f.layout.xaxis.categoryarray) == ["000", "111"]


def test_glossary_additions_present():
    for t in ("基底状態・ビット列", "位相", "正規化", "ショット（測定回数）", "BLAS", "スループット", "資源競合",
              "論理コア・物理コア", "tracemalloc", "解析解", "TVD（全変動距離）", "KiB・MiB・GiB", "一時配列"):
        assert t in content.GLOSSARY, t
    assert len(content.GLOSSARY) >= 40


def test_bitstring_axis_is_categorical_so_leading_zeros_are_kept():
    """回帰テスト: 横軸ラベル "000" が数値 0 と解釈されて桁が消えない(画面で「0」と表示された不具合)。"""
    f = charts.probability_chart(["000", "111"], [0.5, 0.5], [0.49, 0.51], 100)
    assert f.layout.xaxis.type == "category"
    assert list(f.data[0].x) == ["000", "111"] and list(f.layout.xaxis.categoryarray) == ["000", "111"]


def test_glossary_additions_present():
    for t in ("基底状態・ビット列", "位相", "正規化", "ショット（測定回数）", "BLAS", "スループット", "資源競合",
              "論理コア・物理コア", "tracemalloc", "解析解", "TVD（全変動距離）", "KiB・MiB・GiB", "一時配列"):
        assert t in content.GLOSSARY, t
    assert len(content.GLOSSARY) >= 40


def test_top_outcomes_ties_ordered_by_bitstring_index():
    """回帰テスト: 同じ確率(GHZ の 000 と 111)は番号の小さい順 = 000 が先(公開画面では 111 が先だった)。"""
    p = sv.probabilities(sv.simulate(circuits.ghz(3)))
    assert explain.top_outcomes(p, 8).tolist() == [0, 7]


def test_no_environment_ambiguous_pc_wording_in_ui():
    src = (APP_DIR / "streamlit_app.py").read_text(encoding="utf-8") + (APP_DIR / "charts.py").read_text(encoding="utf-8")
    assert "このPCの搭載メモリ" not in src  # 公開版ではクラウドのサーバー。作者の PC と誤解させない


def test_log_axis_ticks_are_powers_of_ten():
    df = make_bench_df()
    ax = charts.time_chart(df).layout.yaxis
    assert ax.type == "log" and ax.dtick == 1 and ax.exponentformat == "power"
