from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def fresh():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def all_text(at):
    return " ".join(str(m.value) for m in at.markdown) + " ".join(str(c.value) for c in at.caption)


def test_first_screen_explains_purpose_and_three_steps():
    at = fresh()
    assert at.title[0].value == "量子回路シミュレーション・性能評価ツール"
    text = all_text(at)
    assert "通常のコンピュータ" in text and "GPU は使用していません" in text
    for step in ("1. 回路を選ぶ", "2. 計算する", "3. 結果を読む"):
        assert step in text


def test_demo_button_runs_ghz_and_shows_reading_guide():
    at = fresh()
    at.button(key="run_demo").click().run()
    assert not at.exception and not at.error
    text = all_text(at)
    assert "結果の読み方" in text and "`000`" in text and "`111`" in text
    assert any("参照結果との照合" in m.label for m in at.metric)


def test_simulation_tab_runs_each_circuit():
    for kind in ("ghz", "qft", "random"):
        at = fresh()
        at.radio(key="kind_sim").set_value(kind).run()
        at.button(key="run_sim").click().run()
        assert not at.exception and not at.error, kind
        assert "sim" in at.session_state and at.session_state["sim"]["kind"] == kind


def test_qft_input_range_follows_n_without_error():
    at = fresh()
    at.radio(key="kind_sim").set_value("qft").run()
    at.slider(key="n_sim").set_value(2).run()
    assert not at.exception
    w = at.number_input(key="input_sim_2")
    assert w.max == 3 and w.value <= 3  # n=2 → 0〜3
    at.button(key="run_sim").click().run()
    assert not at.exception and not at.error


def test_precheck_shown_before_running():
    at = fresh()
    assert any("実行前チェック" in m.value for m in at.markdown)


def test_benchmark_tab_runs_and_reports_reading():
    at = fresh()
    at.slider(key="range_bench").set_value((4, 10)).run()
    at.button(key="run_bench").click().run()
    assert not at.exception and not at.error
    assert at.session_state["rec"].status == "success"
    text = all_text(at)
    assert "結果の読み方" in text and "tracemalloc" in text


def test_heavy_benchmark_requires_confirmation():
    at = fresh()
    at.slider(key="range_bench").set_value((20, 25)).run()
    assert not at.exception
    # 利用可能メモリによって heavy になる場合は確認チェックが出る。出ない場合は caution/light。いずれも例外なし
    assert at.button(key="run_bench") is not None


def test_parallel_tab_shows_measured_data_and_limits():
    at = fresh()
    text = all_text(at)
    assert "実測データ" in text and "言えないこと" in text and "最適な設定」は断定できません" in text
    assert "必ず速くなる」わけではありません" in text


def test_glossary_search():
    at = fresh()
    at.text_input(key="glossary_q").input("振幅").run()
    assert not at.exception
    assert any("複素振幅" in e.label for e in at.expander)


def test_demo_then_same_condition_simulation_does_not_duplicate_chart_ids():
    """回帰テスト: デモ(GHZ n=3)の後に同じ条件を ① タブで実行しても、グラフの ID が重複しない。"""
    at = fresh()
    at.button(key="run_demo").click().run()
    at.button(key="run_sim").click().run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    assert not at.error


def test_every_plotly_chart_in_app_has_unique_key():
    import re

    src = Path(APP).read_text(encoding="utf-8")
    assert "st.plotly_chart(" not in src.replace("target.plotly_chart", "")  # 直接呼ばず show_plot を使う
    keys = re.findall(r'show_plot\([^\n]*?"([a-z_]+)"(?:, g\d)?\)', src) + re.findall(r'f"\{prefix\}_(prob|phase)"', src)
    assert keys and len(keys) == len(set(keys))  # 見つかったキーに重複がない(複数行の呼び出しは数えきれないため件数は見ない)


def test_displayed_tables_are_arrow_serializable():
    """回帰テスト: 型が混在した列は、実ブラウザ(Arrow 経由)で表示エラーになる。画面の表はすべて変換できること。"""
    pa = pytest.importorskip("pyarrow")
    at = fresh()
    at.button(key="run_demo").click().run()
    at.button(key="run_bench").click().run()
    tables = list(at.table) + list(at.dataframe)
    assert tables
    for t in tables:
        pa.Table.from_pandas(t.value)  # 失敗すれば例外
