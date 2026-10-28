"""実験設定とジョブ実行。UI・CLI のどちらからも同じ関数を呼ぶ。

``execute(config)`` は例外を投げず、成功・失敗とも ``JobRecord`` を返す。
将来 Slurm 等から呼ぶ場合も「設定ファイル → execute → results/<run_id>/」の形を保つ。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
import tomllib
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import circuits, reference, statevector as sv
from .benchmark import environment_info, measure_peak_memory, measure_time

MAX_QUBITS = 25  # 状態1本 512 MiB、ピーク約1.5 GiB。これ以上は拒否する
SUPPORTED_BACKENDS = ("numpy-cpu",)


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    backend: str = "numpy-cpu"
    circuit: str = "ghz"
    qubits: list[int] = field(default_factory=lambda: [4, 8, 12])
    depth: int = 4  # random 回路のみ
    input_value: int = 0  # 初期基底状態(整数)。qft の入力に使う
    shots: int = 10000
    seed: int = 0
    warmup: int = 1
    repeats: int = 5
    measure_memory: bool = True
    verify: bool = True

    def validate(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"backend {self.backend!r} is not implemented (available: {SUPPORTED_BACKENDS}). "
                "GPU backends are not implemented."
            )
        if self.circuit not in circuits.CIRCUIT_KINDS:
            raise ValueError(f"circuit must be one of {circuits.CIRCUIT_KINDS}")
        if not self.qubits or any(not 1 <= n <= MAX_QUBITS for n in self.qubits):
            raise ValueError(f"qubits must be non-empty with 1 <= n <= {MAX_QUBITS}")
        if self.shots < 1 or self.repeats < 1 or self.warmup < 0 or self.depth < 1:
            raise ValueError("shots, repeats, depth must be >= 1 and warmup >= 0")
        if any(not 0 <= self.input_value < 2**n for n in self.qubits):
            raise ValueError("input_value must fit in the smallest qubit count")

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = ExperimentConfig(**data)
    cfg.validate()
    return cfg


def config_to_toml(cfg: ExperimentConfig) -> str:
    lines = []
    for k, v in asdict(cfg).items():
        if isinstance(v, bool):
            s = "true" if v else "false"
        elif isinstance(v, str):
            s = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, list):
            s = "[" + ", ".join(str(x) for x in v) + "]"
        else:
            s = str(v)
        lines.append(f"{k} = {s}")
    return "\n".join(lines) + "\n"


def run_point(cfg: ExperimentConfig, n: int) -> dict:
    """1つの量子ビット数 n について 検証・サンプリング・計測 を行い、1行の結果を返す。"""
    circ = circuits.build(cfg.circuit, n, cfg.depth, cfg.seed)
    run = lambda: sv.simulate(circ, cfg.input_value)  # noqa: E731
    state = run()
    probs = sv.probabilities(state)
    row: dict = {
        # 実験条件(CSV単体で条件が分かるようにする)
        "circuit": cfg.circuit, "depth": cfg.depth, "input_value": cfg.input_value,
        "seed": cfg.seed, "warmup": cfg.warmup, "repeats": cfg.repeats,
        "status": "ok",
        "qubits": n,
        "gates": len(circ),
        "shots": cfg.shots,
        "norm_error": float(abs(probs.sum() - 1.0)),
        "statevector_bytes_theory": sv.statevector_bytes(n),
    }

    counts = sv.sample_counts(probs, cfg.shots, cfg.seed)
    row["tvd_sampling_vs_sim"] = reference.total_variation_distance(counts / cfg.shots, probs)
    # 有限ショットの期待TVDの目安(理論): 非ゼロ確率状態数 k に対し ~ sqrt(k/(2π·shots))
    k = int(np.count_nonzero(probs > 1e-12))
    row["tvd_expected_scale_theory"] = float(np.sqrt(k / (2 * np.pi * cfg.shots)))

    if cfg.verify:
        ref_kind, ref = None, reference.analytic_reference(cfg.circuit, n, cfg.input_value)
        if ref is not None:
            ref_kind = "analytic"
        elif n <= reference.DENSE_MAX_QUBITS:
            ref_kind, ref = "dense", reference.dense_reference(circ, cfg.input_value)
        row["reference_kind"] = ref_kind or "none"
        row["max_abs_error_vs_reference"] = (
            reference.max_abs_error(state, ref) if ref is not None else float("nan")
        )
    del state, probs

    t = measure_time(run, cfg.warmup, cfg.repeats)
    row.update(
        time_mean_s=t["mean"], time_std_s=t["std"], time_min_s=t["min"],
        time_median_s=t["median"], time_max_s=t["max"], time_sem_s=t["sem"], time_cv=t["cv"],
        times_s=json.dumps(t["times"]),
        gates_per_s=len(circ) / t["mean"] if t["mean"] > 0 else float("nan"),
    )
    if cfg.measure_memory:
        row["peak_mem_bytes_measured"] = measure_peak_memory(run)
    return row


@dataclass
class JobRecord:
    run_id: str
    status: str  # "success" | "failed"
    exit_code: int
    started_at: str
    duration_s: float
    config: dict
    config_digest: str
    environment: dict
    error: str | None = None
    out_dir: str | None = None
    log: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def execute(cfg: ExperimentConfig, out_root: str | Path | None = "results") -> JobRecord:
    """設定を実行して JobRecord を返す。out_root=None なら保存しない(UI用)。"""
    started = datetime.now()
    run_id = f"{started:%Y%m%d-%H%M%S}-{cfg.name}-{uuid.uuid4().hex[:6]}"
    log: list[str] = []

    def say(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        log.append(line)
        logging.getLogger("qsimbench").info(msg)

    t0 = time.perf_counter()
    rows: list[dict] = []
    cur: int | None = None
    status, code, error = "success", 0, None
    try:
        cfg.validate()
        say(f"start run_id={run_id} circuit={cfg.circuit} qubits={cfg.qubits} backend={cfg.backend}")
        for n in cfg.qubits:
            cur = n
            say(f"n={n}: running")
            rows.append(run_point(cfg, n))
            say(f"n={n}: done mean={rows[-1]['time_mean_s']:.4g}s")
    except MemoryError:
        status, code, error = "failed", 3, "MemoryError: " + traceback.format_exc()
        say("FAILED: out of memory")
    except Exception:  # noqa: BLE001 - 失敗もレコードとして残す
        status, code, error = "failed", 1, traceback.format_exc()
        say("FAILED: " + error.strip().splitlines()[-1])
    if status == "failed" and cur is not None:  # 失敗した点もCSVに1行残す
        rows.append({"circuit": cfg.circuit, "seed": cfg.seed, "shots": cfg.shots, "qubits": cur,
                     "status": "failed", "error": error.strip().splitlines()[-1]})
    duration = time.perf_counter() - t0
    say(f"finish status={status} duration={duration:.3f}s")

    rec = JobRecord(
        run_id=run_id, status=status, exit_code=code,
        started_at=started.isoformat(timespec="seconds"), duration_s=duration,
        config=asdict(cfg), config_digest=cfg.digest(), environment=environment_info(),
        error=error, log=log, rows=rows,
    )
    if out_root is not None:
        rec.out_dir = str(save(rec, out_root))
    return rec


def save(rec: JobRecord, out_root: str | Path) -> Path:
    """results/<run_id>/ を新規作成して書き出す。既存ディレクトリは上書きせず連番を付ける。"""
    base = Path(out_root) / rec.run_id
    out, i = base, 1
    while out.exists():
        i += 1
        out = Path(f"{base}-{i}")
    out.mkdir(parents=True)
    (out / "config.toml").write_text(config_to_toml(ExperimentConfig(**rec.config)), encoding="utf-8")
    meta = {k: v for k, v in asdict(rec).items() if k not in ("rows", "log")}
    (out / "job.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run.log").write_text("\n".join(rec.log) + "\n", encoding="utf-8")
    if rec.rows:
        rec.dataframe().to_csv(out / "results.csv", index=False)
    return out
