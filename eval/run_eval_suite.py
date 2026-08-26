"""Run paper-style Tables 1–7 eval; write under eval/runs/<timestamp>/.

Usage (project root, venv active):
  pip install -e .
  python eval/run_eval_suite.py
  .\\eval\\run_eval_suite.ps1

Paper protocol: 12 envs × 12 episodes (EPISODES_PER_ENV). Default variants: base + lora/all.
Estimated wall time (RTX 3060, 0.5B 4-bit): ~3.5–4 h per variant → ~7–8 h for base+all.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lora.gpu_env  # noqa: F401

import torch

from config import (
    ALL_GAMES,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    DPO_BETA,
    EPISODES_PER_ENV,
    EVAL_GAMES,
    EVAL_MAX_TOKENS,
    EVAL_SEED,
    EVAL_DEFAULT_VARIANTS,
    EVAL_DIR,
    EVAL_RUNS_DIR,
    EVAL_EST_SECONDS_PER_VARIANT,
    GAME_ABBREV,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_DIR,
    LORA_R,
    MERGE_ALPHA,
    MODEL_ID,
    NUM_EPOCHS,
    PAPER_MODEL_ID,
    PAPER_TABLE1,
    PAPER_TABLE2,
    PAPER_TABLE3,
    PAPER_TABLE4,
    PAPER_TABLE5,
    PAPER_TABLE6,
    PAPER_TABLE7,
    PAPER_LORA_ALPHA,
    PAPER_LORA_R,
    PAPER_LORA_TARGET_MODULES,
    TRAINING_VARIANTS,
    USE_4BIT,
    VARIANT_LABELS,
)
from eval.metrics import metrics_to_dict
from eval.paths import new_run_dir, resolve_run_dir, utc_stamp, write_latest_pointer
from eval.progress import EvalLogger
from eval.run_title import build_run_title, infer_train_epochs
from eval.run_table import eval_variant, fmt, fmt_cell, save_json


def _discover_variants(ckpt_root: Path) -> list[str]:
    from eval.checkpoints import discover_all_adapters

    found = ["base"]
    for name in sorted(discover_all_adapters(lora_dir=ckpt_root).keys()):
        if name not in found:
            found.append(name)
    if not ckpt_root.is_dir():
        return found
    for path in sorted(ckpt_root.iterdir()):
        if not path.is_dir() or path.name.startswith("checkpoint-"):
            continue
        if (path / "adapter_config.json").exists() and path.name not in found:
            found.append(path.name)
        nested = path / "adapter"
        if (nested / "adapter_config.json").exists() and path.name not in found:
            found.append(path.name)
    return found


def _paper_refs() -> dict:
    return {
        "table1": PAPER_TABLE1,
        "table2": PAPER_TABLE2,
        "table3": PAPER_TABLE3,
        "table4": PAPER_TABLE4,
        "table5": PAPER_TABLE5,
        "table6": PAPER_TABLE6,
        "table7": PAPER_TABLE7,
    }


def _hyperparams(args: argparse.Namespace) -> dict:
    model_id = args.model_id or (PAPER_MODEL_ID if args.paper else MODEL_ID)
    lora_r = args.lora_r if args.lora_r is not None else (PAPER_LORA_R if args.paper else LORA_R)
    lora_alpha = (
        args.lora_alpha
        if args.lora_alpha is not None
        else (PAPER_LORA_ALPHA if args.paper else LORA_ALPHA)
    )
    ckpt = str(args.checkpoint_dir)
    train_epochs = infer_train_epochs(args.checkpoint_dir)
    return {
        "model_id": model_id,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": LORA_DROPOUT,
        "dpo_beta": DPO_BETA,
        "merge_alpha": MERGE_ALPHA,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "train_epochs": train_epochs,
        "episodes_per_env": args.episodes,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "checkpoint_dir": ckpt,
        "variants_evaluated": args.variants,
        "paper_mode": args.paper,
        "use_4bit": USE_4BIT,
        "quantization": "4bit" if USE_4BIT else "none",
        "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE if USE_4BIT else None,
        "bnb_4bit_compute_dtype": BNB_4BIT_COMPUTE_DTYPE if USE_4BIT else None,
        "run_title": build_run_title(
            {
                "model_id": model_id,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "train_epochs": train_epochs,
                "epochs": NUM_EPOCHS,
                "episodes_per_env": args.episodes,
                "checkpoint_dir": ckpt,
            }
        ),
        "paper_title": build_run_title({}, kind="paper"),
        "paper_protocol": {
            "envs": len(ALL_GAMES),
            "episodes_per_env": args.episodes,
            "matches_paper": args.episodes == EPISODES_PER_ENV and not args.paper,
        },
    }


def _training_manifest(ckpt_root: Path) -> dict:
    manifest = {}
    for name, spec in TRAINING_VARIANTS.items():
        ckpt = ckpt_root / name if name != "merge" else ckpt_root / "merge"
        manifest[name] = {
            **spec,
            "checkpoint": str(ckpt),
            "exists": ckpt.exists()
            and (name == "merge" or (ckpt / "adapter_config.json").exists()),
        }
    return manifest


def _format_table1_local(rows: dict[str, dict]) -> list[str]:
    lines = [
        f"{'Model':<32} {'fc(ID)':>8} {'fc(HO)':>8} {'CR(ID)':>10} {'CR(HO)':>10}",
        "-" * 72,
    ]
    for variant, m in rows.items():
        label = VARIANT_LABELS.get(variant, variant)
        lines.append(
            f"{label:<32} {m['fc_id']:>8.2f} {m['fc_ho']:>8.2f} "
            f"{fmt(m['cr_id']):>10} {fmt(m['cr_ho']):>10}"
        )
    return lines


def _format_table1_paper(paper: dict) -> list[str]:
    lines = [
        f"{'Model (paper)':<32} {'fc(ID)':>8} {'fc(HO)':>8} {'CR(ID)':>10} {'CR(HO)':>10}",
        "-" * 72,
    ]
    for variant, r in paper.items():
        if variant == "haiku":
            continue
        label = VARIANT_LABELS.get(variant, variant)
        lines.append(
            f"{label:<32} {r['fc_id']:>8.2f} {r['fc_ho']:>8.2f} "
            f"{fmt(r['cr_id']):>10} {fmt(r['cr_ho']):>10}"
        )
    return lines


def _format_table2_block(rows: dict[str, dict], variants: list[str]) -> list[str]:
    header = f"{'Game':<22}" + "".join(f" {v:>8}" for v in variants)
    lines = [header, "-" * len(header)]
    for game in ALL_GAMES:
        line = f"{game:<22}"
        for v in variants:
            cr = rows.get(v, {}).get("per_game_cr", {}).get(game, 0.0)
            line += f" {cr:>8.2f}"
        lines.append(line)
    return lines


def _format_table2_paper(variants: list[str]) -> list[str]:
    header = f"{'Game':<22}" + "".join(f" {v:>8}" for v in variants if v != "haiku")
    lines = [header, "-" * len(header)]
    for game in ALL_GAMES:
        line = f"{game:<22}"
        for v in variants:
            if v == "haiku":
                continue
            val = PAPER_TABLE2.get(game, {}).get(v)
            line += f" {val if val is not None else 'n/a':>8}"
        lines.append(line)
    return lines


def _format_table3_block(rows: dict[str, dict], variants: list[str]) -> list[str]:
    axes = sorted({k for v in rows.values() for k in v.get("per_axis_cr", {})})
    if not axes:
        axes = list(next(iter(PAPER_TABLE3.values())).keys()) if PAPER_TABLE3 else []
    header = f"{'Axis':<18}" + "".join(f" {v:>8}" for v in variants)
    lines = [header, "-" * len(header)]
    for axis in axes:
        line = f"{axis:<18}"
        for v in variants:
            cr = rows.get(v, {}).get("per_axis_cr", {}).get(axis, 0.0)
            line += f" {cr:>8.2f}"
        lines.append(line)
    return lines


def _format_table3_paper(variants: list[str]) -> list[str]:
    axes = list(PAPER_TABLE3.keys())
    header = f"{'Axis (paper)':<18}" + "".join(f" {v:>8}" for v in variants if v != "haiku")
    lines = [header, "-" * len(header)]
    for axis in axes:
        line = f"{axis:<18}"
        for v in variants:
            if v == "haiku":
                continue
            val = PAPER_TABLE3.get(axis, {}).get(v)
            line += f" {val if val is not None else 'n/a':>8}"
        lines.append(line)
    return lines


def _format_table4_block(rows: dict[str, dict], variants: list[str]) -> list[str]:
    abbrevs = [GAME_ABBREV[g] for g in ALL_GAMES]
    header = f"{'Model':<16}" + "".join(f" {a:>6}" for a in abbrevs)
    lines = [header, "-" * len(header)]
    for v in variants:
        line = f"{v:<16}"
        fc_map = rows.get(v, {}).get("per_game_fc", {})
        for game in ALL_GAMES:
            line += f" {fmt_cell(fc_map.get(game)):>6}"
        lines.append(line)
    return lines


def _format_table4_paper() -> list[str]:
    abbrevs = [GAME_ABBREV[g] for g in ALL_GAMES]
    header = f"{'Model (paper)':<16}" + "".join(f" {a:>6}" for a in abbrevs)
    lines = [header, "-" * len(header)]
    for model, row in PAPER_TABLE4.items():
        line = f"{model:<16}"
        for a in abbrevs:
            line += f" {fmt_cell(row.get(a)):>6}"
        lines.append(line)
    return lines


def _format_table5_block(manifest: dict) -> list[str]:
    lines = [
        f"{'Variant':<14} {'Pairs(paper)':>12} {'Checkpoint':>22} {'Trained':>8}",
        "-" * 60,
    ]
    for name, spec in manifest.items():
        exists = "OK" if spec.get("exists") else "MISSING"
        pairs = spec.get("pairs", 0)
        ckpt = Path(spec.get("checkpoint", "")).name or "-"
        lines.append(f"{name:<14} {pairs:>12} {ckpt:>22} {exists:>8}")
    lines += [
        "",
        "논문 Table 5 best step / eval loss (참조):",
        f"{'Variant':<14} {'Pairs':>8} {'Best step':>10} {'Eval loss':>10}",
        "-" * 46,
    ]
    for v, r in PAPER_TABLE5.items():
        step = r["best_step"] if r["best_step"] is not None else "-"
        loss = f"{r['best_eval_loss']:.4f}" if r["best_eval_loss"] is not None else "-"
        lines.append(f"{v:<14} {r['pairs']:>8} {str(step):>10} {loss:>10}")
    return lines


def _format_table6_block() -> list[str]:
    lines = [
        f"{'round':>6} {'TFT':>8} {'blind':>8} {'oracle':>12} {'helps':>8}",
        "-" * 46,
    ]
    for row in PAPER_TABLE6:
        oracle = row["oracle"] + (" (HELPS)" if row["helps"] else "")
        lines.append(
            f"{row['round']:>6} {row['tft']:>8} {row['blind']:>8} {oracle:>12} "
            f"{str(row['helps']):>8}"
        )
    return lines


def _format_table7_block(rows: dict[str, dict], variant: str) -> list[str]:
    m = rows.get(variant, {})
    lines = [
        f"{'env':<22} {'ac':>8} {'fc':>8} {'exploit':>8} {'paper_fc':>8}",
        "-" * 58,
    ]
    for game in ALL_GAMES:
        ref = PAPER_TABLE7[game]
        lines.append(
            f"{game:<22} {fmt_cell(m.get('per_game_ac', {}).get(game)):>8} "
            f"{fmt_cell(m.get('per_game_fc', {}).get(game)):>8} "
            f"{fmt_cell(m.get('per_game_exploitability', {}).get(game)):>8} "
            f"{fmt_cell(ref['fc']):>8}"
        )
    return lines


def build_result_md(
    *,
    suite: dict,
    rows: dict[str, dict],
    ok_variants: list[str],
    manifest: dict,
    run_dir: Path,
    run_id: str,
) -> str:
    hp = suite["hyperparams"]
    local_title = hp.get("run_title") or build_run_title(hp, kind="local")
    paper_title = hp.get("paper_title") or build_run_title(hp, kind="paper")
    try:
        run_rel = run_dir.relative_to(ROOT)
    except ValueError:
        run_rel = run_dir
    est_h = len(ok_variants) * EVAL_EST_SECONDS_PER_VARIANT / 3600
    lines = [
        f"# {local_title}",
        "",
        f"> Paper reference: {paper_title}",
        "",
        f"- Run ID: `{run_id}`",
        f"- 완료: {suite['finished_at']}",
        f"- Wall time: {suite['wall_seconds'] / 3600:.2f} h (예상 ~{est_h:.1f} h)",
        f"- Model: `{hp.get('model_id')}`",
        f"- Train epochs: {hp.get('train_epochs', hp.get('epochs'))}",
        f"- Checkpoint dir: `{hp.get('checkpoint_dir')}`",
        f"- Variants: {', '.join(ok_variants)}",
        f"- Episodes / env: {hp.get('episodes_per_env')} (논문: {EPISODES_PER_ENV})",
        f"- 출력: `{run_rel}`",
        "",
        "### 논문 프로토콜 vs 이번 run",
        "",
        "| 항목 | 논문 | 이번 run |",
        "|------|------|----------|",
        f"| tag | `{paper_title}` | `{local_title}` |",
        f"| env 수 | 12 | 12 |",
        f"| episodes/env | {EPISODES_PER_ENV} | {hp.get('episodes_per_env')} |",
        f"| eval model | {PAPER_MODEL_ID} | {hp.get('model_id')} |",
        f"| train epochs | {NUM_EPOCHS} | {hp.get('train_epochs', '?')} |",
        f"| LoRA r / α | {PAPER_LORA_R} / {PAPER_LORA_ALPHA} | {hp.get('lora_r')} / {hp.get('lora_alpha')} |",
        f"| LoRA variants | 6 + merge | {', '.join(ok_variants)} |",
        "",
        "아래 **local** 표는 이번 run, **paper** 표는 PAPER.pdf 참조값입니다.",
        "",
        "---",
        "",
        f"## Table 1 — Headline · {local_title}",
        "",
        "협력 빈도 fc(ID/HO)와 cumulative reward CR(ID/HO). ID=인식적 게임 8종, HO=숨겨진 목표 4종.",
        "",
        "### 이번 eval (local)",
        "",
        *_format_table1_local({v: rows[v] for v in ok_variants}),
        "",
        f"### {paper_title}",
        "",
        *_format_table1_paper(PAPER_TABLE1),
        "",
        "---",
        "",
        "## Table 2 — Per-game cumulative reward",
        "",
        "12 env 각각 평균 CR. PD 3종: pd-classic, pd-tight, pd-high-temptation.",
        "",
        "### 이번 eval (local)",
        "",
        *_format_table2_block({v: rows[v] for v in ok_variants}, ok_variants),
        "",
        f"### {paper_title} (base / all)",
        "",
        *_format_table2_paper(["base", "all"]),
        "",
        "---",
        "",
        "## Table 3 — Per-opponent-axis CR",
        "",
        "상대 유형 axis별 평균 CR (ID/HO × Stationary/Stochastic/Strategic).",
        "",
        "### 이번 eval (local)",
        "",
        *_format_table3_block({v: rows[v] for v in ok_variants}, ok_variants),
        "",
        f"### {paper_title} (base / all)",
        "",
        *_format_table3_paper(["base", "all"]),
        "",
        "---",
        "",
        "## Table 4 — Final-round coupling fc",
        "",
        "마지막 라운드 reasoning–action coupling (0–1).",
        "",
        "### 이번 eval (local)",
        "",
        *_format_table4_block({v: rows[v] for v in ok_variants}, ok_variants),
        "",
        f"### {paper_title}",
        "",
        *_format_table4_paper(),
        "",
        "약어: pd-c, pd-t, pd-h, stag, nego, bos, mp, ttt, auct, dd, p-b, ipd",
        "",
        "---",
        "",
        "## Table 5 — Training manifest",
        "",
        "LoRA DPO variant별 data / checkpoint 존재 여부.",
        "",
        *_format_table5_block(manifest),
        "",
        "---",
        "",
        "## Table 6 — Stag-hunt anti-coordination (논문 예시)",
        "",
        "TFT 상대에 blind vs oracle 행동 (논문 Appendix F). Table 6은 논문 고정 예시 — rollout 없음.",
        "",
        *_format_table6_block(),
        "",
        "---",
        "",
    ]
    for v in ok_variants:
        lines += [
            f"## Table 7 — Companion metrics (`{v}`)",
            "",
            "env별 action consistency (ac), final-round fc, exploitability. `paper_fc` = 논문 Table 7.",
            "",
            *_format_table7_block(rows, v),
            "",
            "---",
            "",
        ]

    lines += [
        "## 파일 목록",
        "",
        "| 경로 | 설명 |",
        "|------|------|",
        "| `result.md` | 이 문서 |",
        "| `suite.json` | 전체 metrics |",
        "| `paper_refs.json` | 논문 Table 1–7 참조값 |",
        "| `tables/table{1-7}.json` | 테이블별 JSON |",
        "| `metrics/<variant>.json` | variant별 raw metrics |",
        "| `logs/eval_suite.log` | rollout 진행 로그 |",
        "",
        f"최신 run 포인터: `runs/latest.json` → `{run_rel}`",
        "",
    ]
    return "\n".join(lines)


def _resolve_eval_args(ns: argparse.Namespace) -> Namespace:
    paper = ns.paper
    return Namespace(
        mode="lora",
        model_id=ns.model_id or (PAPER_MODEL_ID if paper else MODEL_ID),
        checkpoint_dir=ns.checkpoint_dir,
        episodes=ns.episodes,
        seed=ns.seed,
        lora_r=ns.lora_r if ns.lora_r is not None else (PAPER_LORA_R if paper else LORA_R),
        lora_alpha=ns.lora_alpha
        if ns.lora_alpha is not None
        else (PAPER_LORA_ALPHA if paper else LORA_ALPHA),
        lora_target=ns.lora_target or (PAPER_LORA_TARGET_MODULES if paper else "q_proj,v_proj"),
        dpo_beta=DPO_BETA,
        merge_alpha=MERGE_ALPHA,
        lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        no_4bit=not USE_4BIT,
        max_tokens=ns.max_tokens,
        logger=ns.logger,
    )


def _write_table_jsons(run_dir: Path, suite: dict) -> None:
    tables = run_dir / "tables"
    save_json(tables / "table1.json", suite.get("table1", {}))
    save_json(tables / "table2.json", suite.get("table2", {}))
    save_json(tables / "table3.json", suite.get("table3", {}))
    save_json(tables / "table4.json", suite.get("table4", {}))
    save_json(tables / "table5.json", suite.get("table5_manifest", {}))
    save_json(tables / "table6.json", {"paper_reference": PAPER_TABLE6})
    save_json(tables / "table7.json", suite.get("table7", {}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval suite -> eval/runs/<timestamp>/")
    parser.add_argument("--checkpoint-dir", type=Path, default=LORA_DIR)
    parser.add_argument(
        "--variants",
        default=",".join(EVAL_DEFAULT_VARIANTS),
        help="Comma-separated (default: base,all)",
    )
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_ENV)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--paper", action="store_true", help="3B paper model + lora_3b/")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-target", default=None)
    parser.add_argument("--max-tokens", type=int, default=EVAL_MAX_TOKENS)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override timestamp folder name (default: UTC YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing eval/runs/<run-id>/logs/eval_suite.log",
    )
    parser.add_argument(
        "--games",
        default=None if EVAL_GAMES.lower() == "all" else EVAL_GAMES,
        help="Comma-separated games to run (default: all)",
    )
    args = parser.parse_args(argv)

    if args.games:
        from eval.resume import parse_games_arg

        try:
            args.games_list = parse_games_arg(args.games)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.games_list = None

    if args.resume and not args.run_id:
        parser.error("--resume requires --run-id (existing folder under runs/<id>/eval/)")

    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in args.variants:
        if v != "base" and v not in _discover_variants(args.checkpoint_dir):
            print(f"WARNING: variant '{v}' adapter not found under {args.checkpoint_dir}")

    run_id = args.run_id or utc_stamp()
    if args.resume:
        run_dir = resolve_run_dir(config.RUNS_DIR / run_id)
        if not run_dir.is_dir():
            parser.error(f"--resume: run dir not found: {run_dir}")
    else:
        run_dir = new_run_dir(run_id)
    log_path = run_dir / "logs" / "eval_suite.log"
    if args.resume and not log_path.is_file():
        parser.error(f"--resume: log not found: {log_path}")
    args.logger = EvalLogger(log_path=log_path, append=args.resume)

    est_h = len(args.variants) * EVAL_EST_SECONDS_PER_VARIANT / 3600
    print(f"Eval suite -> {run_dir}")
    print(f"Run ID: {run_id}")
    print(f"Variants: {args.variants}")
    print(f"Episodes/game: {args.episodes} (paper={EPISODES_PER_ENV})")
    print(f"Estimated wall time: ~{est_h:.1f} h")
    print(f"Log: {log_path}", flush=True)

    save_json(run_dir / "paper_refs.json", _paper_refs())
    manifest = _training_manifest(args.checkpoint_dir)

    eval_args = _resolve_eval_args(args)
    eval_args.resume = args.resume
    eval_args.games_list = args.games_list
    eval_args.run_dir = run_dir
    eval_args.log_file = log_path
    rows: dict[str, dict] = {}
    timings: dict[str, float] = {}
    suite_started = time.perf_counter()

    for i, variant in enumerate(args.variants, start=1):
        t0 = time.perf_counter()
        print(f"\n[{i}/{len(args.variants)}] Evaluating variant={variant} ...", flush=True)
        try:
            metrics = eval_variant(eval_args, variant)
            rows[variant] = metrics_to_dict(metrics)
            save_json(run_dir / "metrics" / f"{variant}.json", rows[variant])
        except Exception as exc:
            print(f"ERROR variant={variant}: {exc}", flush=True)
            rows[variant] = {"error": str(exc)}
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        timings[variant] = time.perf_counter() - t0
        print(f"  done variant={variant} in {timings[variant] / 60:.1f} min", flush=True)

    ok_variants = [v for v in args.variants if v in rows and "error" not in rows[v]]
    suite = {
        "run_id": run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - suite_started,
        "timings_seconds": timings,
        "hyperparams": _hyperparams(args),
        "table1": {
            v: {k: rows[v][k] for k in ("fc_id", "fc_ho", "cr_id", "cr_ho") if k in rows[v]}
            for v in ok_variants
        },
        "table2": {v: rows[v].get("per_game_cr", {}) for v in ok_variants},
        "table3": {v: rows[v].get("per_axis_cr", {}) for v in ok_variants},
        "table4": {v: rows[v].get("per_game_fc", {}) for v in ok_variants},
        "table5_manifest": manifest,
        "table6": {"paper_reference": PAPER_TABLE6},
        "table7": {
            v: {
                "ac": rows[v].get("per_game_ac", {}),
                "fc": rows[v].get("per_game_fc", {}),
                "exploitability": rows[v].get("per_game_exploitability", {}),
            }
            for v in ok_variants
        },
        "variants": args.variants,
        "rows": rows,
    }
    save_json(run_dir / "suite.json", suite)
    _write_table_jsons(run_dir, suite)

    result_md = build_result_md(
        suite=suite,
        rows=rows,
        ok_variants=ok_variants,
        manifest=manifest,
        run_dir=run_dir,
        run_id=run_id,
    )
    (run_dir / "result.md").write_text(result_md, encoding="utf-8")
    from eval.build_latex_md import write_latex_md

    latex_path = write_latex_md(run_dir, suite, ok_variants=ok_variants)
    write_latest_pointer(run_dir)

    print(f"\nSaved {run_dir / 'result.md'}")
    print(f"Saved {latex_path}")
    print(f"Latest pointer: {config.RUNS_DIR / 'latest.json'}")
    print(f"Total wall time: {suite['wall_seconds'] / 3600:.2f} h")
    return 0 if ok_variants else 1


if __name__ == "__main__":
    raise SystemExit(main())
