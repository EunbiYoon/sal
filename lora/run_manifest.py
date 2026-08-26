"""Persist LoRA train run metadata (command, config snapshot, README)."""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import (
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    PER_DEVICE_BATCH,
    PROJECT_ROOT,
    STUDENT_MODEL,
    TEACHER_MODEL,
    TRAIN_BF16,
    TRAIN_FP16,
    USE_4BIT,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_config_snapshot(
    *,
    variant: str,
    pairs_path: Path,
    pairs_count: int,
    model_id: str,
    paper: bool,
    epochs: int,
    lr: float,
    beta: float,
    lora_r: int,
    lora_alpha: int,
    lora_target: str,
    max_length: int,
    grad_accum: int,
    publish_dir: Path,
) -> dict[str, Any]:
    pairs_path = pairs_path.resolve()
    snap: dict[str, Any] = {
        "variant": variant,
        "pairs": pairs_path.relative_to(PROJECT_ROOT).as_posix()
        if pairs_path.is_relative_to(PROJECT_ROOT)
        else str(pairs_path),
        "pairs_count": pairs_count,
        "model_id": model_id,
        "student_model": model_id or STUDENT_MODEL,
        "teacher_model": TEACHER_MODEL,
        "paper": paper,
        "epochs": epochs,
        "learning_rate": lr,
        "dpo_beta": beta,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_target": lora_target,
        "max_length": max_length,
        "gradient_accumulation_steps": grad_accum,
        "per_device_train_batch_size": PER_DEVICE_BATCH,
        "use_4bit": USE_4BIT,
        "quantization": "4bit" if USE_4BIT else "none",
        "bnb_4bit_quant_type": BNB_4BIT_QUANT_TYPE if USE_4BIT else None,
        "bnb_4bit_compute_dtype": BNB_4BIT_COMPUTE_DTYPE if USE_4BIT else None,
        "fp16": TRAIN_FP16,
        "bf16": TRAIN_BF16,
        "publish_dir": publish_dir.relative_to(PROJECT_ROOT).as_posix()
        if publish_dir.is_relative_to(PROJECT_ROOT)
        else str(publish_dir),
    }
    data_manifest = pairs_path.parent / "latest_manifest.json"
    if data_manifest.is_file():
        snap["pairs_data_manifest"] = data_manifest.relative_to(PROJECT_ROOT).as_posix()
    dpo_run = _infer_dpo_run_dir(pairs_path)
    if dpo_run is not None:
        snap["dpo_source_run"] = dpo_run.relative_to(PROJECT_ROOT).as_posix()
        dpo_snap = dpo_run / "config_snapshot.yaml"
        if dpo_snap.is_file():
            snap["dpo_config_snapshot"] = dpo_snap.relative_to(PROJECT_ROOT).as_posix()
    return snap


def _infer_dpo_run_dir(pairs_path: Path) -> Path | None:
    """Map dpo/data/<run>/a_beta_all.jsonl -> dpo/runs/<run> when present."""
    parent = pairs_path.parent.name
    if parent == "data":
        return None
    candidate = PROJECT_ROOT / "dpo" / "runs" / parent
    return candidate if candidate.is_dir() else None


def persist_run_manifest(
    run_dir: Path,
    *,
    argv: list[str] | None,
    args: Namespace,
    snapshot: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd_argv = list(argv if argv is not None else sys.argv)
    command = shlex.join(cmd_argv)

    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    pairs_path = Path(snapshot["pairs"])
    if not pairs_path.is_absolute():
        pairs_path = PROJECT_ROOT / pairs_path
    if pairs_path.is_file():
        shutil.copy2(pairs_path, run_dir / "pairs_source.jsonl")

    data_manifest = pairs_path.parent / "latest_manifest.json"
    if data_manifest.is_file():
        shutil.copy2(data_manifest, run_dir / "pairs_manifest.json")

    manifest: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running",
        "command": command,
        "argv": cmd_argv,
        "config_snapshot": "config_snapshot.yaml",
        "run_dir": str(run_dir.resolve()),
        "adapter_dir": str((run_dir / "adapter").resolve()),
        "publish_dir": str(
            (PROJECT_ROOT / snapshot["publish_dir"]).resolve()
            if not Path(snapshot["publish_dir"]).is_absolute()
            else Path(snapshot["publish_dir"]).resolve()
        ),
        "timestamped_out": getattr(args, "no_timestamp_out", False) is False,
        "cli": {
            "pairs": snapshot["pairs"],
            "out": snapshot["publish_dir"],
            "paper": snapshot["paper"],
            "epochs": snapshot["epochs"],
            "model_id": snapshot["model_id"],
            "max_length": snapshot["max_length"],
            "grad_accum": snapshot["gradient_accumulation_steps"],
        },
        "resolved_config": snapshot,
        "train_metrics": None,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "README.md").write_text(_format_run_readme(manifest, snapshot), encoding="utf-8")


def finalize_run_manifest(
    run_dir: Path,
    *,
    status: str = "completed",
    train_metrics: dict[str, Any] | None = None,
) -> None:
    path = run_dir / "run_info.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["finished_at"] = utc_now_iso()
    manifest["status"] = status
    if train_metrics is not None:
        manifest["train_metrics"] = train_metrics
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    snapshot = manifest.get("resolved_config") or {}
    (run_dir / "README.md").write_text(
        _format_run_readme(manifest, snapshot, train_metrics=train_metrics),
        encoding="utf-8",
    )


def publish_adapter(adapter_dir: Path, publish_dir: Path) -> None:
    """Copy final adapter files to variant dir (skip training checkpoints)."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    for item in adapter_dir.iterdir():
        if item.name.startswith("checkpoint-"):
            continue
        if item.is_dir():
            dest = publish_dir / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, publish_dir / item.name)


def _format_run_readme(
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    train_metrics: dict[str, Any] | None = None,
) -> str:
    metrics = train_metrics or manifest.get("train_metrics")
    metrics_block = ""
    if metrics:
        metrics_block = (
            "\n## 학습 결과\n\n"
            f"| 항목 | 값 |\n|------|-----|\n"
            f"| train_runtime (s) | `{metrics.get('train_runtime')}` |\n"
            f"| train_loss | `{metrics.get('train_loss')}` |\n"
            f"| global_step | `{metrics.get('global_step')}` |\n"
        )

    dpo_src = snapshot.get("dpo_source_run") or "(unknown)"
    return f"""# LoRA train run `{Path(manifest["run_dir"]).name}`

DPO LoRA 학습 1회 실행 기록입니다. 재현·비교용으로 **명령어·설정 스냅샷·데이터 출처**를 저장합니다.

## 실행 명령

```bash
{manifest["command"]}
```

- 시작: `{manifest.get("started_at")}`
- 종료: `{manifest.get("finished_at") or "(running)"}`
- 상태: `{manifest.get("status")}`
- variant: `{snapshot.get("variant")}`
- publish: `{snapshot.get("publish_dir")}`

## 설정 요약 (`config_snapshot.yaml`)

| 항목 | 값 |
|------|-----|
| student model (blind / LoRA target) | `{snapshot.get("student_model")}` |
| teacher model (oracle) | `{snapshot.get("teacher_model")}` |
| pairs | `{snapshot.get("pairs")}` ({snapshot.get("pairs_count")} rows) |
| epochs | `{snapshot.get("epochs")}` |
| LoRA | r=`{snapshot.get("lora_r")}`, alpha=`{snapshot.get("lora_alpha")}`, target=`{snapshot.get("lora_target")}` |
| max_length | `{snapshot.get("max_length")}` |
| grad_accum | `{snapshot.get("gradient_accumulation_steps")}` |
| DPO beta | `{snapshot.get("dpo_beta")}` |
| paper mode | `{snapshot.get("paper")}` |
| DPO rollout run | `{dpo_src}` |
{metrics_block}
## 출력 파일

| 경로 | 설명 |
|------|------|
| `config_snapshot.yaml` | 학습 하이퍼파라미터 스냅샷 |
| `run_info.json` | 기계-readable 메타데이터 |
| `pairs_source.jsonl` | 학습에 사용한 pairs 복사본 |
| `pairs_manifest.json` | `dpo/data/.../latest_manifest.json` (있을 때) |
| `adapter/` | LoRA 체크포인트 + 최종 adapter |

## Eval

publish 경로 adapter로 Table eval:

```bash
./eval/eval.sh
```
"""
