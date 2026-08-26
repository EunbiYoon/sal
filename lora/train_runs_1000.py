"""Train LoRA (1 epoch) for each dpo/runs/*_1000 folder."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "bos_1000",
    "matchingpennies_1000",
    "negotiation_1000",
    "pd_1000",
    "staghunt_1000",
    "tictactoe_1000",
]
LOG = ROOT / "results" / "train_1000_epoch1.log"


def log(msg: str) -> None:
    line = f"{msg}\n"
    print(msg, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    LOG.write_text(f"=== batch start {datetime.now().isoformat()} ===\n", encoding="utf-8")
    py = sys.executable
    for name in RUNS:
        pairs = ROOT / "dpo" / "data" / name / "a_beta_all.jsonl"
        out = ROOT / "lora" / name
        log(f"\n=== TRAIN {name} {datetime.now().isoformat()} ===")
        cmd = [
            py,
            "-m",
            "lora.train",
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--epochs",
            "1",
        ]
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc != 0:
            log(f"FAILED {name} exit={rc}")
            return rc
        log(f"DONE {name}")
    log(f"\n=== ALL FINISHED {datetime.now().isoformat()} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
