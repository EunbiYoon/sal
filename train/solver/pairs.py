"""Load preference pairs whose chosen actions are pinned by the solver."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset


def load_pairs(path: Path) -> Dataset:
    """Load prompt/chosen/rejected JSONL rows as a Hugging Face dataset."""
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            rows.append(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
    if not rows:
        raise ValueError(f"No pairs in {path}")
    return Dataset.from_list(rows)
