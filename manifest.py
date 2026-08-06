"""Fixed prompt manifest for the staleness experiment.

subset_seed picks WHICH rows; it is independent of the decode seed so the
prompt set never changes between runs/treatments (paired design).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

TASKS = ("gsm8k", "humaneval", "mt-bench")


def _load_rows(dataset_root: str, name: str) -> list[dict]:
    path = Path(dataset_root) / f"{name}.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_manifest(
    *,
    dataset_root: str,
    samples_per_task: int,
    subset_seed: int,
) -> list[dict]:
    manifest: list[dict] = []
    for task in TASKS:
        rows = _load_rows(dataset_root, task)
        idxs = list(range(len(rows)))
        rng = random.Random(subset_seed)
        rng.shuffle(idxs)
        for row_idx in sorted(idxs[:samples_per_task]):
            prompt = rows[row_idx]["turns"][0]
            manifest.append(
                {
                    "dataset": task,
                    "row_idx": row_idx,
                    "prompt": prompt,
                    "prompt_sha1": hashlib.sha1(
                        prompt.encode("utf-8")
                    ).hexdigest()[:12],
                }
            )
    return manifest
