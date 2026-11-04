"""Run history: one JSON file per run under ``runs/``.

Plain files rather than a database: easy to diff, commit as examples, or
load into a notebook with ``json.load``. API keys are never part of a run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from slmbench.config import runs_dir
from slmbench.runner import RunResult


@dataclass
class RunIndexEntry:
    path: Path
    run_id: str
    started_at: str
    name: str
    models: list[str]
    n_prompts: int
    simulated: bool


def save_run(result: RunResult, directory: Path | None = None) -> Path:
    directory = Path(directory or runs_dir())
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(result.started_at).strftime("%Y-%m-%d_%H-%M-%S")
    path = directory / f"{stamp}_{result.run_id}.json"
    path.write_text(result.to_json())
    return path


def load_run(path: Path) -> RunResult:
    return RunResult.from_dict(json.loads(Path(path).read_text()))


def list_runs(directory: Path | None = None) -> list[RunIndexEntry]:
    directory = Path(directory or runs_dir())
    if not directory.exists():
        return []
    entries = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # a half-written or hand-edited file shouldn't break the History tab
        cfg = data.get("config", {})
        entries.append(
            RunIndexEntry(
                path=path,
                run_id=data.get("run_id", path.stem),
                started_at=data.get("started_at", ""),
                name=data.get("name", ""),
                models=cfg.get("models", []),
                n_prompts=cfg.get("n_prompts", 0),
                simulated=bool(data.get("simulated")),
            )
        )
    return entries
