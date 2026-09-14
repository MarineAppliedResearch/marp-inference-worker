"""Durable local control selected by the volunteer who owns this machine."""

import json
import os
from pathlib import Path


VALID_ACTIONS = {"running", "finish", "stop"}


def state_path() -> Path:
    state_dir = Path(os.environ.get("MARP_WORKER_STATE_DIR") or Path("data") / "worker")
    return state_dir / "operator-control.json"


def read_action() -> str:
    path = state_path()
    if not path.is_file():
        return "running"
    try:
        action = str(json.loads(path.read_text(encoding="utf-8")).get("action"))
    except (OSError, ValueError):
        return "finish"
    return action if action in VALID_ACTIONS else "finish"


def write_action(action: str) -> None:
    if action not in VALID_ACTIONS:
        raise ValueError(f"unknown operator action: {action}")
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"action": action}, indent=2), encoding="utf-8")
    temporary.replace(path)
