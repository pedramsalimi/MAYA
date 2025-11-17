from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

_DEFAULT_CONFIG = Path(__file__).with_name("config.jsonl")
_CONFIG_PATH = Path(os.getenv("MAYA_AGENTS_CONFIG", str(_DEFAULT_CONFIG)))

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_COMMENT_LINE = re.compile(r"//.*?$", re.M)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _clean(text: str) -> str:
    text = _COMMENT_BLOCK.sub("", text)
    text = _COMMENT_LINE.sub("", text)
    return _TRAILING_COMMA.sub(r"\1", text)


def _load_specs() -> Dict[str, Dict[str, Any]]:
    if not _CONFIG_PATH.exists():
        return {}

    raw = _CONFIG_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    text = _clean(raw)

    if text.startswith("{") and text.endswith("}"):
        specs = json.loads(text)
        if not isinstance(specs, dict):
            raise ValueError("Top-level value must be an object.")
        for agent_id, spec in specs.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Agent '{agent_id}' must map to an object.")
        return specs

    specs: Dict[str, Dict[str, Any]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("Each JSONL line must be an object.")
        for agent_id, spec in data.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Agent '{agent_id}' must map to an object.")
            specs[agent_id] = spec

    return specs


_SPECS = _load_specs()


def load_agent_specs() -> Dict[str, Dict[str, Any]]:
    return _SPECS
