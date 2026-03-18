"""Ontology loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ontology(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Ontology at {path} must be a JSON object.")
    return data
