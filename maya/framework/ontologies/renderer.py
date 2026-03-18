"""Prompt-friendly ontology rendering."""

from __future__ import annotations

from typing import Any


def render_ontology_block(ontology: dict[str, Any], *, limit: int = 4) -> str:
    entities = ontology.get("entities") or []
    lines = [f"Ontology: {ontology.get('ontology_name', 'unknown')}"]
    for item in entities[:limit]:
        label = item.get("label", item.get("concept_id", "unknown"))
        rule_lines = item.get("prompt_rules") or []
        lines.append(f"- {label}")
        for rule in rule_lines[:2]:
            lines.append(f"  - {rule}")
    return "\n".join(lines)
