from __future__ import annotations

from typing import List, NotRequired, TypedDict

from langchain.agents import AgentState as BaseAgentState


class LocalCitation(TypedDict, total=False):
    """Structured metadata describing a retrieved local-literature snippet."""

    title: str
    excerpt: str
    source_path: NotRequired[str]
    page_number: NotRequired[int]


class HealthRagState(BaseAgentState):
    """State extension for the local-literature health agent."""

    citations: NotRequired[List[LocalCitation]]
