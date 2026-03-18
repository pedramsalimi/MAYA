from __future__ import annotations

from typing import List, NotRequired, TypedDict

from langchain.agents import AgentState as BaseAgentState


class PubMedCitation(TypedDict, total=False):
    """Structured metadata describing a PubMed snippet."""

    title: str
    summary: str
    url: str
    pmid: NotRequired[str]
    journal: NotRequired[str]
    year: NotRequired[str]
    score: NotRequired[float]


class HealthRagState(BaseAgentState):
    """State extension for the health agent (optional citation ledger)."""

    citations: NotRequired[List[PubMedCitation]]
