from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import requests
from langchain_core.tools import BaseTool, tool


# --------------------------------------------------------------------------- #
# Exceptions & data containers
# --------------------------------------------------------------------------- #


class ToolNotAvailableError(RuntimeError):
    """Raised when a requested tool is unavailable (missing deps, etc.)."""


@dataclass
class PubMedArticle:
    """Minimal container for downstream summarisation."""

    title: str
    abstract: str
    url: str
    pmid: str | None
    journal: str | None
    year: str | None

    def as_payload(self) -> Mapping[str, str | float]:
        payload: dict[str, str | float] = {
            "title": self.title,
            "summary": self.abstract,
            "url": self.url,
        }
        if self.pmid:
            payload["pmid"] = self.pmid
        if self.journal:
            payload["journal"] = self.journal
        if self.year:
            payload["year"] = self.year
        return payload


# --------------------------------------------------------------------------- #
# Plain PubMed helpers (E-utilities)
# --------------------------------------------------------------------------- #


def _search_pmids(query: str, limit: int) -> List[str]:
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": limit,
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("esearchresult", {}).get("idlist", [])


def _parse_article(node: ET.Element) -> PubMedArticle | None:
    article = node.find(".//Article")
    if article is None:
        return None

    pmid = node.findtext(".//MedlineCitation/PMID")
    title = (article.findtext("ArticleTitle") or "").strip() or "Untitled result"
    abstract_parts = [
        (elem.text or "").strip()
        for elem in article.findall(".//Abstract/AbstractText")
        if (elem.text or "").strip()
    ]
    abstract = " ".join(abstract_parts).strip() or "Abstract unavailable."
    journal = node.findtext(".//Journal/Title")
    pub_date = node.find(".//JournalIssue/PubDate")
    year = None
    if pub_date is not None:
        year = (
            pub_date.findtext("Year")
            or pub_date.findtext("MedlineDate")
            or pub_date.findtext("Month")
        )

    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    return PubMedArticle(
        title=title,
        abstract=abstract,
        url=url,
        pmid=pmid,
        journal=journal,
        year=year,
    )


def _fetch_articles(pmids: Iterable[str]) -> List[PubMedArticle]:
    ids = [pid for pid in pmids if pid]
    if not ids:
        return []

    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(ids),
            "rettype": "abstract",
            "retmode": "xml",
        },
        timeout=20,
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    articles: List[PubMedArticle] = []
    for node in root.findall(".//PubmedArticle"):
        article = _parse_article(node)
        if article:
            articles.append(article)
    return articles


# --------------------------------------------------------------------------- #
# LangChain tool surface
# --------------------------------------------------------------------------- #


@tool("pubmed_health_rag")
def pubmed_health_rag(question: str, max_results: int = 4) -> str:
    """Fetch and rank PubMed abstracts, returning JSON citations for the agent."""
    query = question.strip()
    if not query:
        raise ValueError("Provide a non-empty question for PubMed search.")

    try:
        pmids = _search_pmids(query, limit=max(10, int(max_results)))
        articles = _fetch_articles(pmids)
    except requests.RequestException as exc:
        return json.dumps(
            {
                "question": query,
                "error": f"PubMed request failed: {exc}",
                "citations": [],
            }
        )

    if not articles:
        return json.dumps(
            {
                "question": query,
                "note": "No PubMed documents were returned.",
                "citations": [],
            }
        )

    top_k = articles[: max(1, int(max_results))]
    return json.dumps(
        {
            "question": query,
            "citations": [item.as_payload() for item in top_k],
            "usage": {"retrieved": len(articles), "returned": len(top_k)},
        },
        ensure_ascii=False,
    )


_TOOL_REGISTRY: Mapping[str, BaseTool] = {
    "pubmed_health_rag": pubmed_health_rag,
}


def get_tools(tool_ids: Sequence[str]) -> List[BaseTool]:
    """Convert tool names declared in config into LangChain tool instances."""
    resolved: List[BaseTool] = []
    for ident in tool_ids:
        tool_obj = _TOOL_REGISTRY.get(ident)
        if tool_obj is None:
            raise ToolNotAvailableError(f"Unknown tool '{ident}'.")
        resolved.append(tool_obj)
    return resolved


__all__ = ["pubmed_health_rag", "get_tools", "ToolNotAvailableError", "PubMedArticle"]
