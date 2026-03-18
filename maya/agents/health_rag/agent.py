"""Local-literature health RAG agent for the legacy MAYA supervisor."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, dynamic_prompt
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

from maya.agents.health_rag.state import HealthRagState
from maya.framework.ontologies import load_ontology, render_ontology_block
from maya.framework.rag.corpus import chunk_snapshot_path, corpus_root, load_chunk_snapshot, load_corpus_documents, split_documents
from maya.framework.rag.vectorstore import build_embeddings, build_vector_store, load_rag_vector_config


AGENT_ID = "health_rag"


class DisambiguationDecision(BaseModel):
    needs_clarification: bool = Field(description="Whether the question is too underspecified to answer safely.")
    ambiguity_reason: str = Field(default="", description="Short reason for the decision.")
    clarification_question: str = Field(default="", description="A single focused clarification question when clarification is required.")


class RetrieveLocalLiterature(AgentMiddleware[HealthRagState, Any]):
    state_schema = HealthRagState

    def __init__(self, retriever: Any):
        self.retriever = retriever

    def before_model(self, state: HealthRagState, runtime: Any) -> dict[str, Any] | None:
        question = ""
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                question = message.content.strip()
                break
        if not question:
            return {"citations": []}

        documents = self.retriever.invoke(question)
        citations: list[dict[str, Any]] = []
        for document in documents[:3]:
            excerpt = re.sub(r"\s+", " ", document.page_content).strip()
            if len(excerpt) > 500:
                excerpt = excerpt[:497].rsplit(" ", 1)[0] + "..."
            citations.append(
                {
                    "title": document.metadata.get("title", ""),
                    "source_path": document.metadata.get("source_path", ""),
                    "page_number": document.metadata.get("page_number", 0),
                    "excerpt": excerpt,
                }
            )
        return {"citations": citations}


@dynamic_prompt
def health_rag_prompt(request: ModelRequest) -> str:
    citations = request.state.get("citations") or []
    if not citations:
        return (
            "You are MAYA's health literature assistant. "
            "If the retrieved local literature does not contain a relevant answer, say that clearly and briefly."
        )

    context = "\n\n".join(
        f"{item.get('title', 'Local literature')} (page {int(item.get('page_number', 0)) + 1})\n{item.get('excerpt', '')}"
        for item in citations
    )
    return (
        "You are MAYA's health literature assistant. "
        "Answer only from the retrieved local literature below. "
        "Write 2 to 4 short plain-English sentences. "
        "Do not invent facts. If the retrieved context is insufficient, say so briefly.\n\n"
        f"Retrieved local literature:\n{context}"
    )


@dataclass(frozen=True)
class HealthRagHandle:
    name: str
    agent: Any
    ontology_block: str

    def assess_clarification(self, question: str, *, model: Any) -> DisambiguationDecision:
        if not question.strip() or model is None:
            return DisambiguationDecision(needs_clarification=False)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You decide whether a health question needs clarification before MAYA answers from local literature.\n"
                    "Use the ontology below as the policy source for deciding whether the question is underspecified.\n"
                    "Return a DisambiguationDecision.\n"
                    "Set needs_clarification to true only when MAYA lacks enough information for a safe, grounded answer from local literature.\n"
                    "If needs_clarification is true, provide one focused clarification_question.\n"
                    "If needs_clarification is false, leave clarification_question empty.\n\n"
                    "{ontology}",
                ),
                ("human", "User question: {question}"),
            ]
        )
        try:
            chain = prompt | model.with_structured_output(DisambiguationDecision, strict=True)
            decision = chain.invoke({"question": question, "ontology": self.ontology_block})
            if isinstance(decision, DisambiguationDecision):
                return decision
        except Exception:
            pass
        return DisambiguationDecision(needs_clarification=False)


def build(spec: dict[str, Any] | None = None) -> HealthRagHandle:
    root_dir = Path(__file__).resolve().parents[2]
    vector_config = load_rag_vector_config()
    documents = load_chunk_snapshot(chunk_snapshot_path(root_dir))
    if not documents and vector_config.backend != "pgvector":
        source_documents = load_corpus_documents(root_dir)
        documents = split_documents(source_documents) if source_documents else []
    if not documents and vector_config.backend != "pgvector":
        raise RuntimeError(f"No local health literature found under {corpus_root(root_dir)}.")

    embeddings = build_embeddings(vector_config)
    vector_store = build_vector_store(vector_config, embeddings, pre_delete=False)
    if vector_config.backend == "in_memory" and documents:
        vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold" if vector_config.backend == "pgvector" else "similarity",
        search_kwargs={"k": 4, "score_threshold": 0.28} if vector_config.backend == "pgvector" else {"k": 4},
    )
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        temperature=0,
        api_version="2024-12-01-preview",
        azure_endpoint="https://mayaagent.openai.azure.com/",
    )
    agent = create_agent(
        llm,
        tools=[],
        middleware=[RetrieveLocalLiterature(retriever), health_rag_prompt],
        state_schema=HealthRagState,
        name=AGENT_ID,
    )
    ontology_path = root_dir / "framework" / "ontologies" / "disambiguation_ontology.json"
    return HealthRagHandle(
        name=AGENT_ID,
        agent=agent,
        ontology_block=render_ontology_block(load_ontology(ontology_path), limit=5),
    )
