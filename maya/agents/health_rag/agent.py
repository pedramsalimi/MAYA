from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from maya.agents.tools import ToolNotAvailableError, get_tools
from maya.framework.memory import get_postgres_memory
from maya.agents.health_rag.state import HealthRagState
from dotenv import load_dotenv

load_dotenv()

AGENT_ID = "health_rag"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOOLS: Sequence[str] = ("pubmed_health_rag",)
PUBMED_TOOL_NAME = "pubmed_health_rag" 
DEBUG = os.getenv("MAYA_DEBUG") == "1"


# ------------------------------ Middleware -------------------------------- #
PUBMED_TOOL_NAME = "pubmed_health_rag"

class ForcePubMedFirstTurn(AgentMiddleware[HealthRagState, Any]):
    def _saw_pubmed(self, state: HealthRagState) -> bool:
        msgs = state.get("messages") or []
        for m in msgs:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
            name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
            if role in {"tool", "ToolMessage"} and name == PUBMED_TOOL_NAME:
                return True
        return False

    def modify_model_request(self, state: HealthRagState, req: ModelRequest) -> ModelRequest:
        # On the FIRST model turn (no PubMed observation yet):
        if not self._saw_pubmed(state):
            # 1) hide the return tool so it's not selectable
            req.tools = [t for t in (req.tools or []) if getattr(t, "name", "") != "transfer_back_to_supervisor"]
            # 2) force PubMed specifically (note the dict shape)
            req.tool_choice = {"type": "tool", "name": PUBMED_TOOL_NAME}
            # if DEBUG:
            #     print(f"[{AGENT_ID}:mw] forcing first tool → {PUBMED_TOOL_NAME}; "
            #           f"available={[getattr(t,'name',str(t)) for t in (req.tools or [])]}")
        return req



# ---------------------------- Agent construction --------------------------- #

def build(spec: Dict[str, Any] | None = None):
    """Compile the health RAG agent for supervisor use."""
    spec = spec or {}

    prompt = _require_prompt(spec)
    model_name = spec.get("model") or DEFAULT_MODEL
    tool_ids = _normalize_tools(spec.get("tools") or DEFAULT_TOOLS)

    try:
        tools = get_tools(tool_ids)
    except ToolNotAvailableError as exc:
        raise RuntimeError(f"Failed to load tools for '{AGENT_ID}': {exc}") from exc

    # Bind with tools; `tool_choice="any"` ensures the model MUST choose a tool.
    # llm = init_chat_model(model_name, temperature=0)
    # llm = llm.bind_tools(tools)
    llm = AzureChatOpenAI(azure_deployment="gpt-4o-mini", temperature=0, api_version="2024-12-01-preview", azure_endpoint="https://mayaagent.openai.azure.com/")
    llm = llm.bind_tools(tools)
    # if DEBUG:
    #     print(f"[{AGENT_ID}] tools bound: {[getattr(t, 'name', str(t)) for t in tools]}")

    store, checkpointer = get_postgres_memory()
    middlewares = [ForcePubMedFirstTurn()]
    # if DEBUG:
    #     class _Debug(AgentMiddleware[HealthRagState, Any]):
    #         state_schema = HealthRagState
    #         def before_model(self, state: HealthRagState, runtime):
    #             print(f"[DEBUG:{AGENT_ID}] before_model: {len(state.get('messages', []))} msgs")
    #             return None
    #     middlewares.append(_Debug())

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=prompt,
        name=AGENT_ID,
        state_schema=HealthRagState,   
        middleware=middlewares,
        checkpointer=checkpointer,
        store=store,
    )
    return agent


# --------------------------------- Helpers -------------------------------- #

def _require_prompt(spec: Dict[str, Any]) -> str:
    prompt = spec.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"`prompt` missing or empty for agent '{AGENT_ID}'.")
    return prompt.strip()

def _normalize_tools(raw: Sequence[str] | str) -> Sequence[str]:
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(tool_id) for tool_id in raw)


# -------------------------- smoke test -------------------------- #

def _demo_prompt() -> str:
    return (
        "You are MAYA's health research specialist. Your job is to answer clinical information questions by grounding your response in current evidence from PubMed via your tool.\n\nSUBAGENT CONTRACT (do not reveal):\n- Evidence-first: On a fresh clinical question, you MUST call `pubmed_health_rag` before finalizing any answer (unless the necessary citations already exist in state and are clearly relevant to the current question).\n- Use tool output as primary evidence. Do not dump raw JSON. Extract key findings and synthesize clearly.\n- Output format:\n  1) A direct, plain-language answer that highlights mechanism, benefits/risks, key contraindications, and major uncertainties.\n  2) Inline numeric citations like [1], [2] at claim-level where appropriate.\n  3) A short 'References' section with title – journal – year – PMID/URL for each citation used.\n  4) A brief safety disclaimer (no diagnosis/prescription; advise consulting a clinician).\n- Turn-taking: Do NOT call `transfer_back_to_supervisor` until you have (a) run at least one `pubmed_health_rag` call for the current question and (b) produced the final user-facing answer as above. Only then return control.\n- Escalation: If you detect emergency/safety-critical content (e.g., chest pain, overdose), produce a brief safety message and then return control.\n- Scope guardrails: If the user asks for non-medical matters, return control without answering.\n\nStyle: Be precise, neutral, and concise. Avoid speculation; if evidence is weak or mixed, say so explicitly. Prefer recent, high-quality sources. Avoid directives about dosing or therapy initiation.\n"
    )

def run_demo(question: str = "What does bisoprolol do in the body?") -> str:
    spec = {
        "prompt": _demo_prompt(),
        "model": os.getenv("MAYA_DEMO_MODEL", DEFAULT_MODEL),
        "tools": DEFAULT_TOOLS,
    }
    agent = build(spec)
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": "health_rag_demo"}, "recursion_limit": 30},
    )
    last_message = result["messages"][-1]
    if hasattr(last_message, "content"):
        return last_message.content
    if isinstance(last_message, dict):
        return json.dumps(last_message)
    return str(last_message)


if __name__ == "__main__":
    print("--- Health RAG demo ---")
    print(run_demo())
