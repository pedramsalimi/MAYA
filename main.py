def main():
    print("Hello from maya!")


if __name__ == "__main__":
    main()



# old factory.py
# maya/framework/supervisor/factory.py
# from __future__ import annotations

# import importlib
# import os
# from typing import Any, Dict, List, Tuple

# from typing_extensions import TypedDict, Annotated, NotRequired
# from pydantic import BaseModel, Field

# from langchain.chat_models import init_chat_model
# from langchain_core.messages import AnyMessage, SystemMessage
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.managed.is_last_step import RemainingSteps
# from langgraph.types import Command  # ← correct v1 import

# from maya.agents.config import load_agent_specs
# from maya.framework.memory import get_postgres_memory

# from maya.framework.supervisor import prompts
# # --------------------------- State ---------------------------

# class GlobalState(TypedDict):
#     # Required LangGraph message channel & controlled-steps
#     messages: Annotated[List[AnyMessage], add_messages]
#     remaining_steps: RemainingSteps
#     # Optional: for observability
#     active_agent: NotRequired[str]


# # ---------------------- Agent loading ------------------------

# def _build_agent(agent_id: str, spec: Dict[str, Any]):
#     module = importlib.import_module(f"maya.agents.{agent_id}.agent")
#     app = module.build(spec)
#     if not getattr(app, "name", None):
#         raise RuntimeError(f"Agent '{agent_id}' must compile with a name.")
#     return app

# def _load_agents_and_descriptions() -> Tuple[List[Any], Dict[str, str]]:
#     specs = load_agent_specs()
#     agents: List[Any] = []
#     descs: Dict[str, str] = {}
#     for agent_id, spec in specs.items():
#         if spec.get("disabled"):
#             continue
#         agent = _build_agent(agent_id, spec)
#         agents.append(agent)
#         descs[agent.name] = (spec.get("description") or "").strip()
#     if not agents:
#         raise RuntimeError("No enabled agents configured.")
#     return agents, descs


# # --------------------- Router (structured) --------------------

# class RouteChoice(BaseModel):
#     """LLM-structured routing output."""
#     route: str = Field(..., description="Exact agent name to handle the message, or 'none'.")
#     reason: str | None = Field(None, description="Optional brief rationale.")

# def _router_prompt(agent_descriptions: Dict[str, str]) -> str:
#     roster = "\n".join(
#         f"- {name}: {desc or 'No description provided.'}"
#         for name, desc in agent_descriptions.items()
#     ) or "- No specialists available."
#     allowed = ", ".join(sorted(agent_descriptions.keys())) or "none"
#     return (
#         "You are MAYA's router. Choose exactly ONE specialist to handle the user's latest message.\n"
#         "Respond ONLY as JSON matching the schema {route: <agent|none>, reason: <string>}.\n"
#         f"Valid values for 'route': [{allowed}] or 'none'. Do NOT answer the user's question.\n\n"
#         "Specialists:\n" + roster
#     )

# # def _last_user(msgs: List[AnyMessage] | None) -> AnyMessage | None:
# #     msgs = msgs or []
# #     for m in reversed(msgs):
# #         role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
# #         if role in {"human", "user"}:
# #             return m
# #     return None

# def _last_user(msgs: List[AnyMessage] | None) -> AnyMessage | None:
#     msgs = msgs or []
#     if not msgs:
#         return None

#     last = msgs[-1]
#     role = getattr(last, "type", None) or (last.get("role") if isinstance(last, dict) else None)
#     return last if role in {"human", "user"} else None

# # ---------------------- Supervisor build ---------------------

# def build_supervisor(
#     *,
#     model_name: str | None = None,
#     supervisor_name: str = "supervisor",
#     store=None,
#     checkpointer=None,
# ):
#     """
#     Native LangGraph supervisor:
#       • Router uses structured output to select agent.
#       • Returns Command(goto='<agent_name>') to jump.
#       • Each agent runs once for this turn; then END.

#     No injected 'return' tool → workers behave exactly as in isolation.
#     """
#     model_name = model_name or os.getenv("MAYA_SUPERVISOR_MODEL", "openai:gpt-4o-mini")

#     agents, descriptions = _load_agents_and_descriptions()
#     names = {a.name for a in agents}
#     store, checkpointer = get_postgres_memory()

#     for a in agents:
#         if not getattr(a, "name", None):
#             raise ValueError("Every agent must have a stable `name` (e.g., 'health_rag').")

#     base_llm = init_chat_model(model_name, temperature=0)
#     router_llm = base_llm.with_structured_output(RouteChoice)
#     router_sys = SystemMessage(content=_router_prompt(descriptions))
#     supervisor_sys = SystemMessage(content=prompts.system(descriptions))
#     def router_node(state: GlobalState):
#         user = _last_user(state.get("messages"))
#         if user is None:
#             return Command(goto=END)

#         choice = router_llm.invoke([router_sys, user])
#         target = (choice.route or "").strip()

#         if target in names:
#             # record selection (optional) and jump to the agent
#             return Command(goto=target, update={"active_agent": target})

#         # Small-talk/meta or invalid → answer briefly here and end
#         # smalltalk = base_llm.invoke([
#         #     SystemMessage(content="You are MAYA. Respond briefly and warmly. No tools."),
#         #     user,
#         # ])
#         # # Append the reply to history so your CLI prints it
#         # return Command(goto=END, update={"messages": [smalltalk]})
#         history = list(state.get("messages") or [])
#         reply = base_llm.invoke([supervisor_sys, *history[-8:]])  # trim if needed
#         return Command(goto=END, update={"messages": [reply]})

#     # Build: START -> router -> (goto agent) -> END
#     g = StateGraph(GlobalState)
#     g.add_node(supervisor_name, router_node)

#     for a in agents:
#         g.add_node(a.name, a)   # your existing agent (works unchanged)
#         g.add_edge(a.name, END)

#     g.add_edge(START, supervisor_name)

#     return g.compile(name=supervisor_name, store=store, checkpointer=checkpointer)
