from __future__ import annotations
from typing_extensions import TypedDict, NotRequired, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps

class SupervisorState(TypedDict):
    # REQUIRED by prebuilt agents/supervisor in LangGraph v1
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: RemainingSteps

    # Optional
    active_agent: NotRequired[str]
