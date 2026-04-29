from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import AzureChatOpenAI
from .bridge import get_phyxio_service

load_dotenv()

AGENT_ID = "phyxio_exercise_agent"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOOLS: Sequence[str] = ("show_exercises", "calibrate_exercises", "start_exercises")
ROUTINE_ID = "6"


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _call_service(action: str) -> str:
    try:
        service = get_phyxio_service()
    except Exception as exc:
        return f"Phyxio service unavailable: {exc}"

    try:
        if action == "show":
            return _to_text(service.show_routine(ROUTINE_ID))
        if action == "calibrate":
            return _to_text(service.calibrate_routine(ROUTINE_ID))
        if action == "start":
            return _to_text(service.run_routine(ROUTINE_ID))
    except Exception as exc:
        return f"Phyxio action '{action}' failed: {exc}"

    return f"Unknown Phyxio action '{action}'."


def _select_mirror_action(user_text: str) -> str | None:
    text = user_text.lower()

    start_requested = any(
        phrase in text
        for phrase in (
            "start",
            "begin",
            "let's go",
            "lets go",
            "run it",
            "run routine",
            "start workout",
            "start my workout",
        )
    )
    if start_requested and any(
        phrase in text
        for phrase in (
            "chest pain",
            "faint",
            "fainting",
            "severe breathless",
            "severe dizziness",
            "dizzy",
            "bleeding",
        )
    ):
        return "safety_stop"

    if any(phrase in text for phrase in ("calibrate", "setup", "set up", "align", "alignment", "check form")):
        return "calibrate"
    if start_requested:
        return "start"
    if any(
        phrase in text
        for phrase in (
            "show",
            "view",
            "see",
            "what exercises",
            "which exercises",
            "my exercises",
            "exercise routine",
            "routine",
        )
    ):
        return "show"
    return None


def _action_response(action: str, service_result: str) -> str:
    if service_result.startswith("Phyxio service unavailable:") or " failed: " in service_result:
        return service_result
    if action == "show":
        return "I'm showing your routine on the mirror now.\n\nStep into view of the camera."
    if action == "calibrate":
        return "I'm calibrating your exercises on the mirror now.\n\nStep into view of the camera."
    if action == "start":
        return (
            "I'm starting your exercise routine now.\n\n"
            "Move at a comfortable pace, and stop if you feel pain or dizziness."
        )
    return service_result


@tool("show_exercises")
def show_exercises(_: str = ROUTINE_ID) -> str:
    """Show the configured exercise routine on the mirror."""

    return _call_service("show")


@tool("calibrate_exercises")
def calibrate_exercises(_: str = ROUTINE_ID) -> str:
    """Calibrate the configured exercise routine on the mirror."""

    return _call_service("calibrate")


@tool("start_exercises")
def start_exercises(_: str = ROUTINE_ID) -> str:
    """Start the configured exercise routine on the mirror."""

    return _call_service("start")


_TOOL_REGISTRY: Mapping[str, BaseTool] = {
    "show_exercises": show_exercises,
    "calibrate_exercises": calibrate_exercises,
    "start_exercises": start_exercises,
}


def _require_prompt(spec: Dict[str, Any]) -> str:
    prompt = spec.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"`prompt` missing or empty for agent '{AGENT_ID}'.")
    return prompt.strip()


def _normalize_tools(raw: Sequence[str] | str) -> Sequence[str]:
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(tool_id) for tool_id in raw)


def _resolve_tools(tool_ids: Sequence[str]) -> Sequence[BaseTool]:
    resolved: list[BaseTool] = []
    missing: list[str] = []
    for ident in tool_ids:
        tool_obj = _TOOL_REGISTRY.get(ident)
        if tool_obj is None:
            missing.append(ident)
            continue
        resolved.append(tool_obj)
    if missing:
        raise RuntimeError(f"Unknown tools for '{AGENT_ID}': {', '.join(missing)}")
    return resolved


def _extract_latest_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role") or msg.get("type")
            if role in {"human", "user"}:
                text = msg.get("content")
                return _to_text(text).strip() if text is not None else ""
        else:
            role = getattr(msg, "type", None)
            if role in {"human", "user"}:
                text = getattr(msg, "content", "")
                return _to_text(text).strip()
    return ""


class _PhyxioExerciseNode:
    def __init__(self, runner):
        self.name = AGENT_ID
        self._runner = runner

    @staticmethod
    def _extract_tool_call_id(messages: Any) -> str | None:
        if not isinstance(messages, list) or not messages:
            return None
        for msg in reversed(messages):
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    first = tool_calls[0]
                    if isinstance(first, dict):
                        call_id = first.get("id")
                        return str(call_id) if call_id else None
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                first = tool_calls[0]
                if isinstance(first, dict):
                    call_id = first.get("id")
                    return str(call_id) if call_id else None
                call_id = getattr(first, "id", None)
                return str(call_id) if call_id else None
        return None

    def __call__(self, state: Dict[str, Any], config=None, store: Any = None):
        del store

        messages = state.get("messages", [])
        user_text = _extract_latest_user_text(messages)
        tool_call_id = self._extract_tool_call_id(messages)
        if not user_text:
            user_text = "Help with the user's latest exercise request."

        selected_action = _select_mirror_action(user_text)
        if selected_action == "safety_stop":
            answer = (
                "Let's not start exercises right now. Stop and seek urgent medical help "
                "if you have chest pain, fainting, severe breathlessness, or severe dizziness."
            )
            if tool_call_id:
                return {
                    "messages": [
                        ToolMessage(
                            content=answer,
                            tool_call_id=tool_call_id,
                            name="RouteState",
                            additional_kwargs={"source": AGENT_ID},
                        )
                    ]
                }
            return {"messages": [AIMessage(content=answer)]}

        if selected_action in {"show", "calibrate", "start"}:
            tool_name = {
                "show": "show_exercises",
                "calibrate": "calibrate_exercises",
                "start": "start_exercises",
            }[selected_action]
            answer = _action_response(selected_action, _TOOL_REGISTRY[tool_name].invoke({}))
            if tool_call_id:
                return {
                    "messages": [
                        ToolMessage(
                            content=answer,
                            tool_call_id=tool_call_id,
                            name="RouteState",
                            additional_kwargs={"source": AGENT_ID},
                        )
                    ]
                }
            return {"messages": [AIMessage(content=answer)]}

        invoke_config: Dict[str, Any] = {}
        if isinstance(config, dict):
            configurable = config.get("configurable")
            if isinstance(configurable, dict):
                thread_id = configurable.get("thread_id")
                user_id = configurable.get("user_id")
                forwarded = {}
                if thread_id is not None:
                    forwarded["thread_id"] = thread_id
                if user_id is not None:
                    forwarded["user_id"] = user_id
                if forwarded:
                    invoke_config["configurable"] = forwarded
            if "recursion_limit" in config:
                invoke_config["recursion_limit"] = config["recursion_limit"]

        try:
            result = self._runner.invoke(
                {"messages": [HumanMessage(content=user_text)]},
                config=invoke_config or None,
            )
        except Exception as exc:
            error_text = f"Exercise agent failed: {exc}"
            if tool_call_id:
                return {
                    "messages": [
                        ToolMessage(content=error_text, tool_call_id=tool_call_id, name="RouteState")
                    ]
                }
            return {"messages": [AIMessage(content=error_text)]}

        answer = ""
        if isinstance(result, dict):
            messages = result.get("messages") or []
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    answer = _to_text(last.get("content", ""))
                else:
                    answer = _to_text(getattr(last, "content", ""))
        if not answer:
            answer = _to_text(getattr(result, "content", "")) or "Exercise request completed."

        if tool_call_id:
            return {
                "messages": [
                    ToolMessage(
                        content=answer,
                        tool_call_id=tool_call_id,
                        name="RouteState",
                        additional_kwargs={"source": AGENT_ID},
                    )
                ]
            }
        return {"messages": [AIMessage(content=answer)]}


def build(spec: Dict[str, Any] | None = None):
    spec = spec or {}

    prompt = _require_prompt(spec)
    model_name = str(spec.get("model") or DEFAULT_MODEL)
    tool_ids = _normalize_tools(spec.get("tools") or DEFAULT_TOOLS)
    tools = _resolve_tools(tool_ids)

    llm = AzureChatOpenAI(
        azure_deployment=model_name,
        temperature=0,
        api_version="2024-12-01-preview",
        azure_endpoint="https://mayaagent.openai.azure.com/",
    )

    runner = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        name=AGENT_ID,
    )
    return _PhyxioExerciseNode(runner)
