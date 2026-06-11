from __future__ import annotations

import json
import os
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
DEFAULT_TOOLS: Sequence[str] = (
    "show_exercises",
    "calibrate_exercises",
    "start_exercises",
    "list_exercises",
    "get_exercise_history",
)
ROUTINE_ID = os.getenv("MAYA_PHYXIO_ROUTINE_ID", "").strip()
FALLBACK_ROUTINE_ID = "6"


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _normalize_routine_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        return json.loads(data)
    if isinstance(data, dict):
        return data
    raise TypeError(f"Unexpected routine list type: {type(data).__name__}")


def _routine_candidates(routine_id: str | None = None) -> list[str]:
    candidates: list[str] = []
    for value in (routine_id, ROUTINE_ID, FALLBACK_ROUTINE_ID):
        text = str(value or "").strip()
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def _resolve_routine_id(routine_id: str | None = None, data: dict[str, Any] | None = None) -> str:
    if data is None:
        data = _get_routine_list()

    error = data.get("error")
    if error:
        raise RuntimeError(str(error))

    routines = data.get("routines") or {}
    if not isinstance(routines, dict) or not routines:
        raise RuntimeError("No assigned Phyxio routines were returned.")

    for candidate in _routine_candidates(routine_id):
        if candidate in routines:
            return candidate
        try:
            numeric_candidate = str(int(candidate))
        except (TypeError, ValueError):
            numeric_candidate = ""
        if numeric_candidate and numeric_candidate in routines:
            return numeric_candidate

    return str(next(iter(routines.keys())))


def _call_service(action: str, routine_id: str | None = None) -> str:
    try:
        service = get_phyxio_service()
    except Exception as exc:
        return f"Phyxio service unavailable: {exc}"

    try:
        resolved_routine_id = _resolve_routine_id(routine_id)
        if action == "show":
            return _to_text(service.show_routine(resolved_routine_id))
        if action == "calibrate":
            return _to_text(service.calibrate_routine(resolved_routine_id))
        if action == "start":
            return _to_text(service.run_routine(resolved_routine_id))
    except Exception as exc:
        return f"Phyxio action '{action}' failed: {exc}"

    return f"Unknown Phyxio action '{action}'."


def _get_routine_list() -> dict[str, Any]:
    service = get_phyxio_service()
    if hasattr(service, "get_routine_list"):
        data = service.get_routine_list()
    else:
        data = service._phyxio.routine.get_list()

    return _normalize_routine_payload(data)


def _get_session_history(page: int = 1) -> dict[str, Any]:
    service = get_phyxio_service()
    get_sessions = getattr(service, "get_sessions", None)
    if not callable(get_sessions):
        raise RuntimeError("Phyxio session history is not available in this connector version.")
    return _normalize_routine_payload(get_sessions(page=page))


def _format_goal(value: Any, unit: str) -> str:
    text = str(value or "").strip()
    if not text or text in {"0", "None", "null"}:
        return ""
    return f"{text} {unit}"


def _brief_description(value: Any, max_chars: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    first_sentence = text.split(". ", 1)[0].rstrip(".")
    if len(first_sentence) <= max_chars:
        return first_sentence
    return first_sentence[: max_chars - 1].rstrip() + "."


def _parse_page(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 1
    try:
        page = int(text)
    except ValueError:
        return 1
    return max(page, 1)


def _get_routine(data: dict[str, Any], routine_id: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved_routine_id = _resolve_routine_id(routine_id, data)
    routines = data.get("routines") or {}
    routine = routines.get(resolved_routine_id)
    if not isinstance(routine, dict):
        raise RuntimeError(f"Routine {resolved_routine_id} is not available.")
    return resolved_routine_id, routine


def _format_routine_exercises(data: dict[str, Any], routine_id: str | None = None) -> str:
    error = data.get("error")
    if error:
        return f"I couldn't fetch your exercise list: {error}"

    exercises = data.get("exercises") or {}
    try:
        resolved_routine_id, routine = _get_routine(data, routine_id)
    except Exception as exc:
        return f"I couldn't fetch your exercise list: {exc}"

    routine_name = str(routine.get("name") or f"Routine {resolved_routine_id}").strip()
    routine_exercises = routine.get("exercises") or []
    if not routine_exercises:
        return f"{routine_name} does not currently list any exercises."

    lines = [f"Your current routine is {routine_name}. Exercises:"]
    for index, item in enumerate(routine_exercises, start=1):
        exercise_id = str(item.get("id") or "").strip()
        exercise = exercises.get(exercise_id) or exercises.get(item.get("id")) or {}
        name = str(exercise.get("name") or f"Exercise {exercise_id or index}").strip()
        description = str(exercise.get("description") or "").strip()
        goals = [
            goal
            for goal in (
                _format_goal(item.get("iterations"), "repetitions"),
                _format_goal(item.get("runtime"), "seconds max"),
            )
            if goal
        ]
        calibration = "calibration needed" if exercise.get("needs_calibration") else ""
        details = ", ".join(part for part in [*goals, calibration] if part)
        line = f"{index}. {name}"
        if details:
            line += f" ({details})"
        if description:
            line += f": {description}"
        lines.append(line)

    return "\n".join(lines)


def _format_spoken_routine_summary(data: dict[str, Any], routine_id: str | None = None) -> str:
    error = data.get("error")
    if error:
        return f"I couldn't fetch your exercise list: {error}"

    exercises = data.get("exercises") or {}
    try:
        resolved_routine_id, routine = _get_routine(data, routine_id)
    except Exception as exc:
        return f"I couldn't fetch your exercise list: {exc}"

    routine_name = str(routine.get("name") or f"Routine {resolved_routine_id}").strip()
    routine_exercises = routine.get("exercises") or []
    if not routine_exercises:
        return f"{routine_name} does not currently list any exercises."

    count = len(routine_exercises)
    plural = "s" if count != 1 else ""
    lines = [f"Your current routine is {routine_name}, with {count} exercise{plural}."]
    for index, item in enumerate(routine_exercises, start=1):
        exercise_id = str(item.get("id") or "").strip()
        exercise = exercises.get(exercise_id) or exercises.get(item.get("id")) or {}
        name = str(exercise.get("name") or f"Exercise {exercise_id or index}").strip()
        goals = [
            goal
            for goal in (
                _format_goal(item.get("iterations"), "repetitions"),
                _format_goal(item.get("runtime"), "seconds max"),
            )
            if goal
        ]
        description = _brief_description(exercise.get("description"))
        details = ", ".join(goals)
        line = f"{index}. {name}"
        if details:
            line += f": {details}"
        if description:
            line += f". {description}"
        lines.append(line)
    return "\n".join(lines)


def _format_duration(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return ""
    if total <= 0:
        return ""
    minutes, remainder = divmod(total, 60)
    if minutes and remainder:
        return f"{minutes} min {remainder} sec"
    if minutes:
        return f"{minutes} min"
    return f"{remainder} sec"


def _format_session_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "Unknown date"
    return text.replace("T", " ").replace("Z", "").split("+", 1)[0]


def _exercise_lookup(data: dict[str, Any]) -> dict[str, str]:
    exercises = data.get("exercises") or {}
    if not isinstance(exercises, dict):
        return {}
    lookup: dict[str, str] = {}
    for exercise_id, exercise in exercises.items():
        if isinstance(exercise, dict):
            name = str(exercise.get("name") or "").strip()
            if name:
                lookup[str(exercise_id)] = name
    return lookup


def _format_exercise_history(
    session_data: dict[str, Any],
    routine_data: dict[str, Any] | None = None,
    max_sessions: int = 3,
) -> str:
    error = session_data.get("error")
    if error:
        return f"I couldn't fetch your exercise history: {error}"

    sessions = session_data.get("sessions") or []
    if not isinstance(sessions, list) or not sessions:
        return "I couldn't find any previous Phyxio exercise sessions yet."

    exercise_names = _exercise_lookup(routine_data or {})
    page = session_data.get("page")
    prefix = "Recent exercise sessions"
    if page:
        prefix += f" (page {page})"

    lines = [f"{prefix}:"]
    for session in sessions[:max_sessions]:
        if not isinstance(session, dict):
            continue
        date = _format_session_date(session.get("date"))
        routine_id = str(session.get("routine_id") or "").strip()
        results = session.get("results") or []
        if not isinstance(results, list):
            results = []

        completed = sum(
            1
            for result in results
            if isinstance(result, dict) and str(result.get("status") or "").lower() == "completed"
        )
        total = len(results)
        header = f"- {date}"
        if routine_id:
            header += f", routine {routine_id}"
        if total:
            header += f": {completed}/{total} completed"
        lines.append(header)

        for result in results[:4]:
            if not isinstance(result, dict):
                continue
            exercise_id = str(result.get("exercise_id") or "").strip()
            name = exercise_names.get(exercise_id) or f"Exercise {exercise_id or '?'}"
            status = str(result.get("status") or "unknown").replace("_", " ")
            goal_iter = result.get("goal_iter")
            done_iter = result.get("done_iter")
            elapsed = _format_duration(result.get("elapsed_time"))
            parts = [status]
            if goal_iter is not None or done_iter is not None:
                parts.append(f"{done_iter or 0}/{goal_iter or 0} reps")
            if elapsed:
                parts.append(elapsed)
            lines.append(f"  {name}: {', '.join(parts)}")

        if len(results) > 4:
            lines.append(f"  Plus {len(results) - 4} more result(s).")

    if len(sessions) > max_sessions:
        lines.append(f"I found {len(sessions)} sessions on this page; showing the latest {max_sessions}.")

    return "\n".join(lines)


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
            "history",
            "progress",
            "session",
            "sessions",
            "last time",
            "previous exercise",
            "previous workout",
            "past exercise",
            "past workout",
            "how did i do",
        )
    ):
        return "history"
    if any(
        phrase in text
        for phrase in (
            "list",
            "how many exercises",
            "what exercises",
            "which exercises",
            "current exercises",
            "exercise list",
            "what is in my routine",
            "what's in my routine",
            "tell me my exercises",
        )
    ):
        return "list"
    if any(
        phrase in text
        for phrase in (
            "show",
            "open",
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
    if action == "list":
        return service_result
    if action == "history":
        return service_result
    return service_result


@tool("show_exercises")
def show_exercises(routine_id: str = "") -> str:
    """Show the configured exercise routine on the mirror."""

    return _call_service("show", routine_id)


@tool("calibrate_exercises")
def calibrate_exercises(routine_id: str = "") -> str:
    """Calibrate the configured exercise routine on the mirror."""

    return _call_service("calibrate", routine_id)


@tool("start_exercises")
def start_exercises(routine_id: str = "") -> str:
    """Start the configured exercise routine on the mirror."""

    return _call_service("start", routine_id)


@tool("list_exercises")
def list_exercises(routine_id: str = "") -> str:
    """List the exercises in the user's configured Phyxio routine."""

    try:
        return _format_spoken_routine_summary(_get_routine_list(), routine_id)

    except Exception as exc:
        return f"I couldn't fetch your exercise list: {exc}"


@tool("get_exercise_history")
def get_exercise_history(page: str = "1") -> str:
    """Summarize recent Phyxio exercise session history."""

    try:
        routine_data = None
        try:
            routine_data = _get_routine_list()
        except Exception:
            routine_data = None
        return _format_exercise_history(_get_session_history(_parse_page(page)), routine_data)
    except Exception as exc:
        return f"I couldn't fetch your exercise history: {exc}"

_TOOL_REGISTRY: Mapping[str, BaseTool] = {
    "show_exercises": show_exercises,
    "calibrate_exercises": calibrate_exercises,
    "start_exercises": start_exercises,
    "list_exercises": list_exercises,
    "get_exercise_history": get_exercise_history,
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


def _include_default_tools(tool_ids: Sequence[str]) -> Sequence[str]:
    ordered = list(tool_ids)
    for tool_id in DEFAULT_TOOLS:
        if tool_id not in ordered:
            ordered.append(tool_id)
    return tuple(ordered)


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


def _invoke_registered_tool(tool_name: str) -> str:
    tool_obj = _TOOL_REGISTRY[tool_name]
    func = getattr(tool_obj, "func", None)
    if callable(func):
        return _to_text(func(""))
    return _to_text(tool_obj.invoke({}))


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

        if selected_action in {"show", "calibrate", "start", "list", "history"}:
            tool_name = {
                "show": "show_exercises",
                "calibrate": "calibrate_exercises",
                "start": "start_exercises",
                "list": "list_exercises",
                "history": "get_exercise_history",
            }[selected_action]
            answer = _action_response(selected_action, _invoke_registered_tool(tool_name))
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
    tool_ids = _include_default_tools(_normalize_tools(spec.get("tools") or DEFAULT_TOOLS))
    tools = _resolve_tools(tool_ids)

    # llm = AzureChatOpenAI(
    #     azure_deployment=model_name,
    #     temperature=0,
    #     api_version="2024-12-01-preview",
    #     azure_endpoint="https://mayaagent.openai.azure.com/",
    # )
    llm = AzureChatOpenAI(azure_deployment="gpt-4.1-mini", temperature=0, api_version="2025-04-01-preview", azure_endpoint="https://socet-air-6721-resource.services.ai.azure.com/")

    runner = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt,
        name=AGENT_ID,
    )
    return _PhyxioExerciseNode(runner)
