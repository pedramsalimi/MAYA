from __future__ import annotations

import os
import uuid
from threading import Lock
from typing import Any, Dict, Optional, Tuple, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import BadRequestError

from langgraph.types import Command
from maya.framework.supervisor.factory import build_supervisor
from maya.framework.supervisor.utils import (
    strip_citations_and_references,
    strip_markdown,
)

load_dotenv()

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Session Store
# --------------------------------------------------

_state_lock = Lock()
_sessions: Dict[str, Dict[str, Any]] = {}


def _create_session(user_id: str) -> Dict[str, Any]:
    thread_id = f"web-{uuid.uuid4().hex[:10]}"
    base_config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        },
        "recursion_limit": 40,
    }

    graph = build_supervisor()

    return {
        "graph": graph,
        "base_config": base_config,
    }


def _clone_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    cloned = dict(base_config)
    cloned["configurable"] = dict(base_config.get("configurable", {}))
    return cloned


# --------------------------------------------------
# Stream Updates (same logic as CLI)
# --------------------------------------------------

def stream_updates(graph, payload, base_config) -> Tuple[Optional[str], List[Any]]:
    final_text = None
    interrupts = []

    for chunk in graph.stream(
        payload,
        config=_clone_config(base_config),
        stream_mode="updates",
        subgraphs=True,
    ):
        update = chunk[1] if isinstance(chunk, tuple) else chunk
        if not isinstance(update, dict):
            continue

        if "__interrupt__" in update:
            interrupts.extend(update.get("__interrupt__", []))
            continue

        for node_payload in update.values():
            if node_payload is None:
                continue

            if isinstance(node_payload, dict):
                messages = node_payload.get("messages", [])
            elif isinstance(node_payload, list):
                messages = node_payload
            else:
                messages = getattr(node_payload, "messages", [])

            if not messages:
                continue

            last = messages[-1]
            if isinstance(last, dict):
                role = last.get("role") or last.get("type")
                content = last.get("content")
            else:
                role = getattr(last, "type", None)
                content = getattr(last, "content", None)

            if role in {"assistant", "ai"} and content:
                final_text = content

    return final_text, interrupts


def is_tool_call_history_error(err: Exception) -> bool:
    if not isinstance(err, BadRequestError):
        return False
    text = str(err)
    return (
        "assistant message with 'tool_calls' must be followed by tool messages" in text
        and "tool_call_id" in text
    )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.post("/start")
def start():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id", "web_user")

    session_id = uuid.uuid4().hex


    with _state_lock:
        _sessions[session_id] = _create_session(user_id)

    return jsonify({
        "session_id": session_id,
        "message": "Session started."
    })


@app.post("/conversation")
def conversation():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    user_message = (payload.get("user_message") or "").strip()

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    if not user_message:
        return jsonify({"error": "Missing user_message"}), 400

    with _state_lock:
        session = _sessions.get(session_id)

    if not session:
        return jsonify({"error": "Invalid session_id"}), 400

    graph = session["graph"]
    base_config = session["base_config"]

    payload = {"messages": [{"role": "user", "content": user_message}]}

    while True:
        try:
            final_text, interrupts = stream_updates(graph, payload, base_config)
        except Exception as err:
            if is_tool_call_history_error(err):
                # Recover by creating new thread_id
                # new_thread = f"web-{uuid.uuid4().hex[:10]}"
                # base_config["configurable"]["thread_id"] = new_thread
                final_text, interrupts = stream_updates(graph, payload, base_config)
            else:
                return jsonify({"error": str(err)}), 500

        if interrupts:
            # Send interrupt prompt back to frontend
            prompts = [str(i.value) for i in interrupts]
            cleaned_prompt = " ".join(prompts)
            cleaned = strip_citations_and_references(strip_markdown(cleaned_prompt))
            print("TESTING interrupt====")
            print(cleaned)

            return jsonify({
                # "interrupt": True,
                "response": cleaned
            })

        break

    if not final_text:
        return jsonify({"response": ""})

    cleaned = strip_citations_and_references(strip_markdown(final_text))
    print("TESTING====")
    print(cleaned)
    return jsonify({
        "response": cleaned
    })


@app.post("/resume")
def resume():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    answers = payload.get("answers")

    if not session_id or answers is None:
        return jsonify({"error": "Missing session_id or answers"}), 400

    with _state_lock:
        session = _sessions.get(session_id)

    if not session:
        return jsonify({"error": "Invalid session_id"}), 400

    graph = session["graph"]
    base_config = session["base_config"]

    resume_value = answers[0] if isinstance(answers, list) and len(answers) == 1 else answers
    payload = Command(resume=resume_value)

    final_text, _ = stream_updates(graph, payload, base_config)

    cleaned = strip_citations_and_references(strip_markdown(final_text or ""))

    return jsonify({"response": cleaned})


# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
