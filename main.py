from __future__ import annotations

import os
from threading import Lock
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from maya.framework.supervisor import prompts
from maya.framework.supervisor.factory import build_supervisor
from maya.framework.supervisor.utils import (
    strip_citations_and_references,
    strip_markdown,
)

load_dotenv()

app = Flask(__name__)

_state_lock = Lock()
_run_state: Dict[str, Any] = {"graph": None, "base_config": None}


def _init_run_copy():
    """Create the supervisor graph and its base config (mirrors run.py setup)."""
    user_id = os.getenv("MAYA_RUN_USER_ID", "web_user")
    thread_id = os.getenv("MAYA_RUN_THREAD_ID", "web_thread")
    base_config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": 40,
    }
    graph = build_supervisor()
    return graph, base_config


def _clone_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    cloned = dict(base_config)
    configurable = base_config.get("configurable") or {}
    cloned["configurable"] = dict(configurable)
    return cloned


def _run_conversation(graph, base_config, user_text: str) -> str:
    payload = {"messages": [{"role": "user", "content": user_text}]}
    config = _clone_config(base_config)

    final_text: Optional[str] = None
    for chunk in graph.stream(
        payload,
        config=config,
        stream_mode="updates",
        subgraphs=True,
    ):
        update = chunk[1] if (isinstance(chunk, tuple) and len(chunk) == 2) else chunk
        if not isinstance(update, dict):
            continue

        for node_payload in update.values():
            if isinstance(node_payload, list):
                messages = node_payload
            elif isinstance(node_payload, dict):
                messages = node_payload.get("messages")
                if isinstance(messages, dict):
                    messages = list(messages.values())
                elif messages is None:
                    messages = []
                elif not isinstance(messages, list):
                    messages = [messages]
            else:
                continue

            if not messages:
                continue

            last = messages[-1]
            if isinstance(last, dict):
                role = last.get("role") or last.get("type")
                content = last.get("content")
            else:
                role = getattr(last, "type", None)
                content = getattr(last, "content", None)

            if role in {"assistant", "ai"} and isinstance(content, str) and content.strip():
                final_text = content.strip()

    if not final_text:
        return ""

    cleaned = strip_markdown(final_text)
    cleaned = strip_citations_and_references(cleaned)
    return cleaned


@app.post("/start")
def start_session():
    payload = request.get_json(silent=True) or {}
    status = str(payload.get("status", "")).strip().upper()
    if status != "START":
        return jsonify({"error": "Send {'status': 'START'} to initialize the session."}), 400

    graph, base_config = _init_run_copy()
    with _state_lock:
        _run_state["graph"] = graph
        _run_state["base_config"] = base_config

    return jsonify({"message": "Supervisor session initialised."})


@app.post("/conversation")
def conversation():
    payload = request.get_json(silent=True) or {}
    user_message = (payload.get("user_message") or "").strip()
    if not user_message:
        return jsonify({"error": "Missing user_message."}), 400

    with _state_lock:
        graph = _run_state.get("graph")
        base_config = _run_state.get("base_config")

    if graph is None or base_config is None:
        return jsonify({"error": "Session not started. Call /start first."}), 400

    try:
        response_text = _run_conversation(graph, base_config, user_message)
    except Exception as exc:  # pragma: no cover - surface runtime failures
        return jsonify({"error": f"Conversation failed: {exc}"}), 500

    return jsonify({"response": response_text})


if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG") == "1"
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=debug_enabled)
