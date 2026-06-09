from __future__ import annotations

import os
import sys
from typing import Any, Dict

import requests

BASE_URL = os.getenv("MAYA_API_URL", "http://127.0.0.1:8000").rstrip("/")

session_id: str | None = None


def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}{path}"
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def start_session() -> None:
    global session_id
    data = _post_json("/start", {"user_id": "cli_user"})
    session_id = data.get("session_id")
    print(f"[start] {data}")


def send_message(message: str) -> str:
    global session_id

    if not session_id:
        raise RuntimeError("Session not started")

    data = _post_json(
        "/conversation",
        {
            "session_id": session_id,
            "user_message": message,
        },
    )

    # Handle interrupt case
    if data.get("interrupt"):
        prompts = data.get("response", [])
        answers = []
        for prompt in prompts:
            answer = input(f"MAYA: {prompt}\nYou: ").strip()
            answers.append(answer)

        data = _post_json(
            "/resume",
            {
                "session_id": session_id,
                "answers": answers,
            },
        )

    reply = data.get("response", "")
    print(f"MAYA: {reply}")
    return reply


def interactive_loop() -> int:
    print("Chat with the Flask agent. Ctrl+C or blank line to exit.\n")

    try:
        start_session()
    except requests.RequestException as exc:
        print(f"[error] Could not reach {BASE_URL}: {exc}", file=sys.stderr)
        return 1

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            break

        try:
            send_message(user_text)
        except requests.RequestException as exc:
            print(f"[error] Request failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(interactive_loop())
