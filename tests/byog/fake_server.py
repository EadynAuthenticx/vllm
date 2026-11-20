# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C0: Lightweight OpenAI-compatible fake vLLM server for CPU tests."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class FakeServerState:
    """Thread-safe behavior controls and call counters for FakeVLLMServer."""

    model_name: str = "fake/test-model"
    canned_answers: dict[str, str] = field(default_factory=dict)
    default_answer: str = "The answer is #### 42"
    latency_s: float = 0.0
    fail_after_n: int | None = None
    fail_status: int = 500
    allow_reset_cache: bool = True
    health_ok: bool = True

    lock: threading.Lock = field(default_factory=threading.Lock)
    completion_calls: int = 0
    chat_calls: int = 0
    reset_calls: int = 0
    health_calls: int = 0
    received_prompts: list[str] = field(default_factory=list)


def _solve_canned_or_arithmetic(prompt: str, state: FakeServerState) -> str:
    for needle, answer in state.canned_answers.items():
        if needle in prompt:
            return answer
    # Simple deterministic match for mini GSM8K fixture questions
    last_q = prompt.rsplit("Question:", 1)[-1]
    if "Janet's ducks" in last_q:
        return "She has 16 - 3 - 4 = 9 eggs and makes 9 * 2 = 18 dollars. #### 18"
    if "2 bolts of blue fiber" in last_q:
        return "It takes 2 + 1 = 3 bolts. #### 3"
    if "12 apples" in last_q and "5 apples" in last_q:
        return "12 - 5 = 7 apples left. #### 7"
    if "3 boxes" in last_q and "8 pencils" in last_q:
        return "3 * 8 = 24 pencils. #### 24"
    if "60 miles per hour" in last_q and "3 hours" in last_q:
        return "60 * 3 = 180 miles. #### 180"
    if "45 pages" in last_q and "15 pages" in last_q:
        return "45 + 15 = 60 pages. #### 60"
    if "What is " in last_q and "+" in last_q and "?" in last_q:
        expr = last_q.split("What is ", 1)[1].split("?", 1)[0]
        parts = [p.strip() for p in expr.split("+")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            ans = int(parts[0]) + int(parts[1])
            return f"The sum is {ans}. #### {ans}"
    return state.default_answer


class FakeVLLMServer:
    """In-process threaded OpenAI-compatible HTTP server for fast CPU testing."""

    def __init__(self, state: FakeServerState | None = None, host: str = "127.0.0.1"):
        self.state = state or FakeServerState()
        self.host = host
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("FakeVLLMServer has not been started.")
        port = self._server.server_address[1]
        return f"http://{self.host}:{port}"

    def start(self) -> FakeVLLMServer:
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/health":
                    with state.lock:
                        state.health_calls += 1
                        ok = state.health_ok
                    self._send_json(
                        200 if ok else 503, {"status": "ok" if ok else "err"}
                    )
                    return
                if self.path == "/v1/models":
                    self._send_json(
                        200,
                        {"object": "list", "data": [{"id": state.model_name}]},
                    )
                    return
                self._send_json(404, {"error": f"unknown path {self.path}"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8") or "{}")

                if self.path == "/reset_prefix_cache":
                    with state.lock:
                        state.reset_calls += 1
                        allowed = state.allow_reset_cache
                    if allowed:
                        self._send_json(200, {"status": "reset"})
                    else:
                        self._send_json(404, {"error": "dev mode disabled"})
                    return

                if state.latency_s > 0:
                    time.sleep(state.latency_s)

                if self.path == "/v1/completions":
                    with state.lock:
                        state.completion_calls += 1
                        total = state.completion_calls + state.chat_calls
                        prompt = str(payload.get("prompt", ""))
                        state.received_prompts.append(prompt)
                        should_fail = (
                            state.fail_after_n is not None
                            and total > state.fail_after_n
                        )
                    if should_fail:
                        self._send_json(
                            state.fail_status, {"error": "injected failure"}
                        )
                        return
                    text = _solve_canned_or_arithmetic(prompt, state)
                    toks = max(1, len(text.split()))
                    self._send_json(
                        200,
                        {
                            "id": f"cmpl-{total}",
                            "object": "text_completion",
                            "model": payload.get("model") or state.model_name,
                            "choices": [
                                {"index": 0, "text": text, "finish_reason": "stop"}
                            ],
                            "usage": {
                                "prompt_tokens": max(1, len(prompt.split())),
                                "completion_tokens": toks,
                                "total_tokens": max(1, len(prompt.split())) + toks,
                            },
                        },
                    )
                    return

                if self.path == "/v1/chat/completions":
                    with state.lock:
                        state.chat_calls += 1
                        total = state.completion_calls + state.chat_calls
                        messages = payload.get("messages") or []
                        prompt = "\n".join(str(m.get("content", "")) for m in messages)
                        state.received_prompts.append(prompt)
                        should_fail = (
                            state.fail_after_n is not None
                            and total > state.fail_after_n
                        )
                    if should_fail:
                        self._send_json(
                            state.fail_status, {"error": "injected failure"}
                        )
                        return
                    text = _solve_canned_or_arithmetic(prompt, state)
                    toks = max(1, len(text.split()))
                    self._send_json(
                        200,
                        {
                            "id": f"chatcmpl-{total}",
                            "object": "chat.completion",
                            "model": payload.get("model") or state.model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": text},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": max(1, len(prompt.split())),
                                "completion_tokens": toks,
                                "total_tokens": max(1, len(prompt.split())) + toks,
                            },
                        },
                    )
                    return

                self._send_json(404, {"error": f"unknown path {self.path}"})

        self._server = ThreadingHTTPServer((self.host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> FakeVLLMServer:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.stop()
