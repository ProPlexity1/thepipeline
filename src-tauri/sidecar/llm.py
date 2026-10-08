"""Local chat model runtime: a private llama.cpp server for the agent.

The server runs only while the agent is awake. It listens on a random loopback
port and requires a random API key, so other local programs and web pages can't
use it. It "sleeps" (the process exits, freeing VRAM and RAM) when a video,
image or voice job needs the GPU, when asked, or after a few idle minutes.
"""
import json
import os
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from model_registry import MODEL_CONFIG, MODELS_DIR, resolve_asset_path

IDLE_SLEEP_SECONDS = 10 * 60
LOAD_TIMEOUT_SECONDS = 300
LOG_DIR = MODELS_DIR.parent / "logs"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AgentBusy(RuntimeError):
    pass


class LlamaServer:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc: Optional[subprocess.Popen] = None
        self.model_id: Optional[str] = None
        self.port = 0
        self.key = ""
        self.state = "sleeping"  # sleeping | loading | ready
        self.last_used = 0.0
        self.on_state: Callable[[str, Optional[str]], None] = lambda state, model: None
        # Held for the whole of an agent turn; the GPU scheduler waits on it
        # before putting the agent to sleep, so a reply is never cut off.
        self.turn_lock = threading.Lock()
        # Replies in progress. The GPU scheduler waits for these to finish before it puts the
        # agent to sleep, so the agent always completes its answer (e.g. after queueing a video).
        self.active_turns = 0
        self._turns_lock = threading.Lock()
        self.gpu_busy: Callable[[], bool] = lambda: False
        self._log = None
        threading.Thread(target=self._idle_watch, daemon=True).start()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _set_state(self, state: str):
        self.state = state
        try:
            self.on_state(state, self.model_id)
        except Exception:
            pass

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, model_id: str) -> tuple[str, str]:
        """Start (or switch to) `model_id` and wait until it answers."""
        with self.lock:
            if self.running() and self.model_id == model_id and self.state == "ready":
                self.last_used = time.time()
                return f"http://127.0.0.1:{self.port}", self.key
            if self.gpu_busy():
                raise AgentBusy("A video, image or voice job is using the graphics card right now. "
                                "The agent wakes up again as soon as it finishes.")
            self.stop()
            cfg = MODEL_CONFIG[model_id]
            rt = cfg["runtime"]
            engine_dir = resolve_asset_path(rt["engine"])
            exe = engine_dir / "llama-server.exe"
            if not exe.exists():
                raise RuntimeError("The chat engine isn't installed. Reinstall the model from the Models page.")
            root = resolve_asset_path(rt["model_root"])
            self.port, self.key, self.model_id = _free_port(), secrets.token_urlsafe(24), model_id
            args = [
                str(exe), "-m", str(root / rt["model_file"]), "--host", "127.0.0.1", "--port", str(self.port),
                "--api-key", self.key, "-c", str(rt.get("context", 32768)), "--jinja", "--no-webui",
                "-fa", "on", "--parallel", "1",
            ]
            if rt.get("mmproj_file"):
                args += ["--mmproj", str(root / rt["mmproj_file"])]
            args += [str(a) for a in rt.get("extra_args", [])]
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self._log = open(LOG_DIR / "llama-server.log", "w", encoding="utf-8", errors="replace")
            env = {k: v for k, v in os.environ.items() if k not in ("NEURALCUT_TOKEN", "HF_TOKEN")}
            self._set_state("loading")
            self.proc = subprocess.Popen(args, cwd=str(engine_dir), stdout=self._log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=env,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.time() + LOAD_TIMEOUT_SECONDS
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    tail = self.log_tail()
                    self.stop()
                    raise RuntimeError(f"The chat model stopped while loading. {tail}")
                try:
                    r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                    if r.status_code == 200:
                        self.last_used = time.time()
                        self._set_state("ready")
                        return f"http://127.0.0.1:{self.port}", self.key
                except requests.RequestException:
                    pass
                time.sleep(0.5)
            self.stop()
            raise RuntimeError("The chat model took too long to load.")

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    self.proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None
            if self._log:
                self._log.close()
                self._log = None
            if self.state != "sleeping":
                self._set_state("sleeping")

    def begin_turn(self):
        with self._turns_lock:
            self.active_turns += 1

    def end_turn(self):
        with self._turns_lock:
            self.active_turns = max(0, self.active_turns - 1)

    def sleep_for_gpu_job(self, max_wait: float = 900):
        """Called by the GPU scheduler: let any reply in progress finish, then exit."""
        deadline = time.time() + max_wait
        while self.active_turns > 0 and time.time() < deadline:
            time.sleep(0.5)
        if not self.running():
            return
        with self.turn_lock:
            self.stop()

    def log_tail(self, n: int = 6) -> str:
        try:
            lines = (LOG_DIR / "llama-server.log").read_text(encoding="utf-8", errors="replace").splitlines()
            return " | ".join(l.strip() for l in lines[-n:] if l.strip())[-600:]
        except OSError:
            return ""

    def _idle_watch(self):
        while True:
            time.sleep(30)
            if (self.running() and not self.turn_lock.locked() and self.active_turns == 0
                    and time.time() - self.last_used > IDLE_SLEEP_SECONDS):
                self.stop()

    # ── chat ─────────────────────────────────────────────────────────────────
    def chat_stream(self, model_id: str, messages: list, tools: Optional[list],
                    on_delta: Callable[[str, str], None], cancelled: Callable[[], bool]) -> dict:
        """One streamed completion. on_delta(kind, text) gets 'reasoning' and
        'content' pieces as they arrive. Returns the assembled message."""
        base, key = self.ensure(model_id)
        body = {"messages": messages, "stream": True, "temperature": 0.7, "top_p": 0.9}
        if tools:
            body["tools"] = tools
        content, reasoning, calls = [], [], {}
        with requests.post(f"{base}/v1/chat/completions", json=body, stream=True, timeout=(10, 600),
                           headers={"Authorization": f"Bearer {key}"}) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Chat model error {r.status_code}: {r.text[:300]}")
            r.encoding = "utf-8"  # SSE has no charset; requests would otherwise assume Latin-1
            for raw in r.iter_lines(decode_unicode=True):
                if cancelled():
                    break
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                        on_delta("reasoning", delta["reasoning_content"])
                    if delta.get("content"):
                        content.append(delta["content"])
                        on_delta("content", delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] += fn.get("name") or ""
                        slot["arguments"] += fn.get("arguments") or ""
        self.last_used = time.time()
        return {
            "content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": [calls[i] for i in sorted(calls)],
        }


server = LlamaServer()
