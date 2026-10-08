"""A private, headless ComfyUI owned by one job.

Started on a random loopback port with no web UI and no custom nodes, fed an
API-format graph, and shut down afterwards so its memory is freed for other
models. Used by every ComfyUI-backed runner (MiniMax H3, SeedVR2).
"""
import json
import os
import re
import socket
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

import requests

from runners.common import Reporter

STARTUP_TIMEOUT_S = 240

# The engine log is chatty (DB migrations, backend probes); relay only lines a
# user would find meaningful while watching their job.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_ENGINE_KEEP = ("got prompt", "Requested to load", "loaded completely", "loaded partially", "Prompt executed",
                "%|", "rror", "out of memory", "Total VRAM", "Device:", "chunk", "Loading", "Unloaded")
_ENGINE_DROP = ("comfy_kitchen", "nodes_glsl", "comfy_angle", "workflow-templates", "embedded-docs")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ComfyEngine:
    """A ComfyUI process owned by this job."""

    def __init__(self, engine_dir: Path, model_root: Path, work: Path, extra_args: list[str], report: Reporter,
                 folders: tuple[str, ...] = ("diffusion_models", "text_encoders", "vae"),
                 extra_folders: dict[str, Path] | None = None):
        self.engine_dir, self.model_root, self.work, self.report = engine_dir, model_root, work, report
        self.folders = folders
        # Extra {comfy folder name: directory}, for files that live in a shared resource.
        self.extra_folders = extra_folders or {}
        self.extra_args = extra_args
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.log_path = work / "comfyui.log"
        self._log_pos = 0
        self._log = None

    def start(self):
        for d in ("output", "input", "temp", "user", "frontend"):
            (self.work / d).mkdir(parents=True, exist_ok=True)
        (self.work / "frontend" / "index.html").write_text("NeuralCut engine", encoding="utf-8")
        paths_yaml = self.work / "model_paths.yaml"
        root = self.model_root.as_posix()
        paths_yaml.write_text(
            "neuralcut:\n"
            f"  base_path: \"{root}\"\n"
            + "".join(f"  {f}: {f}\n" for f in self.folders)
            + "".join(f"extra_{i}:\n  base_path: \"{d.as_posix()}\"\n  {name}: .\n"
                      for i, (name, d) in enumerate(self.extra_folders.items())), encoding="utf-8")
        cmd = [
            sys.executable, str(self.engine_dir / "main.py"),
            "--listen", "127.0.0.1", "--port", str(self.port),
            "--disable-auto-launch", "--disable-all-custom-nodes", "--preview-method", "none",
            # Fully offline: no cloud "API nodes", and no prompt/workflow embedded in outputs.
            "--disable-api-nodes", "--disable-metadata",
            "--front-end-root", str(self.work / "frontend"),
            "--output-directory", str(self.work / "output"),
            "--input-directory", str(self.work / "input"),
            "--temp-directory", str(self.work / "temp"),
            "--user-directory", str(self.work / "user"),
            "--extra-model-paths-config", str(paths_yaml),
            *self.extra_args,
        ]
        self.report.log(f"Starting ComfyUI on 127.0.0.1:{self.port}")
        log = self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        env = dict(os.environ)
        env.pop("NEURALCUT_TOKEN", None)  # the engine never needs our API secret
        self.proc = subprocess.Popen(
            cmd, cwd=str(self.engine_dir), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            self.pump_log()
            if self.proc.poll() is not None:
                raise RuntimeError(f"The H3 engine exited during startup. {self.tail()}")
            try:
                if requests.get(f"{self.base}/system_stats", timeout=2).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError("The H3 engine did not start in time.")

    def pump_log(self):
        """Relay new engine log lines to our stdout so the UI shows them live."""
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_pos)
                chunk = f.read()
                self._log_pos = f.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            line = _ANSI.sub("", line).strip()
            if line and any(k in line for k in _ENGINE_KEEP) and not any(k in line for k in _ENGINE_DROP):
                print(f"[engine] {line}", flush=True)

    def tail(self, n: int = 15) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return " | ".join(l.strip() for l in lines[-n:] if l.strip())[-1500:]
        except OSError:
            return ""

    def stop(self):
        if self.proc and self.proc.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log:
            self._log.close()  # an open handle would stop the work folder from being deleted
            self._log = None


def remove_work_dir(work: Path, attempts: int = 10):
    """Delete a job's engine folder. Windows releases the killed engine's file
    handles (log, sqlite db) a moment after it exits, so retry briefly."""
    for _ in range(attempts):
        shutil.rmtree(work, ignore_errors=True)
        if not work.exists():
            return
        time.sleep(0.5)


def _format_node_errors(err: dict) -> str:
    parts = [err.get("error", {}).get("message", "Invalid workflow")]
    for node, info in (err.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            parts.append(f"{node}: {e.get('message')} {e.get('details', '')}".strip())
    return "; ".join(parts)[:800]




def run_graph(engine: "ComfyEngine", graph: dict, report: Reporter,
              on_event: Callable[[str, dict], None]) -> list[Path]:
    """Queue `graph`, relay engine log lines and events until it finishes, and
    return the paths of the files it produced."""
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed

    client_id = uuid.uuid4().hex
    try:
        with connect(f"ws://127.0.0.1:{engine.port}/ws?clientId={client_id}", max_size=None,
                     open_timeout=30) as ws:
            r = requests.post(f"{engine.base}/prompt", json={"prompt": graph, "client_id": client_id}, timeout=60)
            if not r.ok:
                try:
                    raise RuntimeError(_format_node_errors(r.json()))
                except ValueError:
                    raise RuntimeError(f"Engine rejected the job: HTTP {r.status_code}")
            prompt_id = r.json()["prompt_id"]
            while True:
                engine.pump_log()
                try:
                    msg = ws.recv(timeout=2)
                except TimeoutError:
                    if engine.proc.poll() is not None:
                        raise RuntimeError(f"The engine stopped unexpectedly. {engine.tail(6)}")
                    continue
                if isinstance(msg, bytes):
                    continue  # previews are disabled
                data = json.loads(msg)
                kind, body = data.get("type"), data.get("data", {})
                if body.get("prompt_id") not in (None, prompt_id):
                    continue
                if kind == "executing" and body.get("node") is None:
                    break
                if kind == "execution_error":
                    raise RuntimeError(f"{body.get('exception_type', 'Error')}: "
                                       f"{body.get('exception_message', '')}".strip()[:800])
                if kind == "execution_interrupted":
                    raise RuntimeError("Generation was interrupted.")
                on_event(kind, body)
    except ConnectionClosed:
        raise RuntimeError(f"The engine stopped unexpectedly (often: not enough memory). {engine.tail(6)}")

    engine.pump_log()
    hist = requests.get(f"{engine.base}/history/{prompt_id}", timeout=30).json().get(prompt_id, {})
    produced = []
    for node_out in (hist.get("outputs") or {}).values():
        for items in node_out.values():
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and str(it.get("filename", "")).endswith(".mp4"):
                        produced.append(engine.work / "output" / it.get("subfolder", "") / it["filename"])
    return [p for p in produced if p.exists()]
