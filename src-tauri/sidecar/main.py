import sys
# Force UTF-8 on stdout/stderr so Windows CP1252 never crashes on Unicode worker output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import asyncio
import contextlib
import hmac
import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

import agent
import editor
import production
import llm
import storage
import telemetry
from downloader import DownloadCancelled, ModelDownload
from model_registry import (
    MODEL_CONFIG, VMR_METADATA, MODELS_DIR, OUTPUT_DIR,
    check_downloaded, resolve_generation_params,
    load_runtime_profile, save_runtime_profile, get_strategy_ladder,
)

SIDECAR_BUILD = "sidecar-v7-secure-2026-10-01"
WORKER_BUILD = "worker-v7-runners-2026-10-01"  # bump to re-test strategies that failed on older builds

# ── Auth ──────────────────────────────────────────────────────────────────────
# The desktop shell generates a random token per launch and hands it to both
# the sidecar (env) and the UI (Tauri command). Every request must carry it, so
# web pages open in the user's browser can't drive this local server.
API_TOKEN = os.environ.get("NEURALCUT_TOKEN") or secrets.token_urlsafe(32)
if not os.environ.get("NEURALCUT_TOKEN"):
    print(f"[NeuralCut] Dev mode: no NEURALCUT_TOKEN given, generated one: {API_TOKEN}", flush=True)

ALLOWED_ORIGINS = [
    "http://tauri.localhost", "https://tauri.localhost", "tauri://localhost",
    "http://localhost:5173", "http://127.0.0.1:5173",
]


def _token_ok(token: Optional[str]) -> bool:
    return bool(token) and hmac.compare_digest(token, API_TOKEN)


class SuppressNoisyEndpoints(logging.Filter):
    NOISY_PATHS = ("/gpu/stats", "/gpu ", "GET /models ", "/storage")

    def filter(self, record):
        return not any(p in record.getMessage() for p in self.NOISY_PATHS)


logging.getLogger("uvicorn.access").addFilter(SuppressNoisyEndpoints())


class RedactToken(logging.Filter):
    """uvicorn logs WebSocket URLs, which carry the API token; keep it out of log files."""

    def filter(self, record):
        if API_TOKEN in record.getMessage():
            record.msg = record.getMessage().replace(API_TOKEN, "***")
            record.args = ()
        return True


logging.getLogger("uvicorn.error").addFilter(RedactToken())

app = FastAPI()


@app.middleware("http")
async def require_token(request: Request, call_next):
    if request.method != "OPTIONS" and request.url.path != "/health":
        token = request.headers.get("x-neuralcut-token") or request.query_params.get("token")
        if not _token_ok(token):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


# Added last = outermost: answers CORS preflight before the token check.
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["content-type", "x-neuralcut-token"],
)
# Blocks DNS-rebinding: a hostile page can't reach us under its own hostname.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

# ── State ─────────────────────────────────────────────────────────────────────


def _initial_state(model_id: str) -> dict:
    downloaded = check_downloaded(model_id, MODELS_DIR)
    return {"downloaded": downloaded, "downloading": False, "progress": 100.0 if downloaded else 0.0,
            "speed_mbps": 0.0, "eta_seconds": 0, "error": None}


models_db = {mid: _initial_state(mid) for mid in MODEL_CONFIG}
active_downloads: dict[str, ModelDownload] = {}
active_connections: list[WebSocket] = []
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def downloaded_map() -> dict[str, bool]:
    return {mid: s["downloaded"] for mid, s in models_db.items()}


def publish_log(job_id: str, raw: str):
    line = telemetry.clean_line(raw)
    mode = telemetry.add_log(job_id, line) if line else None
    if mode:
        broadcast_from_thread({"type": "job_log", "job_id": job_id, "line": line, "replace": mode == "replace"})


# Latest job_status per job, so a UI that missed WebSocket events (reconnect,
# reload) can resync from GET /jobs instead of showing a stuck job forever.
job_states: dict[str, dict] = {}


def broadcast_from_thread(payload: dict):
    if payload.get("type") == "job_status" and payload.get("job_id"):
        job_states[payload["job_id"]] = payload
        while len(job_states) > 200:
            job_states.pop(next(iter(job_states)))
    if main_event_loop is None or not active_connections:
        return

    async def _send_all():
        for conn in list(active_connections):
            try:
                await conn.send_json(payload)
            except Exception:
                if conn in active_connections:
                    active_connections.remove(conn)

    asyncio.run_coroutine_threadsafe(_send_all(), main_event_loop)


# ── Request models ────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(max_length=4000)
    negative_prompt: str = Field(default="", max_length=2000)
    model_id: str
    profile: str = "balanced"
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    seed: Optional[int] = None
    # Optional: run this enhancer on the result as soon as it is generated.
    enhancer_id: Optional[str] = None
    enhance_target: Optional[str] = None
    # Image-to-video: an image from the image gallery (file name) to animate.
    start_image: Optional[str] = Field(default=None, max_length=120)
    # Optional last frame for models that support it (MiniMax H3).
    end_image: Optional[str] = Field(default=None, max_length=120)


class ImageUpload(BaseModel):
    data_url: str = Field(max_length=25_000_000)


class EnhanceRequest(BaseModel):
    source: str = Field(max_length=120)  # an output file name, e.g. video_ab12cd34.mp4
    enhancer_id: str
    target: str = "1080p"


class EditClip(BaseModel):
    name: str = Field(max_length=120)
    start: float = Field(default=0.0, ge=0)
    end: Optional[float] = Field(default=None, gt=0)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)


class EditOverlay(BaseModel):
    kind: str = Field(pattern="^(image|text)$")
    image: Optional[str] = Field(default=None, max_length=120)  # image gallery name
    text: str = Field(default="", max_length=300)
    start: float = Field(default=0.0, ge=0)
    end: float = Field(default=3.0, gt=0)
    x: float = Field(default=0.5, ge=0, le=1)
    y: float = Field(default=0.5, ge=0, le=1)
    width: float = Field(default=0.2, ge=0.02, le=1)
    size: float = Field(default=0.07, ge=0.02, le=0.3)
    opacity: float = Field(default=1.0, ge=0, le=1)
    color: str = Field(default="#ffffff", pattern="^#[0-9a-fA-F]{6}$")
    background: bool = False


class EditAudio(BaseModel):
    name: str = Field(max_length=120)          # audio library file name
    start: float = Field(default=0.0, ge=0)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)


class EditRequest(BaseModel):
    clips: list[EditClip] = Field(min_length=1, max_length=editor.MAX_CLIPS)
    transition: str = Field(default="cut", pattern="^(cut|fade)$")
    fade_seconds: float = Field(default=0.5, ge=0.1, le=3.0)
    fade_edges: bool = True
    resolution: str = Field(default="auto", pattern="^(auto|720p|1080p)$")
    title: str = Field(default="", max_length=200)
    overlays: list[EditOverlay] = Field(default_factory=list, max_length=30)
    audio: list[EditAudio] = Field(default_factory=list, max_length=20)


class AgentMessage(BaseModel):
    text: str = Field(default="", max_length=20000)
    images: list[str] = Field(default_factory=list, max_length=6)  # data: URLs, resized by the UI
    model_id: Optional[str] = None


class SettingsUpdate(BaseModel):
    allow_web_research: Optional[bool] = None
    agent_model: Optional[str] = None
    brave_api_key: Optional[str] = Field(default=None, max_length=200)


class LicenseRequest(BaseModel):
    key: str = Field(max_length=200)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if (origin and origin not in ALLOWED_ORIGINS) or not _token_ok(websocket.query_params.get("token")):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    active_connections.append(websocket)
    try:
        await websocket.send_json({"type": "connection_established"})
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        if websocket in active_connections:
            active_connections.remove(websocket)


# ── Downloads ─────────────────────────────────────────────────────────────────

# Downloads still in progress are listed on disk, so if the engine restarts or
# crashes mid-download it picks them up again by itself on the next start.
PENDING_DOWNLOADS = MODELS_DIR.parent / "pending_downloads.json"
_pending_lock = threading.Lock()


def _set_pending(model_id: str, pending: bool):
    with _pending_lock:
        try:
            ids = set(json.loads(PENDING_DOWNLOADS.read_text(encoding="utf-8"))) if PENDING_DOWNLOADS.exists() else set()
        except ValueError:
            ids = set()
        (ids.add if pending else ids.discard)(model_id)
        try:
            PENDING_DOWNLOADS.write_text(json.dumps(sorted(ids)), encoding="utf-8")
        except OSError:
            pass


def _begin_download(model_id: str):
    models_db[model_id].update(downloading=True, error=None)
    active_downloads[model_id] = ModelDownload(model_id, lambda p: None)
    _set_pending(model_id, True)
    threading.Thread(target=_run_download, args=(model_id,), daemon=True).start()


def resume_pending_downloads():
    try:
        ids = json.loads(PENDING_DOWNLOADS.read_text(encoding="utf-8")) if PENDING_DOWNLOADS.exists() else []
    except ValueError:
        ids = []
    for model_id in ids:
        if model_id in models_db and not models_db[model_id]["downloaded"] and model_id not in active_downloads:
            print(f"[NeuralCut] Resuming interrupted download: {model_id}", flush=True)
            _begin_download(model_id)
        elif model_id not in models_db or models_db[model_id]["downloaded"]:
            _set_pending(model_id, False)


def _run_download(model_id: str):
    dl = active_downloads[model_id]
    state = models_db[model_id]

    def on_progress(p: dict):
        state.update({k: p[k] for k in ("progress", "speed_mbps", "eta_seconds") if k in p})
        broadcast_from_thread(p)

    dl.on_progress = on_progress
    try:
        dl.run()
        state.update(downloaded=True, downloading=False, progress=100.0, speed_mbps=0.0, eta_seconds=0, error=None)
        print(f"[NeuralCut] Model {model_id} downloaded and verified", flush=True)
    except DownloadCancelled:
        state.update(downloading=False, speed_mbps=0.0, eta_seconds=0, error=None)
        print(f"[NeuralCut] Download cancelled: {model_id} (partial files kept for resume)", flush=True)
    except Exception as e:
        state.update(downloading=False, speed_mbps=0.0, eta_seconds=0, error=str(e))
        print(f"[NeuralCut] Download failed for {model_id}: {e}", flush=True)
    finally:
        active_downloads.pop(model_id, None)
        _set_pending(model_id, False)  # finished, paused or failed: only a dead engine leaves it pending
        broadcast_from_thread({"type": "download_progress", "model_id": model_id, **state})


LICENSE_LOG = MODELS_DIR.parent / "accepted_licenses.json"


class DownloadRequest(BaseModel):
    accept_license: bool = False


@app.post("/models/download/{model_id}")
def start_download(model_id: str, body: Optional[DownloadRequest] = None):
    if model_id not in models_db:
        raise HTTPException(404, "Model not found")
    gate = MODEL_CONFIG[model_id].get("license_gate")
    if gate:
        if not (body and body.accept_license):
            raise HTTPException(409, f"You need to accept the {gate['name']} before downloading.")
        try:
            log = json.loads(LICENSE_LOG.read_text(encoding="utf-8")) if LICENSE_LOG.exists() else {}
        except ValueError:
            log = {}
        log[model_id] = {"license": gate["name"], "url": gate.get("url"),
                         "accepted_at": datetime.now().isoformat(timespec="seconds")}
        LICENSE_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    state = models_db[model_id]
    if state["downloaded"] or model_id in active_downloads:
        return {"status": "success"}
    _begin_download(model_id)
    return {"status": "success"}


@app.post("/models/download/{model_id}/cancel")
def cancel_download(model_id: str):
    dl = active_downloads.get(model_id)
    if dl:
        dl.cancel()
    return {"status": "success"}


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    if model_id not in models_db:
        raise HTTPException(404, "Model not found")
    if model_id in active_downloads:
        active_downloads[model_id].cancel()
        time.sleep(1.0)
    removed = storage.delete_model(model_id, downloaded_map())
    for mid in models_db:  # a shared folder may have gone with it
        if not active_downloads.get(mid):
            models_db[mid].update(_initial_state(mid))
    return {"status": "success", "removed": removed}


# ── Storage ───────────────────────────────────────────────────────────────────

@app.get("/storage")
def storage_report():
    return storage.report(downloaded_map())


@app.delete("/storage/orphans/{name}")
def delete_orphan(name: str):
    try:
        storage.delete_orphan(name, downloaded_map())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "success"}


# ── Outputs / gallery ─────────────────────────────────────────────────────────

def _write_output_meta(video_path: str, req: GenerateRequest, job_id: str, params: dict, elapsed: float):
    meta = {
        "job_id": job_id, "prompt": req.prompt, "model_id": req.model_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "settings": {k: params.get(k) for k in ("width", "height", "num_frames", "fps", "steps", "seed", "profile")},
        "start_image": req.start_image,
    }
    Path(video_path).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


@app.get("/outputs")
def list_outputs():
    items = []
    for mp4 in sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not storage.OUTPUT_NAME_RE.match(mp4.name):
            continue
        meta = {}
        meta_path = mp4.with_suffix(".json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        items.append({
            "name": mp4.name, "path": str(mp4), "bytes": mp4.stat().st_size,
            "created_at": meta.get("created_at") or datetime.fromtimestamp(mp4.stat().st_mtime).isoformat(timespec="seconds"),
            "prompt": meta.get("prompt", ""), "model_id": meta.get("model_id", ""),
            "job_id": meta.get("job_id", mp4.stem), "settings": meta.get("settings", {}),
            "enhanced_from": meta.get("enhanced_from"), "enhance_target": meta.get("enhance_target"),
            "elapsed_seconds": meta.get("elapsed_seconds"),
            "edit": meta.get("edit"),
            **{k: v for k, v in editor.probe(mp4).items()},
        })
    return items


@app.get("/images")
def list_images():
    items = []
    storage.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for img in sorted(storage.IMAGE_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not storage.IMAGE_NAME_RE.match(img.name):
            continue
        meta = {}
        if img.with_suffix(".json").exists():
            try:
                meta = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
            except ValueError:
                pass
        items.append({"name": img.name, "bytes": img.stat().st_size, "prompt": meta.get("prompt", ""),
                      "model_id": meta.get("model_id", ""), "settings": meta.get("settings", {}),
                      "uploaded": meta.get("uploaded", False),
                      "created_at": meta.get("created_at") or datetime.fromtimestamp(img.stat().st_mtime).isoformat(timespec="seconds")})
    return items


@app.get("/images/{name}/file")
def image_file(name: str):
    try:
        path = storage.image_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


@app.delete("/images/{name}")
def delete_image(name: str):
    try:
        path = storage.image_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"status": "success"}


@app.post("/images/upload")
def upload_image(body: ImageUpload):
    """Add a picture (logo, product shot, reference) to the image gallery."""
    import base64
    import io
    from PIL import Image
    m = re.match(r"^data:image/(png|jpeg|jpg|webp);base64,(.+)$", body.data_url, re.S)
    if not m:
        raise HTTPException(422, "Upload a PNG, JPEG or WebP image.")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(m.group(2), validate=True)))
        img.load()
    except Exception:
        raise HTTPException(422, "That file isn't a readable image.")
    if img.width * img.height > 40_000_000:
        raise HTTPException(413, "That image is too large.")
    storage.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    name = f"upload_{uuid.uuid4().hex[:8]}.png"
    img.save(storage.IMAGE_DIR / name)  # re-encoding drops anything that isn't pixels
    (storage.IMAGE_DIR / name).with_suffix(".json").write_text(json.dumps({
        "uploaded": True, "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"width": img.width, "height": img.height}}), encoding="utf-8")
    return {"name": name, "width": img.width, "height": img.height}


@app.get("/outputs/{name}/file")
def output_file(name: str):
    """Streams a gallery video (with byte ranges for seeking). Only names
    matching the output pattern inside OUTPUT_DIR can be served."""
    try:
        path = storage.output_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="video/mp4")


@app.delete("/outputs/{name}")
def delete_output(name: str):
    try:
        path = storage.output_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"status": "success"}


@app.post("/outputs/{name}/reveal")
def reveal_output(name: str):
    try:
        path = storage.output_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not path.exists():
        raise HTTPException(404, "Not found")
    subprocess.Popen(["explorer", f"/select,{path}"])
    return {"status": "success"}


@app.post("/folders/{which}/open")
def open_folder(which: str):
    folders = {"models": MODELS_DIR, "outputs": OUTPUT_DIR}
    log_dir = os.environ.get("NEURALCUT_LOG_DIR")
    if log_dir:
        folders["logs"] = Path(log_dir)
    if which not in folders:
        raise HTTPException(404, "Unknown folder")
    folders[which].mkdir(parents=True, exist_ok=True)
    os.startfile(str(folders[which]))  # fixed, known folders only
    return {"status": "success", "path": str(folders[which])}


# ── Generation queue ──────────────────────────────────────────────────────────
# One heavy job at a time: video models use nearly all VRAM and most RAM, so
# running two would only make both crawl or crash.

WORKER_SCRIPT = Path(__file__).parent / "generate_worker.py"


class Worker:
    """A generation process plus a thread that keeps its stderr drained."""

    def __init__(self, args: list[str], warm: bool):
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), *args],
            stdin=subprocess.PIPE if warm else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", bufsize=1, env=dict(os.environ),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.stderr_tail: list[str] = []
        self.label = "warm" if warm else "cold"
        self.job_id: Optional[str] = None
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self.proc.stderr:
            self.stderr_tail.append(line)
            del self.stderr_tail[:-60]
            print(f"[worker:{self.label}] {line}", end="", flush=True)
            if self.job_id:
                publish_log(self.job_id, line)

    def alive(self):
        return self.proc.poll() is None


class WarmPool:
    """Keeps one worker pre-started with torch/diffusers already imported, so a
    job doesn't wait ~40s for imports. Each worker still serves a single job."""

    def __init__(self):
        self.lock = threading.Lock()
        self.spare: Optional[Worker] = None

    def prewarm(self):
        # A warm worker holds ~1.5GB of RAM; don't take it from something else
        # that is running (a game, another app). Jobs then start cold instead.
        try:
            import psutil
            if psutil.virtual_memory().available < 6 * 1024 ** 3:
                print("[NeuralCut] Low free RAM; not pre-starting a worker", flush=True)
                return
        except Exception:
            pass
        with self.lock:
            if self.spare is None or not self.spare.alive():
                self.spare = Worker(["--serve"], warm=True)

    def take(self, params_file: Path, strategy: str, job_id: str) -> Worker:
        with self.lock:
            w, self.spare = self.spare, None
        if w is not None and w.alive():
            w.job_id = job_id
            try:
                job = json.dumps({"params_path": str(params_file), "strategy": strategy})
                w.proc.stdin.write(job + "\n")
                w.proc.stdin.flush()
                w.proc.stdin.close()
                w.label = "job"
                return w
            except OSError:
                pass
        w = Worker([str(params_file), "--strategy", strategy], warm=False)
        w.job_id = job_id
        return w

    def shutdown(self):
        with self.lock:
            if self.spare and self.spare.alive():
                self.spare.proc.kill()
            self.spare = None


warm_pool = WarmPool()
generation_queue: "queue.Queue[tuple[str, GenerateRequest]]" = queue.Queue()
active_processes: dict[str, subprocess.Popen] = {}
cancelled_jobs: set[str] = set()


gpu_job_running = threading.Event()


def generation_queue_worker():
    while True:
        job_id, req = generation_queue.get()
        try:
            if job_id in cancelled_jobs:
                continue
            # One heavy model at a time: the agent finishes its reply, then sleeps.
            gpu_job_running.set()
            llm.server.sleep_for_gpu_job()
            if isinstance(req, EnhanceRequest):
                run_enhance_job(job_id, req)
            elif isinstance(req, VoiceRequest):
                run_voice_job(job_id, req)
            else:
                run_generation_subprocess(job_id, req)
        except Exception as e:
            broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "error",
                                   "progress": 0.0, "eta": 0, "outputPath": None, "error": str(e)})
        finally:
            if generation_queue.empty():
                gpu_job_running.clear()
            generation_queue.task_done()


threading.Thread(target=generation_queue_worker, daemon=True).start()


def _order_strategies(model_id: str) -> tuple[list[dict], dict]:
    profile = load_runtime_profile(model_id)
    ladder = get_strategy_ladder(model_id)
    verdicts = profile.get("verdicts", {})
    preferred = profile.get("preferred_strategy")

    def should_try(name):
        v = verdicts.get(name, {})
        if v.get("status") in (None, "working", "untested"):
            return True
        return v.get("sidecar_build_last_tested") != WORKER_BUILD  # retry failures on new builds

    ordered = [s for s in ladder if s["name"] == preferred and verdicts.get(preferred, {}).get("status") == "working"]
    ordered += [s for s in ladder if s not in ordered and should_try(s["name"])]
    if not ordered:
        # Never re-attempt strategies that ran out of memory on this hardware.
        ordered = [s for s in ladder if verdicts.get(s["name"], {}).get("status") != "failed_oom"] or list(ladder)
    return ordered, profile


def run_job(job_id: str, model_id: str, params: dict, on_done):
    """Run one job in a worker process and announce the outcome. Each strategy
    attempt runs in a fresh Python process, so CUDA crashes and leaked memory
    can't carry over between attempts or jobs. `on_done(output_path, seconds)`
    runs before "done" is announced and may return extra fields for the event."""
    started = time.time()
    params_file = Path(tempfile.gettempdir()) / f"neuralcut_job_{job_id}.json"
    params_file.write_text(json.dumps(params), encoding="utf-8")
    ordered, profile = _order_strategies(model_id)
    print(f"[NeuralCut] Job {job_id}: strategies {[s['name'] for s in ordered]}", flush=True)
    last_error = "No strategies attempted"
    output_path = None

    try:
        for strategy in ordered:
            name = strategy["name"]
            if job_id in cancelled_jobs:
                break
            broadcast_from_thread({"type": "strategy_attempt", "job_id": job_id, "strategy": name, "status": "starting"})
            worker = warm_pool.take(params_file, name, job_id)
            proc = worker.proc
            active_processes[job_id] = proc
            stderr_tail = worker.stderr_tail
            saw_done = False
            try:
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"[worker:{job_id}] {line}", flush=True)
                        publish_log(job_id, line)
                        continue
                    if payload.get("type") == "worker_ready":
                        continue
                    status = payload.get("status")
                    if status == "done":
                        saw_done = True
                        output_path = payload.get("outputPath")
                        continue  # announced below, after metadata is written
                    if status == "error":
                        last_error = payload.get("error") or "unknown error"
                        continue  # the ladder may still recover; final error sent below
                    broadcast_from_thread(payload)
            finally:
                proc.wait()
                time.sleep(0.2)  # let the drain thread catch the last stderr lines
                active_processes.pop(job_id, None)

            if job_id in cancelled_jobs:
                break
            if saw_done and output_path:
                profile.setdefault("verdicts", {})[name] = {
                    "status": "working", "sidecar_build_last_tested": WORKER_BUILD,
                    "tested_at": datetime.now().isoformat(), "last_error": None,
                }
                profile["preferred_strategy"] = name
                save_runtime_profile(model_id, profile)
                break

            tail = "".join(stderr_tail).lower()
            code = proc.returncode
            if code in (3221225477, -1073741819):
                category, desc = "native_crash", "The GPU driver crashed (access violation)."
            elif "out of memory" in tail or "outofmemoryerror" in tail or "out of memory" in last_error.lower():
                category, desc = "oom", "Ran out of GPU memory."
            else:
                category, desc = "error", last_error if last_error != "No strategies attempted" else f"Exit code {code}"
            last_error = desc
            profile.setdefault("verdicts", {})[name] = {
                "status": f"failed_{category}", "sidecar_build_last_tested": WORKER_BUILD,
                "tested_at": datetime.now().isoformat(), "last_error": desc[:500], "exit_code": code,
            }
            save_runtime_profile(model_id, profile)
            broadcast_from_thread({"type": "strategy_attempt", "job_id": job_id, "strategy": name,
                                   "status": "failed", "category": category, "error": desc[:300]})
            time.sleep(2.0)  # let the driver reclaim VRAM before the next attempt
    finally:
        params_file.unlink(missing_ok=True)
        # Warm up the next worker while the user looks at this result.
        threading.Thread(target=warm_pool.prewarm, daemon=True).start()

    if job_id in cancelled_jobs:
        broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "cancelled",
                               "progress": 0.0, "eta": 0, "outputPath": None, "error": None})
        _notify_production(job_id, None, "Cancelled")
    elif output_path:
        elapsed = time.time() - started
        extra = on_done(output_path, elapsed) or {}
        broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "done", "progress": 100.0,
                               "eta": 0, "outputPath": output_path, "error": None,
                               "elapsed_seconds": round(elapsed, 1), **extra})
        _notify_production(job_id, output_path, None)
    else:
        broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "error", "progress": 0.0,
                               "eta": 0, "outputPath": None, "error": last_error})
        _notify_production(job_id, None, last_error)


def _notify_production(job_id: str, output_path: Optional[str], error: Optional[str]):
    try:
        production.on_job_finished(job_id, output_path, error)
    except Exception as e:
        print(f"[NeuralCut] production update failed for {job_id}: {e}", flush=True)


def run_generation_subprocess(job_id: str, req: GenerateRequest):
    resolved = resolve_generation_params(req.model_id, req.model_dump())
    params = {**req.model_dump(), **resolved, "job_id": job_id}
    if req.start_image:
        params["start_image"] = str(storage.image_path(req.start_image))
    if req.end_image:
        params["end_image"] = str(storage.image_path(req.end_image))

    def on_done(output_path: str, elapsed: float):
        _write_output_meta(output_path, req, job_id, params, elapsed)
        telemetry.record_timing(req.model_id, params, elapsed)
        if req.enhancer_id:
            try:
                child = queue_enhance(EnhanceRequest(source=Path(output_path).name, enhancer_id=req.enhancer_id,
                                                     target=req.enhance_target or "1080p"), parent=job_id)
                return {"enhance_job_id": child}
            except HTTPException as e:  # the video itself is fine; just report why it wasn't enhanced
                print(f"[NeuralCut] Auto-enhance skipped for {job_id}: {e.detail}", flush=True)

    run_job(job_id, req.model_id, params, on_done)


def _enhance_target(enhancer_id: str, target: str) -> dict:
    targets = MODEL_CONFIG[enhancer_id].get("enhance", {}).get("targets", [])
    return next((t for t in targets if t["id"] == target), targets[0] if targets else {"id": "1080p", "short_edge": 1080})


def run_enhance_job(job_id: str, req: EnhanceRequest):
    src = storage.output_path(req.source)
    src_meta = {}
    if src.with_suffix(".json").exists():
        try:
            src_meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
        except ValueError:
            pass
    target = _enhance_target(req.enhancer_id, req.target)
    params = {"job_id": job_id, "model_id": req.enhancer_id, "source": req.source,
              "target_short_edge": target["short_edge"], "prompt": src_meta.get("prompt", "")}

    def on_done(output_path: str, elapsed: float):
        frames = (src_meta.get("settings") or {}).get("num_frames") or 0
        meta = {
            **{k: v for k, v in src_meta.items() if k in ("prompt", "model_id")},
            "job_id": job_id, "created_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "enhanced_from": req.source, "enhancer_id": req.enhancer_id, "enhance_target": target["id"],
            "settings": {**(src_meta.get("settings") or {}), "enhanced": target["id"]},
        }
        Path(output_path).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        telemetry.record_timing(req.enhancer_id, {"width": target["short_edge"], "height": 1,
                                                  "num_frames": frames, "steps": 1}, elapsed)

    run_job(job_id, req.enhancer_id, params, on_done)


def queue_enhance(req: EnhanceRequest, parent: Optional[str] = None) -> str:
    if req.enhancer_id not in MODEL_CONFIG or MODEL_CONFIG[req.enhancer_id].get("identity", {}).get("kind") != "enhancer":
        raise HTTPException(404, "Unknown enhancer")
    refresh_installed(req.enhancer_id)
    if not models_db[req.enhancer_id]["downloaded"]:
        raise HTTPException(409, "Install the enhancer from the Models page first.")
    try:
        if not storage.output_path(req.source).exists():
            raise HTTPException(404, "That video no longer exists.")
    except ValueError:
        raise HTTPException(400, "Invalid video name")
    job_id = uuid.uuid4().hex[:8]
    target = _enhance_target(req.enhancer_id, req.target)
    job_states[job_id] = {"type": "job_status", "job_id": job_id, "status": "queued", "progress": 0.0,
                          "eta": 0, "outputPath": None, "error": None}
    generation_queue.put((job_id, req))
    est = telemetry.estimate_seconds(req.enhancer_id, {"width": target["short_edge"], "height": 1, "num_frames": 0, "steps": 1})
    broadcast_from_thread({
        "type": "job_created", "job_id": job_id, "parent_job_id": parent, "kind": "enhance",
        "model_id": req.enhancer_id, "source": req.source,
        "summary": f"{MODEL_CONFIG[req.enhancer_id]['identity']['display_name']} to {target['id']}",
        "estimate_seconds": est["seconds"] if est else None,
    })
    return job_id


@app.post("/enhance")
def enhance(req: EnhanceRequest):
    job_id = queue_enhance(req)
    return {"job_id": job_id, "status": "queued"}


# ── Editor ────────────────────────────────────────────────────────────────────
# Renders run on the CPU in their own lane, so stitching never waits for (or
# blocks) a GPU job. One render at a time keeps the PC responsive.

edit_lock = threading.Lock()


def run_edit_job(job_id: str, req: EditRequest, clips: list[dict], overlays: list[dict], audio_tracks: list[dict]):
    started = time.time()

    def progress(pct: float, msg: str):
        broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "post_processing",
                               "progress": round(pct, 1), "eta": 0, "message": msg,
                               "outputPath": None, "error": None})
        publish_log(job_id, msg)

    def track(proc):
        active_processes[job_id] = proc

    with edit_lock:
        if job_id in cancelled_jobs:
            broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "cancelled",
                                   "progress": 0.0, "eta": 0, "outputPath": None, "error": None})
            return
        out = OUTPUT_DIR / f"video_{job_id}.mp4"
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            result = editor.render(clips, out, req.transition, req.fade_seconds, req.fade_edges,
                                   req.resolution, progress, track, overlays=overlays, audio_tracks=audio_tracks)
        except Exception as e:
            cancelled = job_id in cancelled_jobs
            broadcast_from_thread({"type": "job_status", "job_id": job_id,
                                   "status": "cancelled" if cancelled else "error", "progress": 0.0, "eta": 0,
                                   "outputPath": None, "error": None if cancelled else str(e)})
            return
        finally:
            active_processes.pop(job_id, None)
        elapsed = time.time() - started
        meta = {
            "job_id": job_id, "model_id": "editor", "created_at": datetime.now().isoformat(timespec="seconds"),
            "prompt": req.title.strip() or f"Edit of {len(clips)} clip{'s' if len(clips) != 1 else ''}",
            "elapsed_seconds": round(elapsed, 1),
            "edit": {"clips": [{"name": c["name"], "start": c["start"], "end": c["end"]} for c in clips],
                     "transition": req.transition, "fade_seconds": req.fade_seconds,
                     "fade_edges": req.fade_edges, "resolution": req.resolution},
            "settings": {"width": result["width"], "height": result["height"], "fps": editor.FPS,
                         "num_frames": int(result["duration"] * editor.FPS)},
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "done", "progress": 100.0,
                               "eta": 0, "outputPath": str(out), "error": None,
                               "elapsed_seconds": round(elapsed, 1)})


@app.post("/edit/render")
def edit_render(req: EditRequest):
    clips = []
    for c in req.clips:
        try:
            path = storage.output_path(c.name)
        except ValueError:
            raise HTTPException(400, f"Invalid video name: {c.name}")
        if not path.exists():
            raise HTTPException(404, f"{c.name} no longer exists.")
        info = editor.probe(path)
        if not info["duration"] or not info["width"]:
            raise HTTPException(422, f"{c.name} couldn't be read as a video.")
        end = min(c.end if c.end is not None else info["duration"], info["duration"])
        start = min(c.start, max(0.0, end - 0.1))
        if end - start < 0.1:
            raise HTTPException(422, f"{c.name}: the trimmed part is too short.")
        clips.append({"name": c.name, "path": path, "start": round(start, 3), "end": round(end, 3), "info": info,
                      "speed": c.speed, "volume": c.volume})
    overlays = []
    for ov in req.overlays:
        item = ov.model_dump()
        if ov.kind == "image":
            try:
                ipath = storage.image_path(ov.image or "")
            except ValueError:
                raise HTTPException(400, "Invalid overlay image")
            if not ipath.is_file():
                raise HTTPException(404, f"Overlay image {ov.image} no longer exists.")
            item["path"] = ipath
        elif not ov.text.strip():
            continue
        overlays.append(item)
    audio_tracks = []
    for a in req.audio:
        try:
            apath = storage.audio_path(a.name)
        except ValueError:
            raise HTTPException(400, "Invalid audio name")
        if not apath.is_file():
            raise HTTPException(404, f"Audio {a.name} no longer exists.")
        audio_tracks.append({"path": apath, "start": a.start, "volume": a.volume})
    job_id = uuid.uuid4().hex[:8]
    job_states[job_id] = {"type": "job_status", "job_id": job_id, "status": "queued", "progress": 0.0,
                          "eta": 0, "outputPath": None, "error": None}
    threading.Thread(target=run_edit_job, args=(job_id, req, clips, overlays, audio_tracks), daemon=True).start()
    durs = [(c["end"] - c["start"]) / c["speed"] for c in clips]
    total = sum(durs)
    if req.transition == "fade" and len(durs) > 1:
        total -= max(0.1, min(req.fade_seconds, min(durs) / 2)) * (len(durs) - 1)
    return {"job_id": job_id, "status": "queued", "duration": round(total, 2)}


@app.post("/generate")
def generate(req: GenerateRequest):
    if req.model_id not in MODEL_CONFIG:
        raise HTTPException(404, f"Unknown model: {req.model_id}")
    if MODEL_CONFIG[req.model_id].get("identity", {}).get("kind") == "enhancer":
        raise HTTPException(400, "That's an enhancer, not a video model.")
    refresh_installed(req.model_id)
    if not models_db[req.model_id]["downloaded"]:
        raise HTTPException(409, "This model isn't downloaded yet.")
    if not req.prompt.strip():
        raise HTTPException(422, "Please describe the video you want.")
    caps = MODEL_CONFIG[req.model_id].get("capabilities", {})
    for extra in filter(None, [req.end_image]):
        try:
            if not storage.image_path(extra).is_file():
                raise HTTPException(404, "That ending image no longer exists.")
        except ValueError:
            raise HTTPException(400, "Invalid image name")
    if req.start_image:
        try:
            if not storage.image_path(req.start_image).is_file():
                raise HTTPException(404, "That starting image no longer exists.")
        except ValueError:
            raise HTTPException(400, "Invalid image name")
    elif caps.get("image_to_video") and not caps.get("text_to_video"):
        raise HTTPException(422, "This model animates an image. Choose a starting image first.")
    if req.enhancer_id and not models_db.get(req.enhancer_id, {}).get("downloaded"):
        raise HTTPException(409, "The selected enhancer isn't installed. Install it from Models, or turn Enhance off.")
    job_id = uuid.uuid4().hex[:8]
    job_states[job_id] = {"type": "job_status", "job_id": job_id, "status": "queued", "progress": 0.0,
                          "eta": 0, "outputPath": None, "error": None}
    generation_queue.put((job_id, req))
    est = telemetry.estimate_seconds(req.model_id, resolve_generation_params(req.model_id, req.model_dump()))
    return {"job_id": job_id, "status": "queued", "position": generation_queue.qsize(),
            "estimate_seconds": est["seconds"] if est else None}


@app.get("/jobs")
def list_jobs():
    return job_states


@app.get("/jobs/{job_id}/logs")
def get_job_logs(job_id: str):
    return telemetry.job_logs(job_id)


@app.post("/estimate")
def estimate(req: GenerateRequest):
    """How long these settings usually take on this PC (None until it has history)."""
    if req.model_id not in MODEL_CONFIG:
        raise HTTPException(404, "Unknown model")
    return telemetry.estimate_seconds(req.model_id, resolve_generation_params(req.model_id, req.model_dump()))


@app.get("/system/stats")
def system_stats():
    return telemetry.stats_history()


@app.post("/generate/{job_id}/cancel")
def cancel_generation(job_id: str):
    cancelled_jobs.add(job_id)
    proc = active_processes.get(job_id)
    if proc:
        # taskkill /T also stops any child processes the worker started.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "cancelled",
                               "progress": 0.0, "eta": 0, "outputPath": None, "error": None})
    return {"status": "success"}


# ── Info routes ───────────────────────────────────────────────────────────────

# ── Agent ─────────────────────────────────────────────────────────────────────

def _chat_models_installed() -> list[str]:
    ids = [mid for mid, m in MODEL_CONFIG.items() if m.get("identity", {}).get("kind") == "chat"]
    for mid in ids:
        refresh_installed(mid)
    return sorted((mid for mid in ids if models_db[mid]["downloaded"]),
                  key=lambda mid: -MODEL_CONFIG[mid].get("ui", {}).get("quality_rank", 0))


def _agent_model(requested: Optional[str] = None) -> str:
    installed = _chat_models_installed()
    for choice in (requested, agent.load_settings().get("agent_model")):
        if choice in installed:
            return choice
    if not installed:
        raise HTTPException(409, "Install a chat model on the Models page to use the agent.")
    # The best installed model that fits this PC; otherwise the smallest one.
    vram, ram = telemetry.hardware_gb()
    fits = [m for m in installed if MODEL_CONFIG[m]["hardware"].get("minimum_vram_gb", 0) <= vram + 0.5
            and MODEL_CONFIG[m]["hardware"].get("minimum_ram_gb", 0) <= ram + 0.5]
    return (fits or installed[::-1])[0]


def _agent_state(state: str, model_id: Optional[str]):
    broadcast_from_thread({"type": "agent_state", "state": state, "model_id": model_id})


llm.server.on_state = _agent_state
llm.server.gpu_busy = gpu_job_running.is_set


@app.get("/agent/status")
def agent_status():
    installed = _chat_models_installed()
    model = None
    try:
        model = _agent_model() if installed else None
    except HTTPException:
        pass
    return {"state": llm.server.state, "loaded_model": llm.server.model_id if llm.server.running() else None,
            "model_id": model, "installed": installed, "gpu_busy": gpu_job_running.is_set()}


@app.get("/agent/conversations")
def agent_conversations():
    return agent.list_conversations()


@app.post("/agent/conversations")
def agent_new_conversation():
    return agent.new_conversation()


@app.get("/agent/conversations/{conv_id}")
def agent_get_conversation(conv_id: str):
    try:
        return agent.load_conversation(conv_id)
    except (ValueError, OSError):
        raise HTTPException(404, "Conversation not found")


@app.delete("/agent/conversations/{conv_id}")
def agent_delete_conversation(conv_id: str):
    try:
        agent.delete_conversation(conv_id)
    except ValueError:
        raise HTTPException(404, "Conversation not found")
    return {"status": "success"}


@app.post("/agent/conversations/{conv_id}/message")
def agent_message(conv_id: str, msg: AgentMessage):
    if not msg.text.strip() and not msg.images:
        raise HTTPException(422, "Write a message or attach an image.")
    if sum(len(u) for u in msg.images) > 12_000_000:
        raise HTTPException(413, "Those images are too large. Try fewer or smaller ones.")
    if any(not u.startswith("data:image/") for u in msg.images):
        raise HTTPException(422, "Attachments must be images.")
    model_id = _agent_model(msg.model_id)
    text = msg.text
    # Attachments also go to the image gallery, so the agent can use them (logo, product, reference) by name.
    saved = []
    for u in msg.images:
        try:
            saved.append(upload_image(ImageUpload(data_url=u))["name"])
        except HTTPException:
            pass
    if saved:
        text = (text + "\n\n" if text else "") + "[Attached images saved in the gallery as: " + ", ".join(saved) + "]"
    try:
        agent.start_turn(conv_id, model_id, text, msg.images, broadcast_from_thread)
    except (ValueError, OSError):
        raise HTTPException(404, "Conversation not found")
    return {"status": "started", "model_id": model_id}


@app.post("/agent/conversations/{conv_id}/cancel")
def agent_cancel(conv_id: str):
    agent.cancel_turn(conv_id)
    return {"status": "success"}


@app.post("/agent/sleep")
def agent_sleep():
    threading.Thread(target=llm.server.sleep_for_gpu_job, daemon=True).start()
    return {"status": "success"}


@app.get("/settings")
def get_settings():
    s = agent.load_settings()
    key = s.pop("brave_api_key", "")
    return {**s, "brave_api_key_set": bool(key)}


@app.put("/settings")
def put_settings(update: SettingsUpdate):
    values = {k: v for k, v in update.model_dump().items() if v is not None}
    if values.get("agent_model") and values["agent_model"] not in MODEL_CONFIG:
        raise HTTPException(404, "Unknown model")
    agent.save_settings(values)
    return get_settings()


# Tools that let the agent drive the app itself.

def _model_kind(m: dict) -> str:
    return m.get("identity", {}).get("kind", "video")


def _tool_list_models(kind: str = "") -> dict:
    out = []
    for mid, m in MODEL_CONFIG.items():
        k = _model_kind(m)
        if kind and k != kind:
            continue
        refresh_installed(mid)
        if not models_db[mid]["downloaded"]:
            continue
        gen = m.get("generation", {})
        fps = gen.get("defaults", {}).get("fps", 24) or 24
        out.append({
            "id": mid, "kind": k, "name": m["identity"]["display_name"],
            "speed": m.get("ui", {}).get("speed_label"), "description": m.get("ui", {}).get("description"),
            "makes_sound": bool(m.get("capabilities", {}).get("audio_generation")),
            "presets": {p: {kk: v for kk, v in cfg.items() if kk in ("label", "width", "height", "num_frames")}
                        for p, cfg in (m.get("profiles") or {}).items()},
            "max_seconds": round((gen.get("limits", {}).get("num_frames", {}).get("max", 0) or 0) / fps, 1),
        })
    return {"installed_models": out}


def _tool_generate_video(prompt: str, model_id: str = "", preset: str = "balanced",
                         seconds: float = 0, enhance: bool = False) -> dict:
    videos = [mid for mid, m in MODEL_CONFIG.items() if _model_kind(m) == "video"]
    for mid in videos:
        refresh_installed(mid)
    installed = [mid for mid in videos if models_db[mid]["downloaded"]]
    if not model_id:
        model_id = next((m for m in installed if MODEL_CONFIG[m].get("ui", {}).get("recommended")),
                        installed[0] if installed else "")
    if model_id not in installed:
        return {"error": f"Video model '{model_id}' isn't installed. Call list_models first."}
    cfg = MODEL_CONFIG[model_id]
    fps = cfg.get("generation", {}).get("defaults", {}).get("fps", 24)
    frames = int(seconds * fps) + 1 if seconds else None
    enhancer = None
    if enhance:
        enhancer = next((mid for mid, m in MODEL_CONFIG.items() if _model_kind(m) == "enhancer"
                         and models_db.get(mid, {}).get("downloaded") and m.get("ui", {}).get("recommended")), None)
    req = GenerateRequest(prompt=prompt, model_id=model_id, profile=preset, num_frames=frames,
                          enhancer_id=enhancer, enhance_target="1080p" if enhancer else None)
    try:
        res = generate(req)
    except HTTPException as e:
        return {"error": e.detail}
    resolved = resolve_generation_params(model_id, req.model_dump())
    broadcast_from_thread({
        "type": "job_created", "job_id": res["job_id"], "kind": "generate", "model_id": model_id,
        "prompt": prompt, "by_agent": True, "estimate_seconds": res.get("estimate_seconds"),
        "summary": f"{resolved.get('width')}×{resolved.get('height')} · {(resolved.get('num_frames') or 0) / fps:.1f}s",
    })
    return {"job_id": res["job_id"], "status": "queued", "model": model_id, "position_in_queue": res["position"],
            "estimated_seconds": res.get("estimate_seconds"), "will_enhance": bool(enhancer)}


def _tool_list_jobs() -> dict:
    return {"jobs": [{"job_id": j, "status": st.get("status"), "progress": st.get("progress"),
                      "message": st.get("message"), "error": st.get("error"),
                      "output": Path(st["outputPath"]).name if st.get("outputPath") else None}
                     for j, st in list(job_states.items())[-20:]]}


agent.register_tool(agent.Tool(
    "list_models", "List the AI models installed in the app (video, image, voice, chat, enhancer), with presets and limits.",
    {"type": "object", "properties": {"kind": {"type": "string", "enum": ["", "video", "image", "voice", "chat", "enhancer"],
                                               "description": "Only this kind; empty for all"}}},
    _tool_list_models))
agent.register_tool(agent.Tool(
    "generate_video", "Queue one video clip. Returns immediately with a job id; the clip is made in the background "
    "(several minutes each). Use detailed prompts.",
    {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "Detailed visual (and sound, if the model makes sound) description"},
        "model_id": {"type": "string", "description": "Installed video model id from list_models; empty for the default"},
        "preset": {"type": "string", "enum": ["fast", "balanced", "detailed"], "description": "Quality preset"},
        "seconds": {"type": "number", "description": "Clip length in seconds; 0 for the model's default"},
        "enhance": {"type": "boolean", "description": "Upscale to 1080p afterwards (much slower)"}},
     "required": ["prompt"]},
    _tool_generate_video))
agent.register_tool(agent.Tool(
    "list_jobs", "Show the status of recent generation jobs.", {"type": "object", "properties": {}}, _tool_list_jobs))


# ── Productions (agent plans, engine executes) ────────────────────────────────

ASPECTS = {"landscape": (16, 9), "portrait": (9, 16), "square": (1, 1)}


def _installed(kind: str, need: Optional[str] = None) -> list[str]:
    """Installed models of a kind (optionally with a capability), best first."""
    out = []
    for mid, m in MODEL_CONFIG.items():
        if _model_kind(m) != kind or (need and not m.get("capabilities", {}).get(need)):
            continue
        refresh_installed(mid)
        if models_db[mid]["downloaded"]:
            out.append(mid)
    return sorted(out, key=lambda mid: -MODEL_CONFIG[mid].get("ui", {}).get("quality_rank", 0))


def _announce(job_id: str, kind: str, model_id: str, prompt: str, summary: str, est=None):
    broadcast_from_thread({"type": "job_created", "job_id": job_id, "kind": kind, "model_id": model_id,
                           "prompt": prompt, "by_agent": True, "summary": summary, "estimate_seconds": est})


def queue_image_job(prompt: str, opts: dict) -> str:
    models = _installed("image")
    model_id = opts.get("image_model") if opts.get("image_model") in models else (models[0] if models else None)
    if not model_id:
        raise RuntimeError("No image model is installed. Install one from Models > Images.")
    preset = opts.get("image_preset", "balanced")
    cfg = MODEL_CONFIG[model_id]
    base = (cfg.get("profiles") or {}).get(preset, {})
    long_side = max(base.get("width", 1344), base.get("height", 768))
    aw, ah = ASPECTS.get(opts.get("aspect", "landscape"), (16, 9))
    w = long_side if aw >= ah else round(long_side * aw / ah / 64) * 64
    h = round(long_side * ah / aw / 64) * 64 if aw >= ah else long_side
    res = generate(GenerateRequest(prompt=prompt, model_id=model_id, profile=preset, width=w, height=h,
                                   seed=opts.get("seed")))
    _announce(res["job_id"], "image", model_id, prompt, f"Image {w}×{h}")
    return res["job_id"]


def queue_video_job(prompt: str, start_image: Optional[str], opts: dict) -> str:
    i2v = _installed("video", "image_to_video")
    t2v = _installed("video", "text_to_video")
    model_id = opts.get("video_model")
    if model_id not in (i2v + t2v):
        model_id = (i2v[0] if (start_image and i2v) else t2v[0] if t2v else None)
    if not model_id:
        raise RuntimeError("No video model is installed.")
    cfg = MODEL_CONFIG[model_id]
    uses_image = bool(start_image) and cfg.get("capabilities", {}).get("image_to_video")
    fps = cfg.get("generation", {}).get("defaults", {}).get("fps", 24)
    seconds = float(opts.get("seconds") or 5)
    req = GenerateRequest(prompt=prompt, model_id=model_id, profile=opts.get("video_preset", "balanced"),
                          num_frames=int(seconds * fps) + 1, start_image=start_image if uses_image else None,
                          seed=opts.get("seed"))
    res = generate(req)
    r = resolve_generation_params(model_id, req.model_dump())
    _announce(res["job_id"], "generate", model_id, prompt,
              f"{'From image · ' if uses_image else ''}{r.get('width')}×{r.get('height')} · {(r.get('num_frames') or 0) / fps:.1f}s",
              res.get("estimate_seconds"))
    return res["job_id"]


def assemble_production(prod: dict) -> str:
    """Join a production's shots, add the voice-over, logo watermark and end card."""
    live = [sh for sh in prod["shots"] if sh.get("video") and not sh.get("error")]
    clips = []
    for sh in live:
        path = storage.output_path(sh["video"])
        info = editor.probe(path)
        clips.append({"name": sh["video"], "path": path, "start": 0.0, "end": info["duration"], "info": info})
    size = (clips[0]["info"]["width"], clips[0]["info"]["height"])
    job_id = "prod" + uuid.uuid4().hex[:8]
    logo_name = (prod.get("options") or {}).get("logo")
    logo = None
    if logo_name:
        try:
            lp = storage.image_path(logo_name)
            logo = lp if lp.is_file() else None
        except ValueError:
            logo = None
    vo = prod.get("voiceover") or {}
    voice_len = 0.0
    tracks = []
    if vo.get("audio"):
        apath = storage.audio_path(vo["audio"])
        voice_len = editor.probe(apath)["duration"] or 0.0
        tracks.append({"path": apath, "start": 0.5, "volume": 1.0})
    shots_len = sum(c["end"] for c in clips) - 0.4 * (len(clips) - 1)
    card = prod.get("end_card") or {}
    if card.get("lines") or logo:
        png = OUTPUT_DIR / "images" / f"endcard_{job_id}.png"
        editor.make_end_card(card.get("lines") or [prod["title"]], logo, png, size)
        # Long enough to finish the voice-over, never shorter than 4 s.
        card_len = max(4.0, voice_len + 0.5 - shots_len + 1.5)
        cpath = OUTPUT_DIR / f"video_card_{job_id}.mp4"
        editor.make_card_clip(png, card_len, cpath, size)
        clips.append({"name": cpath.name, "path": cpath, "start": 0.0, "end": card_len, "info": editor.probe(cpath)})
    overlays = []
    if logo:
        overlays.append({"kind": "image", "path": logo, "start": 0.6, "end": max(1.0, shots_len - 0.3),
                         "x": 0.935, "y": 0.13, "width": 0.07, "opacity": 0.9, "fade": 0.4})
    out = OUTPUT_DIR / f"video_{job_id}.mp4"
    result = editor.render(clips, out, "fade", 0.4, True, "auto", lambda pct, msg: None,
                           overlays=overlays, audio_tracks=tracks)
    out.with_suffix(".json").write_text(json.dumps({
        "job_id": job_id, "model_id": "editor", "prompt": prod["title"], "created_at": datetime.now().isoformat(timespec="seconds"),
        "edit": {"clips": [{"name": c["name"], "start": 0, "end": c["end"]} for c in clips], "transition": "fade"},
        "production": prod["id"],
        "settings": {"width": result["width"], "height": result["height"], "fps": editor.FPS,
                     "num_frames": int(result["duration"] * editor.FPS)}}, indent=2), encoding="utf-8")
    return out.name


def _queue_voice_for_production(script: str, description: str) -> str:
    models = _installed("voice")
    designers = [m for m in models if MODEL_CONFIG[m].get("capabilities", {}).get("voice_design")]
    if designers:
        req = VoiceRequest(model_id=designers[0], text=script,
                           instruct=description or "A warm, confident, natural voice-over artist, relaxed conversational delivery.")
    elif models:
        req = VoiceRequest(model_id=models[0], text=script, voice="am_michael")
    else:
        raise RuntimeError("No voice model is installed.")
    return voices_speak(req)["job_id"]


def _queue_motion_for_production(image: str, move: str, seconds: float) -> str:
    return animate_image(image, AnimateRequest(move=move if move in (
        "push_in", "pull_out", "pan_left", "pan_right", "rise", "orbit_left", "orbit_right", "drift") else "push_in",
        seconds=max(1.0, min(20.0, seconds))))["job_id"]


def _audio_duration(name: str) -> float:
    return editor.probe(storage.audio_path(name))["duration"] or 0.0


production.hooks.queue_image = queue_image_job
production.hooks.queue_video = queue_video_job
production.hooks.queue_voice = lambda script, desc: _queue_voice_for_production(script, desc)
production.hooks.queue_motion = lambda image, move, seconds: _queue_motion_for_production(image, move, seconds)
production.hooks.audio_duration = lambda name: _audio_duration(name)
production.hooks.assemble = assemble_production
production.hooks.notify = lambda conv_id, text: agent.post_notice(conv_id, text, broadcast_from_thread)
production.hooks.broadcast = broadcast_from_thread


@app.get("/productions")
def productions_list():
    return production.list_all()


@app.get("/productions/{pid}")
def productions_get(pid: str):
    try:
        return production.summary(production.load(pid))
    except (ValueError, OSError):
        raise HTTPException(404, "Production not found")


@app.post("/productions/{pid}/cancel")
def productions_cancel(pid: str):
    try:
        prod = production.cancel(pid)
    except (ValueError, OSError):
        raise HTTPException(404, "Production not found")
    jobs = [j for s in prod["shots"] for j in (s.get("image_job"), s.get("video_job"))]
    jobs.append((prod.get("voiceover") or {}).get("job"))
    for j in jobs:
        if j and job_states.get(j, {}).get("status") not in ("done", "error", "cancelled"):
            try:
                cancel_generation(j)
            except HTTPException:
                pass
    return production.summary(prod)


def _tool_generate_image(prompt: str, aspect: str = "landscape", preset: str = "balanced") -> dict:
    try:
        job_id = queue_image_job(prompt, {"aspect": aspect, "image_preset": preset})
    except (RuntimeError, HTTPException) as e:
        return {"error": getattr(e, "detail", None) or str(e)}
    return {"job_id": job_id, "status": "queued", "image_name_when_done": f"image_{job_id}.png"}


def _tool_start_production(title: str, shots: list, style: str = "", aspect: str = "landscape",
                           mode: str = "image", voiceover_script: str = "", voice_description: str = "",
                           logo_image: str = "", end_card_lines: list | None = None,
                           video_preset: str = "balanced") -> dict:
    if not shots:
        return {"error": "A production needs at least one shot."}
    clean = []
    for i, sh in enumerate(shots[:40]):
        if not isinstance(sh, dict) or not (sh.get("keyframe_prompt") or sh.get("image")):
            return {"error": f"Shot {i + 1} needs a keyframe_prompt (or an existing gallery image)."}
        if mode == "video" and not sh.get("motion_prompt"):
            return {"error": f"Shot {i + 1} needs a motion_prompt in video mode."}
        clean.append(sh)
    if not _installed("image") and not all(sh.get("image") for sh in clean):
        return {"error": "No image model is installed, so keyframes can't be made. Install one from Models > Images."}
    if mode == "image":
        refresh_installed("depth-anything-v2-small")
        if not models_db.get("depth-anything-v2-small", {}).get("downloaded"):
            return {"error": "Photo motion isn't installed (Models > Images). Use mode 'video' or ask the user to install it."}
    for name in [logo_image] + [sh.get("image") for sh in clean]:
        if name:
            try:
                if not storage.image_path(name).is_file():
                    return {"error": f"Image {name} isn't in the gallery. Check the name (uploaded files are called upload_....png)."}
            except ValueError:
                return {"error": f"'{name}' isn't a valid gallery image name."}
    try:
        prod = production.create(
            title, style, clean, {"aspect": aspect, "video_preset": video_preset, "logo": logo_image or None,
                                  "image_preset": "detailed" if mode == "image" else "balanced"},
            mode=mode, voiceover={"script": voiceover_script, "voice_description": voice_description} if voiceover_script else None,
            end_card={"lines": end_card_lines or []} if (end_card_lines or logo_image) else None,
            conv_id=getattr(agent.Turn.context, "conv_id", None))
    except (RuntimeError, HTTPException) as e:
        return {"error": getattr(e, "detail", None) or str(e)}
    eta = "a few minutes" if mode == "image" else "roughly 15-20 minutes per shot"
    return {"production_id": prod["id"], "shots": len(prod["shots"]), "mode": prod["mode"], "status": prod["status"],
            "estimated_time": eta,
            "note": "It runs by itself now; I'll post in this chat when the finished video is ready."}


agent.register_tool(agent.Tool(
    "generate_image", "Queue one image (keyframe, character design, product shot). Returns a job id; the file will be "
    "named image_<job_id>.png in the image gallery.",
    {"type": "object", "properties": {
        "prompt": {"type": "string", "description": "Detailed description: subject, look, lighting, lens, style"},
        "aspect": {"type": "string", "enum": ["landscape", "portrait", "square"]},
        "preset": {"type": "string", "enum": ["fast", "balanced", "detailed"]}},
     "required": ["prompt"]},
    _tool_generate_image))
agent.register_tool(agent.Tool(
    "start_production", "Produce a finished multi-shot video by itself (ads, trailers, explainers, stories). "
    "mode 'image' (default, fast and sharp): a 2K photoreal keyframe per shot, then a cinematic camera move, a natural "
    "voice-over, the logo and an exact end card; a 30 s video takes a few minutes. mode 'video': each keyframe is animated "
    "by a video model so people really move; much slower (15-20 min per shot). Use only after the user approved the plan.",
    {"type": "object", "properties": {
        "title": {"type": "string"},
        "mode": {"type": "string", "enum": ["image", "video"]},
        "style": {"type": "string", "description": "Look shared by every shot, appended to all prompts (e.g. 'premium commercial photograph, natural light, photorealistic')"},
        "aspect": {"type": "string", "enum": ["landscape", "portrait", "square"]},
        "voiceover_script": {"type": "string", "description": "The full narration, written to be spoken (numbers in words). Leave empty for none."},
        "voice_description": {"type": "string", "description": "e.g. 'a warm, confident male narrator in his 40s, relaxed and conversational'"},
        "logo_image": {"type": "string", "description": "Gallery image name of the user's logo (shown as a watermark and on the end card)"},
        "end_card_lines": {"type": "array", "items": {"type": "string"}, "description": "Exact text for the closing card: tagline, phone, website, address"},
        "shots": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"},
            "keyframe_prompt": {"type": "string", "description": "A single photograph: subject, appearance, setting, lighting, framing. Don't mention things that must NOT appear."},
            "move": {"type": "string", "enum": ["push_in", "pull_out", "pan_left", "pan_right", "rise", "orbit_left", "orbit_right", "drift"],
                     "description": "Image mode camera move. People: push_in or pull_out. Scenery: pan, orbit, rise."},
            "motion_prompt": {"type": "string", "description": "Video mode: what moves and how the camera moves"},
            "seconds": {"type": "number", "description": "2 to 6; image mode fits these to the voice-over"},
            "image": {"type": "string", "description": "Optional: use an existing gallery image instead of generating one"}},
            "required": []}}},
     "required": ["title", "shots"]},
    _tool_start_production))

# ── Voices, audio library and photo motion ────────────────────────────────────
# Kokoro voices and photo motion run in a light CPU lane (they never wait behind
# a long video). Qwen3-TTS uses the GPU, so it goes through the GPU queue.

VOICE_VENV_PY = Path(__file__).parent / "voice_venv" / "Scripts" / "python.exe"
light_queue: "queue.Queue[tuple[str, Callable[[], None]]]" = queue.Queue()


def light_queue_worker():
    while True:
        job_id, fn = light_queue.get()
        try:
            if job_id not in cancelled_jobs:
                fn()
        except Exception as e:
            broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": "error",
                                   "progress": 0.0, "eta": 0, "outputPath": None, "error": str(e)[:400]})
        finally:
            light_queue.task_done()


threading.Thread(target=light_queue_worker, daemon=True).start()


def _status(job_id: str, status: str, pct: float, msg: str = "", **extra):
    output_path, error = extra.pop("outputPath", None), extra.pop("error", None)
    broadcast_from_thread({"type": "job_status", "job_id": job_id, "status": status, "progress": round(pct, 1),
                           "eta": 0, "message": msg, "outputPath": output_path, "error": error, **extra})
    if status in ("done", "error", "cancelled"):
        # Voice and photo-motion jobs can belong to a production too.
        _notify_production(job_id, output_path if status == "done" else None,
                           error or ("Cancelled" if status == "cancelled" else None))


def _run_tool_process(job_id: str, args: list[str], env_extra: Optional[dict] = None) -> dict:
    """Run a helper process (voice or motion), relay its log, return its last JSON line."""
    env = dict(os.environ)
    env.pop("NEURALCUT_TOKEN", None)
    env.update(env_extra or {})
    proc = subprocess.Popen(args, cwd=str(Path(__file__).parent), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    active_processes[job_id] = proc
    result, tail = None, []
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    result = json.loads(line)
                    continue
                except ValueError:
                    pass
            tail = (tail + [line])[-8:]
            publish_log(job_id, line)
        proc.wait()
    finally:
        active_processes.pop(job_id, None)
    if job_id in cancelled_jobs:
        raise RuntimeError("Cancelled")
    if proc.returncode != 0 or result is None:
        raise RuntimeError(" | ".join(tail[-3:])[:400] or f"exit code {proc.returncode}")
    return result


class VoiceRequest(BaseModel):
    model_id: str
    text: str = Field(min_length=1, max_length=5000)
    voice: Optional[str] = Field(default=None, max_length=40)         # Kokoro preset voice id
    instruct: Optional[str] = Field(default=None, max_length=1000)    # Qwen voice description
    language: str = Field(default="English", max_length=30)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: Optional[int] = None
    title: str = Field(default="", max_length=120)


class AnimateRequest(BaseModel):
    move: str = Field(default="push_in", pattern="^(push_in|pull_out|pan_left|pan_right|rise|orbit_left|orbit_right|drift)$")
    seconds: float = Field(default=4.0, ge=1.0, le=20.0)
    strength: float = Field(default=1.0, ge=0.2, le=2.0)
    resolution: str = Field(default="1080p", pattern="^(720p|1080p)$")


def _voice_kind(model_id: str) -> str:
    return MODEL_CONFIG[model_id].get("runtime", {}).get("runner", "")


def run_voice_job(job_id: str, req: VoiceRequest):
    storage.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = storage.AUDIO_DIR / f"voice_{job_id}.wav"
    params_file = Path(tempfile.gettempdir()) / f"pipeline_voice_{job_id}.json"
    runner = _voice_kind(req.model_id)
    started = time.time()
    _status(job_id, "generating", 10, "Speaking" if runner == "kokoro" else "Designing the voice and speaking")
    try:
        if runner == "kokoro":
            params_file.write_text(json.dumps({"text": req.text, "voice": req.voice or "am_michael",
                                               "speed": req.speed, "out": str(out)}), encoding="utf-8")
            res = _run_tool_process(job_id, [sys.executable, "-m", "runners.kokoro_tts", str(params_file)],
                                    {"HF_HUB_OFFLINE": "1"})
        else:
            if not VOICE_VENV_PY.exists():
                raise RuntimeError("The voice designer's environment isn't set up. Reinstall it from Models > Voices.")
            from model_registry import resolve_asset_path
            mdir = resolve_asset_path(MODEL_CONFIG[req.model_id]["runtime"]["model_root"])
            params_file.write_text(json.dumps({"model_dir": str(mdir), "text": req.text,
                                               "instruct": req.instruct or "", "language": req.language,
                                               "out": str(out), "seed": req.seed if req.seed is not None else 7}),
                                   encoding="utf-8")
            res = _run_tool_process(job_id, [str(VOICE_VENV_PY), str(Path(__file__).parent / "runners" / "qwen_tts_runner.py"),
                                             str(params_file)], {"HF_HUB_OFFLINE": "1"})
    finally:
        params_file.unlink(missing_ok=True)
    elapsed = time.time() - started
    out.with_suffix(".json").write_text(json.dumps({
        "job_id": job_id, "model_id": req.model_id, "text": req.text, "voice": req.voice, "instruct": req.instruct,
        "language": req.language, "title": req.title.strip() or req.text.strip()[:60],
        "duration": res.get("duration"), "created_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1)}, indent=2), encoding="utf-8")
    _status(job_id, "done", 100, outputPath=str(out), elapsed_seconds=round(elapsed, 1))


@app.get("/voices")
def voices_catalog():
    """Installed voice models and what each offers (preset voices, or voice design)."""
    out = []
    for mid, m in MODEL_CONFIG.items():
        if _model_kind(m) != "voice":
            continue
        refresh_installed(mid)
        item = {"id": mid, "name": m["identity"]["display_name"], "installed": models_db[mid]["downloaded"],
                "runner": m.get("runtime", {}).get("runner"),
                "voice_design": bool(m.get("capabilities", {}).get("voice_design")), "voices": []}
        if item["installed"] and item["runner"] == "kokoro":
            from runners.kokoro_tts import list_voices
            item["voices"] = list_voices()
        out.append(item)
    return out


@app.post("/voices/speak")
def voices_speak(req: VoiceRequest):
    if req.model_id not in MODEL_CONFIG or _model_kind(MODEL_CONFIG[req.model_id]) != "voice":
        raise HTTPException(404, "Unknown voice model")
    refresh_installed(req.model_id)
    if not models_db[req.model_id]["downloaded"]:
        raise HTTPException(409, "Install this voice model from the Models page first.")
    job_id = uuid.uuid4().hex[:8]
    job_states[job_id] = {"type": "job_status", "job_id": job_id, "status": "queued", "progress": 0.0,
                          "eta": 0, "outputPath": None, "error": None}
    if _voice_kind(req.model_id) == "kokoro":
        light_queue.put((job_id, lambda: run_voice_job(job_id, req)))
    else:
        generation_queue.put((job_id, req))  # uses the GPU: waits its turn with video and image jobs
    return {"job_id": job_id, "status": "queued"}


@app.get("/audio")
def list_audio():
    storage.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for f in sorted(storage.AUDIO_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not storage.AUDIO_NAME_RE.match(f.name):
            continue
        meta = {}
        try:
            meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        items.append({"name": f.name, "bytes": f.stat().st_size, "title": meta.get("title") or f.stem,
                      "text": meta.get("text", ""), "voice": meta.get("voice"), "instruct": meta.get("instruct"),
                      "model_id": meta.get("model_id", ""), "duration": meta.get("duration"),
                      "created_at": meta.get("created_at") or datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds")})
    return items


@app.get("/audio/{name}/file")
def audio_file(name: str):
    try:
        path = storage.audio_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="audio/wav" if name.endswith(".wav") else None)


@app.delete("/audio/{name}")
def delete_audio(name: str):
    try:
        path = storage.audio_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    path.unlink(missing_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)
    return {"status": "success"}


@app.post("/images/{name}/animate")
def animate_image(name: str, req: AnimateRequest):
    """Photo motion: a cinematic camera move through a still, with 3D parallax. Seconds per shot."""
    try:
        src = storage.image_path(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not src.is_file():
        raise HTTPException(404, "That image no longer exists.")
    refresh_installed("depth-anything-v2-small")
    if not models_db.get("depth-anything-v2-small", {}).get("downloaded"):
        raise HTTPException(409, "Install Photo motion from the Models page first.")
    job_id = uuid.uuid4().hex[:8]
    job_states[job_id] = {"type": "job_status", "job_id": job_id, "status": "queued", "progress": 0.0,
                          "eta": 0, "outputPath": None, "error": None}
    w, h = (1920, 1080) if req.resolution == "1080p" else (1280, 720)

    def work():
        out = OUTPUT_DIR / f"video_{job_id}.mp4"
        params_file = Path(tempfile.gettempdir()) / f"pipeline_motion_{job_id}.json"
        params_file.write_text(json.dumps({"image": str(src), "out": str(out), "move": req.move, "seconds": req.seconds,
                                           "strength": req.strength, "width": w, "height": h, "fps": 24}), encoding="utf-8")
        started = time.time()
        _status(job_id, "generating", 15, "Reading depth and moving the camera")
        try:
            _run_tool_process(job_id, [sys.executable, "-m", "runners.photo_motion", str(params_file)])
        finally:
            params_file.unlink(missing_ok=True)
        src_meta = {}
        try:
            src_meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        out.with_suffix(".json").write_text(json.dumps({
            "job_id": job_id, "model_id": "photo-motion", "prompt": src_meta.get("prompt") or f"Photo motion of {name}",
            "created_at": datetime.now().isoformat(timespec="seconds"), "elapsed_seconds": round(time.time() - started, 1),
            "start_image": name, "settings": {"width": w, "height": h, "fps": 24, "num_frames": int(req.seconds * 24),
                                              "move": req.move}}, indent=2), encoding="utf-8")
        _status(job_id, "done", 100, outputPath=str(out), elapsed_seconds=round(time.time() - started, 1))

    light_queue.put((job_id, work))
    return {"job_id": job_id, "status": "queued"}


def _tool_speak(text: str, voice_description: str = "", preset_voice: str = "") -> dict:
    models = _installed("voice")
    designers = [m for m in models if MODEL_CONFIG[m].get("capabilities", {}).get("voice_design")]
    if voice_description and designers:
        req = VoiceRequest(model_id=designers[0], text=text, instruct=voice_description)
    elif models:
        kok = [m for m in models if _voice_kind(m) == "kokoro"]
        req = VoiceRequest(model_id=(kok or models)[0], text=text, voice=preset_voice or "am_michael")
    else:
        return {"error": "No voice model is installed. Install one from Models > Voices."}
    res = voices_speak(req)
    broadcast_from_thread({"type": "job_created", "job_id": res["job_id"], "kind": "voice", "model_id": req.model_id,
                           "prompt": text[:200], "by_agent": True, "summary": "Voice-over"})
    return {"job_id": res["job_id"], "audio_name_when_done": f"voice_{res['job_id']}.wav"}


def _tool_animate_image(image: str, move: str = "push_in", seconds: float = 4.0) -> dict:
    try:
        res = animate_image(image, AnimateRequest(move=move, seconds=seconds))
    except HTTPException as e:
        return {"error": e.detail}
    broadcast_from_thread({"type": "job_created", "job_id": res["job_id"], "kind": "generate", "model_id": "photo-motion",
                           "prompt": f"Photo motion ({move})", "by_agent": True, "summary": f"Photo motion · {seconds:.0f}s"})
    return {"job_id": res["job_id"], "video_name_when_done": f"video_{res['job_id']}.mp4"}


agent.register_tool(agent.Tool(
    "speak", "Make a voice-over audio file. Give voice_description for a natural designed voice (e.g. 'warm, confident "
    "male narrator in his 40s, relaxed, conversational'), or preset_voice for a quick preset. The whole script in one call "
    "keeps the voice consistent.",
    {"type": "object", "properties": {"text": {"type": "string"}, "voice_description": {"type": "string"},
                                      "preset_voice": {"type": "string"}}, "required": ["text"]},
    _tool_speak))
agent.register_tool(agent.Tool(
    "animate_image", "Turn a gallery image into a short video with a cinematic camera move and real 3D parallax "
    "(seconds per shot, full photo sharpness). Best for people and products: use push_in or pull_out for people, "
    "orbit/pan for scenery.",
    {"type": "object", "properties": {
        "image": {"type": "string", "description": "Image gallery file name, e.g. image_ab12cd34.png"},
        "move": {"type": "string", "enum": ["push_in", "pull_out", "pan_left", "pan_right", "rise", "orbit_left", "orbit_right", "drift"]},
        "seconds": {"type": "number"}}, "required": ["image"]},
    _tool_animate_image))


@app.get("/health")
def health():
    # Unauthenticated on purpose (used for readiness polling); reveals nothing sensitive.
    return {"running": True, "version": "1.0.0", "build": SIDECAR_BUILD, "comfyui_ready": True,
            "python_version": sys.version.split()[0]}


@app.get("/models/config")
def get_models_config():
    return {"metadata": VMR_METADATA, "models": MODEL_CONFIG}


def refresh_installed(model_id: Optional[str] = None):
    """Re-check files on disk (cheap: a few stat calls), so models installed or
    removed outside a download in this session are picked up."""
    for mid in ([model_id] if model_id else list(models_db)):
        state = models_db.get(mid)
        if state is not None and mid not in active_downloads:
            installed = check_downloaded(mid, MODELS_DIR)
            if installed != state["downloaded"]:
                state.update(downloaded=installed, progress=100.0 if installed else 0.0)


@app.get("/models")
def get_models():
    refresh_installed()
    return models_db


def _system_ram_gb() -> float:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 1024 ** 3
    except Exception:
        return 0.0


def _nvidia_smi(fields: str) -> Optional[list[str]]:
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0:
            return [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        pass
    return None


@app.get("/gpu/stats")
def gpu_stats():
    p = _nvidia_smi("utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw")
    if not p:
        return {"error": "nvidia-smi unavailable"}
    num = lambda s, f=int: f(s) if s not in ("", "[N/A]", "N/A") else 0
    return {"utilization": num(p[0]), "vram_used_mb": num(p[1]), "vram_total_mb": num(p[2]),
            "temperature": num(p[3]), "power_draw": num(p[4], float)}


@app.get("/system")
def system_info():
    p = _nvidia_smi("name,memory.total,driver_version")
    return {
        "gpu_name": p[0] if p else None,
        "vram_gb": round(int(p[1]) / 1024, 1) if p else 0,
        "driver": p[2] if p else None,
        "ram_gb": round(_system_ram_gb(), 1),
        "disk_free_gb": round(storage.shutil.disk_usage(MODELS_DIR).free / 1024 ** 3, 1),
    }


@app.post("/license/validate")
def validate_license(req: LicenseRequest):
    # Placeholder until a real licence server exists: never claims a paid tier.
    return {"valid": False, "tier": "free", "message": "Licensing isn't available yet.",
            "features": ["Unlimited local generation"]}


# ── Startup ───────────────────────────────────────────────────────────────────

def on_startup():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

    # Windows' proactor loop logs a traceback whenever a client drops a socket
    # (e.g. the UI reloading). That's routine, not an error worth reporting.
    default_handler = main_event_loop.get_exception_handler()

    def quiet_resets(loop, context):
        if isinstance(context.get("exception"), ConnectionResetError):
            return
        if default_handler:
            default_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    main_event_loop.set_exception_handler(quiet_resets)
    print(f"[NeuralCut] Sidecar build: {SIDECAR_BUILD}", flush=True)
    print(f"[NeuralCut] {len(MODEL_CONFIG)} models | models: {MODELS_DIR} | outputs: {OUTPUT_DIR}", flush=True)
    threading.Thread(target=warm_pool.prewarm, daemon=True).start()
    threading.Thread(target=sweep_engine_work, daemon=True).start()
    threading.Thread(target=lambda: production.resume_all(lambda j: j in job_states), daemon=True).start()
    telemetry.start_sampler()
    resume_pending_downloads()


def sweep_engine_work():
    """Remove engine folders left behind by a crash or power cut. A folder an
    engine is still using can't be renamed on Windows, so those are skipped."""
    root = MODELS_DIR.parent / "engine-work"
    if not root.is_dir():
        return
    for d in root.iterdir():
        if not d.is_dir() or d.name.endswith(".trash"):
            continue
        trash = d.with_name(d.name + ".trash")
        try:
            d.rename(trash)
        except OSError:
            continue  # in use
        shutil.rmtree(trash, ignore_errors=True)
    for d in root.glob("*.trash"):
        shutil.rmtree(d, ignore_errors=True)


@contextlib.asynccontextmanager
async def lifespan(_app):
    on_startup()
    yield
    warm_pool.shutdown()


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    port = int(os.environ.get("SIDECAR_PORT", "47821"))
    # Bind loopback only; if the port is taken we exit rather than kill whoever owns it.
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", access_log=False)
