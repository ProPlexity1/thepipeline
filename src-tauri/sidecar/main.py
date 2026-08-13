import sys
# Force UTF-8 on stdout/stderr so Windows CP1252 never crashes on Unicode worker output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
import subprocess
import asyncio
import uuid
import json
import threading
import queue
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from model_registry import (
    MODEL_CONFIG, SHARED_RESOURCES, VMR_METADATA,
    MODELS_DIR, OUTPUT_DIR,
    get_model, clamp_settings, resolve_generation_params, resolve_offload_strategy, 
    get_download_manifest, check_downloaded, validate_runtime_assets
)
import uvicorn
import requests
import logging

class SuppressNoisyEndpoints(logging.Filter):
    NOISY_PATHS = ("/gpu/stats", "/gpu ", "GET /models ")
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in self.NOISY_PATHS)

logging.getLogger("uvicorn.access").addFilter(SuppressNoisyEndpoints())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
import tempfile

SIDECAR_BUILD = "ltx-shared-095-queue-2026-07-29"
SIDECAR_SCRIPT = str(Path(__file__).resolve())

# ── Global State ──────────────────────────────────────────────────────────────

def init_models_db():
    """Initialize download tracking for all models in VMR."""
    db = {}
    for model_id in MODEL_CONFIG:
        downloaded = check_downloaded(model_id, MODELS_DIR)
        db[model_id] = {
            "downloaded": downloaded,
            "downloading": False,
            "progress": 100.0 if downloaded else 0.0,
            "speed_mbps": 0.0,
            "eta_seconds": 0,
        }
    return db

models_db = init_models_db()

active_connections = []
cancel_flags: dict[str, bool] = {}
main_event_loop = None

# ── Pydantic Models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    model_id: str = "ltx-video-standard"
    profile: str = "balanced"
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    quantization: Optional[str] = None

class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    running: bool
    version: str
    comfyui_ready: bool
    python_version: str

class LicenseRequest(BaseModel):
    key: str

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        await websocket.send_json({"type": "connection_established"})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)

async def broadcast_job_status(job_id: str, status: str, progress: float, eta: int, output_path: str = None, error: str = None):
    payload = {
        "type": "job_status",
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "eta": eta,
        "outputPath": output_path,
        "error": error,
    }
    for conn in list(active_connections):
        try:
            await conn.send_json(payload)
        except Exception:
            if conn in active_connections:
                active_connections.remove(conn)

def broadcast_from_thread(payload: dict):
    if main_event_loop is None or not active_connections:
        return

    async def _send_all():
        for conn in list(active_connections):
            try:
                await conn.send_json(payload)
            except Exception:
                if conn in active_connections:
                    active_connections.remove(conn)

    try:
        asyncio.run_coroutine_threadsafe(_send_all(), main_event_loop)
    except Exception as e:
        print(f"[NeuralCut] Broadcast error: {e}", flush=True)

# ── Model Download ────────────────────────────────────────────────────────────

def real_model_download(model_id: str, loop=None):
    try:
        import time
        from huggingface_hub import hf_hub_url

        cancel_flags[model_id] = False

        # Get manifest: all files this model needs (including shared_resources if any)
        manifest = get_download_manifest(model_id)

        # Check what is already completed vs. what needs downloading
        queue = []
        completed_bytes = 0
        total_model_bytes = 0

        for entry in manifest:
            if cancel_flags.get(model_id):
                return
            
            local_path = MODELS_DIR / entry["local_path"]
            if local_path.exists():
                size = local_path.stat().st_size
                completed_bytes += size
                total_model_bytes += size
            else:
                queue.append(entry)
                # Fetch remote size with allow_redirects=True
                url = hf_hub_url(
                    entry["repo_id"],
                    entry["filename"],
                    revision=entry.get("revision", "main"),
                )
                try:
                    r = requests.head(url, allow_redirects=True, timeout=10)
                    size = int(r.headers.get("content-length", 0))
                except Exception:
                    size = 0
                total_model_bytes += size

        # Fallback if total_model_bytes is 0
        if total_model_bytes == 0:
            model = get_model(model_id)
            total_model_bytes = int(model["distribution"]["estimated_download_size_gb"] * 1024 * 1024 * 1024)

        if not queue:
            # Already completed
            models_db[model_id]["downloading"] = False
            models_db[model_id]["downloaded"] = True
            models_db[model_id]["progress"] = 100.0
            models_db[model_id]["speed_mbps"] = 0.0
            models_db[model_id]["eta_seconds"] = 0
            broadcast_from_thread({
                "type": "download_progress",
                "model_id": model_id,
                "progress": 100.0,
                "speed_mbps": 0.0,
                "eta_seconds": 0,
                "downloading": False,
                "downloaded": True,
            })
            return

        # Get initial downloaded bytes of temp files in queue
        initial_temp_bytes = 0
        for entry in queue:
            local_path = MODELS_DIR / entry["local_path"]
            temp_path = Path(str(local_path) + ".tmp")
            if temp_path.exists():
                initial_temp_bytes += temp_path.stat().st_size

        downloaded_so_far = completed_bytes + initial_temp_bytes
        last_time = time.time()
        last_bytes = downloaded_so_far

        for file_idx, entry in enumerate(queue):
            if cancel_flags.get(model_id):
                models_db[model_id]["downloading"] = False
                models_db[model_id]["progress"] = 0.0
                return

            repo_id = entry["repo_id"]
            filename = entry["filename"]
            local_path = MODELS_DIR / entry["local_path"]
            
            local_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = Path(str(local_path) + ".tmp")
            file_initial_size = temp_path.stat().st_size if temp_path.exists() else 0

            url = hf_hub_url(repo_id, filename, revision=entry.get("revision", "main"))
            hf_token = os.environ.get("HF_TOKEN", "")
            headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
            if file_initial_size > 0:
                headers["Range"] = f"bytes={file_initial_size}-"

            try:
                response = requests.get(url, headers=headers, stream=True, timeout=60)
                # If range request isn't supported or returns 416, delete temp and start fresh
                if response.status_code == 416 or (file_initial_size > 0 and response.status_code != 206):
                    if temp_path.exists():
                        temp_path.unlink()
                    file_initial_size = 0
                    if "Range" in headers:
                        del headers["Range"]
                    response = requests.get(url, headers=headers, stream=True, timeout=60)
                
                response.raise_for_status()

                file_downloaded = file_initial_size
                with open(temp_path, "ab" if file_initial_size > 0 else "wb") as f:
                    for chunk in response.iter_content(chunk_size=512 * 1024):
                        if cancel_flags.get(model_id):
                            models_db[model_id]["downloading"] = False
                            models_db[model_id]["progress"] = 0.0
                            broadcast_from_thread({
                                "type": "download_progress",
                                "model_id": model_id,
                                "progress": 0.0,
                                "speed_mbps": 0.0,
                                "eta_seconds": 0,
                                "downloading": False,
                                "downloaded": False,
                            })
                            return

                        if chunk:
                            f.write(chunk)
                            file_downloaded += len(chunk)
                            downloaded_so_far += len(chunk)

                            now = time.time()
                            elapsed = now - last_time
                            if elapsed >= 0.5:
                                bytes_delta = downloaded_so_far - last_bytes
                                speed_bps = bytes_delta / elapsed if elapsed > 0 else 0
                                speed_mbps = speed_bps / (1024 * 1024)
                                
                                overall = min((downloaded_so_far / total_model_bytes) * 100.0, 99.9)
                                remaining_bytes = max(total_model_bytes - downloaded_so_far, 0)
                                eta = int(remaining_bytes / speed_bps) if speed_bps > 0 else 0

                                models_db[model_id]["progress"] = round(overall, 1)
                                models_db[model_id]["speed_mbps"] = round(speed_mbps, 1)
                                models_db[model_id]["eta_seconds"] = eta

                                broadcast_from_thread({
                                    "type": "download_progress",
                                    "model_id": model_id,
                                    "progress": round(overall, 1),
                                    "speed_mbps": round(speed_mbps, 1),
                                    "eta_seconds": eta,
                                    "downloading": True,
                                    "downloaded": False,
                                })

                                last_time = now
                                last_bytes = downloaded_so_far

                temp_path.rename(local_path)
                print(f"[NeuralCut] Downloaded file: {filename}", flush=True)

            except Exception as e:
                print(f"[NeuralCut] Download error on file {filename}: {e}", flush=True)
                models_db[model_id]["downloading"] = False
                models_db[model_id]["progress"] = 0.0
                broadcast_from_thread({
                    "type": "download_progress",
                    "model_id": model_id,
                    "progress": 0.0,
                    "speed_mbps": 0.0,
                    "eta_seconds": 0,
                    "downloading": False,
                    "downloaded": False,
                })
                return

        validated, validation_msg = validate_runtime_assets(model_id, MODELS_DIR)
        if not validated:
            raise RuntimeError(validation_msg)

        # Complete!
        models_db[model_id]["downloading"] = False
        models_db[model_id]["downloaded"] = True
        models_db[model_id]["progress"] = 100.0
        models_db[model_id]["speed_mbps"] = 0.0
        models_db[model_id]["eta_seconds"] = 0
        print(f"[NeuralCut] Model {model_id} fully downloaded", flush=True)

        broadcast_from_thread({
            "type": "download_progress",
            "model_id": model_id,
            "progress": 100.0,
            "speed_mbps": 0.0,
            "eta_seconds": 0,
            "downloading": False,
            "downloaded": True,
        })

    except Exception as e:
        print(f"[NeuralCut] Download failed: {e}", flush=True)
        models_db[model_id]["downloading"] = False
        models_db[model_id]["progress"] = 0.0
        models_db[model_id]["downloaded"] = False
        broadcast_from_thread({
            "type": "download_progress",
            "model_id": model_id,
            "progress": 0.0,
            "speed_mbps": 0.0,
            "eta_seconds": 0,
            "downloading": False,
            "downloaded": False,
            "error": str(e),
        })

# ── Generation ──────────────────────────────
active_processes: dict[str, subprocess.Popen] = {}
generation_queue: "queue.Queue[tuple[str, GenerateRequest]]" = queue.Queue()
generation_queue_worker_started = False

def generation_queue_worker():
    while True:
        job_id, req = generation_queue.get()
        try:
            run_generation_subprocess(job_id, req)
        finally:
            generation_queue.task_done()

def ensure_generation_queue_worker():
    global generation_queue_worker_started
    if generation_queue_worker_started:
        return
    generation_queue_worker_started = True
    thread = threading.Thread(target=generation_queue_worker, daemon=True)
    thread.start()

def run_generation_subprocess(job_id: str, req: GenerateRequest):
    params = {
        "job_id": job_id,
        "model_id": req.model_id,
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "profile": req.profile or "balanced",
        "steps": req.steps,
        "cfg_scale": req.cfg_scale,
        "width": req.width,
        "height": req.height,
        "num_frames": req.num_frames,
        "fps": req.fps,
        "seed": req.seed,
        "scheduler": req.scheduler,
        "quantization": req.quantization,
    }

    params_file = Path(tempfile.gettempdir()) / f"neuralcut_job_{job_id}.json"
    with open(params_file, "w") as f:
        json.dump(params, f)

    worker_script = Path(__file__).parent / "generate_worker.py"

    proc = subprocess.Popen(
        [sys.executable, str(worker_script), str(params_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=dict(os.environ),
    )
    active_processes[job_id] = proc

    stderr_lines = []

    def drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)
            print(f"[worker:{job_id}] {line}", end="", flush=True)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    saw_terminal_status = False
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[worker:{job_id}] {line}", flush=True)
                continue
            if payload.get("status") in ("done", "error"):
                saw_terminal_status = True
            broadcast_from_thread(payload)
    finally:
        proc.wait()
        stderr_thread.join(timeout=2)
        active_processes.pop(job_id, None)
        params_file.unlink(missing_ok=True)

    # Worker process died without ever sending a done/error message —
    # this is the "silent crash" case (OOM kill, CUDA/driver abort, segfault).
    if not saw_terminal_status:
        tail = "".join(stderr_lines[-20:]).strip()
        msg = f"Generation process terminated unexpectedly (exit code {proc.returncode})."
        if proc.returncode is not None and proc.returncode < 0:
            msg += " This usually means the OS killed it for using too much memory."
        if tail:
            msg += f" Last output: {tail[-500:]}"
        broadcast_from_thread({
            "type": "job_status", "job_id": job_id, "status": "error",
            "progress": 0.0, "eta": 0, "outputPath": None, "error": msg,
        })
        print(f"[NeuralCut] Job {job_id} crashed silently, exit code {proc.returncode}", flush=True)
# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/models/config")
def get_models_config():
    """Serve the full VMR to the frontend. Frontend uses this to build UI,
    sliders, model chooser, etc."""
    return {
        "metadata": VMR_METADATA,
        "models": MODEL_CONFIG,
    }

@app.get("/health", response_model=StatusResponse)
def health():
    return {
        "running": True,
        "version": "1.0.0",
        "build": SIDECAR_BUILD,
        "script": SIDECAR_SCRIPT,
        "comfyui_ready": True,
        "python_version": sys.version,
    }

@app.get("/gpu/stats")
def gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi", 
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "utilization": int(parts[0]) if len(parts) > 0 else 0,
                "vram_used_mb": int(parts[1]) if len(parts) > 1 else 0,
                "vram_total_mb": int(parts[2]) if len(parts) > 2 else 0,
                "temperature": int(parts[3]) if len(parts) > 3 else 0,
                "power_draw": float(parts[4]) if len(parts) > 4 else 0,
            }
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/gpu")
def gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "detected": True,
                "name": parts[0] if len(parts) > 0 else "Unknown",
                "vram_mb": int(parts[1]) if len(parts) > 1 else 0,
                "vram_gb": round(int(parts[1]) / 1024) if len(parts) > 1 else 0,
                "driver": parts[2] if len(parts) > 2 else "Unknown",
                "temperature": int(parts[3]) if len(parts) > 3 else 0,
            }
    except Exception as e:
        return {"detected": False, "error": str(e)}

@app.get("/models")
def get_models():
    return models_db

@app.post("/models/download/{model_id}")
def start_download(model_id: str):
    if model_id not in models_db:
        return {"status": "error", "message": "Model not found"}

    model = models_db[model_id]
    if model["downloaded"]:
        return {"status": "success", "message": "Model already downloaded"}
    if model["downloading"]:
        return {"status": "success", "message": "Already downloading"}

    models_db[model_id]["downloading"] = True
    models_db[model_id]["progress"] = 0.0

    thread = threading.Thread(
        target=real_model_download,
        args=(model_id, None),
        daemon=True,
    )
    thread.start()

    return {"status": "success", "message": f"Started download for {model_id}"}

@app.get("/models/download/{model_id}/progress")
async def download_progress(model_id: str):
    if model_id not in models_db:
        return {"status": "error", "message": "Model not found"}

    async def event_generator():
        while True:
            model = models_db[model_id]
            data = {
                "progress": model["progress"],
                "downloading": model["downloading"],
                "downloaded": model["downloaded"],
                "speed_mbps": model.get("speed_mbps", 0.0),
                "eta_seconds": model.get("eta_seconds", 0),
            }
            yield f"data: {json.dumps(data)}\n\n"
            if not model["downloading"] or model["downloaded"]:
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    if model_id not in models_db:
        return {"status": "error", "message": "Model not found"}

    cancel_flags[model_id] = True

    model_dir = MODELS_DIR / model_id
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)

    models_db[model_id] = {
        "downloaded": False,
        "downloading": False,
        "progress": 0.0,
        "speed_mbps": 0.0,
        "eta_seconds": 0,
    }
    return {"status": "success", "message": f"Model {model_id} deleted"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Check model exists and is downloaded
    if req.model_id not in MODEL_CONFIG:
        return GenerateResponse(job_id="", status="error", message=f"Unknown model: {req.model_id}")
    
    model = models_db.get(req.model_id, {})
    if not model.get("downloaded"):
        return GenerateResponse(job_id="", status="error", message=f"Model {req.model_id} is not downloaded yet")

    # Resolve and clamp generation parameters taking profile & overrides into account
    req_dict = req.model_dump()
    resolved = resolve_generation_params(req.model_id, req_dict)

    # Update request with resolved values
    req.steps = resolved.get("steps")
    req.cfg_scale = resolved.get("cfg_scale")
    req.width = resolved.get("width")
    req.height = resolved.get("height")
    req.num_frames = resolved.get("num_frames")
    req.fps = resolved.get("fps")
    req.seed = resolved.get("seed")
    req.scheduler = resolved.get("scheduler")
    req.quantization = resolved.get("quantization")

    job_id = str(uuid.uuid4())[:8]
    ensure_generation_queue_worker()
    generation_queue.put((job_id, req))
    return GenerateResponse(job_id=job_id, status="queued", message=f"Job {job_id} queued")

@app.post("/generate/{job_id}/cancel")
def cancel_generation(job_id: str):
    proc = active_processes.get(job_id)
    if not proc:
        return {"status": "error", "message": "Job not found or already finished"}
    proc.terminate()
    return {"status": "success", "message": f"Cancel requested for {job_id}"}

@app.post("/license/validate")
def validate_license(req: LicenseRequest):
    if not req.key.strip():
        return {
            "valid": False,
            "tier": "free",
            "message": "License key cannot be empty",
            "features": ["Basic generation", "512px max", "Watermark"],
        }
    # TODO: replace with real Lemon Squeezy API call
    return {
        "valid": True,
        "tier": "pro",
        "expires_at": "2027-01-01",
        "features": ["Unlimited generation", "HD output", "No watermark", "Priority support", "All model tiers"],
    }

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
     # Set HF token for faster downloads if available
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        print(f"[NeuralCut] HuggingFace authenticated", flush=True)
    else:
        print(f"[NeuralCut] No HF_TOKEN set — downloads may be slower", flush=True)

    print(f"[NeuralCut] Event loop captured", flush=True)
    print(f"[NeuralCut] Sidecar build: {SIDECAR_BUILD}", flush=True)
    print(f"[NeuralCut] Sidecar script: {SIDECAR_SCRIPT}", flush=True)
    print(f"[NeuralCut] VMR loaded: {len(MODEL_CONFIG)} models", flush=True)
    print(f"[NeuralCut] Models dir: {MODELS_DIR}", flush=True)
    print(f"[NeuralCut] Output dir: {OUTPUT_DIR}", flush=True)

# ── Port conflict resolution ──────────────────────────────────────────────────

def kill_port_holder(port: int) -> bool:
    """Kill any process currently holding the given port (Windows-specific)."""
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            # Match lines like:  TCP    127.0.0.1:8188    ...    LISTENING    12345
            if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid == os.getpid():
                    continue  # Don't kill ourselves
                print(f"[NeuralCut] Killing old process on port {port} (PID {pid})", flush=True)
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, timeout=5)
                import time
                time.sleep(0.5)  # Give OS time to release the socket
                return True
    except Exception as e:
        print(f"[NeuralCut] Could not check/kill port holder: {e}", flush=True)
    return False


def is_port_free(port: int) -> bool:
    """Check if a port is available for binding."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_port = int(os.environ.get("SIDECAR_PORT", 8188))
    fallback_ports = [base_port, base_port + 1, base_port + 2]

    chosen_port = None
    for port in fallback_ports:
        if is_port_free(port):
            chosen_port = port
            break
        # Port is occupied — try to kill the holder
        print(f"[NeuralCut Sidecar] Port {port} is busy, attempting to free it...", flush=True)
        kill_port_holder(port)
        if is_port_free(port):
            chosen_port = port
            break

    if chosen_port is None:
        print(f"[NeuralCut Sidecar] FATAL: Could not bind to any port in {fallback_ports}", flush=True)
        sys.exit(1)

    print(f"[NeuralCut Sidecar] Starting on port {chosen_port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=chosen_port, log_level="info", access_log=False)
