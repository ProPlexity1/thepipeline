"""Live telemetry for the UI: system usage history, per-job log lines, and
generation-time estimates learned from this machine's own past runs."""
import json
import re
import statistics
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

# ── System usage sampler ─────────────────────────────────────────────────────
# One background thread samples once a second and keeps the last minute, so a
# graph has history the moment it is shown (like Task Manager).

HISTORY_SECONDS = 60
_history: deque = deque(maxlen=HISTORY_SECONDS)
_sampler_started = False


def _gpu_sample():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if r.returncode == 0:
            p = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            num = lambda s: float(s) if s not in ("", "[N/A]", "N/A") else 0.0
            return {"gpu": num(p[0]), "vram_used_gb": num(p[1]) / 1024, "vram_total_gb": num(p[2]) / 1024,
                    "gpu_temp": num(p[3])}
    except Exception:
        pass
    return {"gpu": 0.0, "vram_used_gb": 0.0, "vram_total_gb": 0.0, "gpu_temp": 0.0}


def _sample_loop():
    import psutil
    psutil.cpu_percent(interval=None)  # prime the counter
    while True:
        t0 = time.time()
        mem = psutil.virtual_memory()
        _history.append({
            "t": round(t0, 1),
            "cpu": psutil.cpu_percent(interval=None),
            "ram_used_gb": (mem.total - mem.available) / 1024 ** 3,
            "ram_total_gb": mem.total / 1024 ** 3,
            **_gpu_sample(),
        })
        time.sleep(max(0.0, 1.0 - (time.time() - t0)))


_hw_cache: tuple[float, float] | None = None


def hardware_gb() -> tuple[float, float]:
    """(total VRAM, total RAM) in GB, for picking models that fit this PC."""
    global _hw_cache
    if _hw_cache is None:
        import psutil
        _hw_cache = (_gpu_sample()["vram_total_gb"], psutil.virtual_memory().total / 1024 ** 3)
    return _hw_cache


def start_sampler():
    global _sampler_started
    if not _sampler_started:
        _sampler_started = True
        threading.Thread(target=_sample_loop, daemon=True).start()


def stats_history() -> list[dict]:
    return list(_history)


# ── Job log lines ────────────────────────────────────────────────────────────

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_NOISE = (
    "triton not found", "flop_counter", "expandable_segments", "should be kept in float32",
    "UserWarning", "warnings.warn", "DeprecationWarning", "FutureWarning",
    "prompt_attention_mask = prompt_attention_mask", "Casting directly with `to()`",
)
_job_logs: dict[str, deque] = {}


def clean_line(raw: str) -> str | None:
    """Human-readable version of a worker output line, or None to drop it."""
    line = _ANSI.sub("", raw).rstrip()
    if "\r" in line:  # progress bars redraw with carriage returns: keep the latest state
        line = line.split("\r")[-1]
    line = line.strip()
    if not line or any(n in line for n in _NOISE):
        return None
    if line.startswith("[REGISTRY]") or line.startswith("W1") or line.startswith("WORKER_"):
        return None
    if line.startswith("{") and line.endswith("}"):
        return None  # structured events are shown as progress, not text
    line = re.sub(r"^\[\s*[0-9.]+s\]\s*", "", line)  # runner timestamps; the UI adds its own clock
    return line[:300]


def add_log(job_id: str, line: str) -> str | None:
    """Store a cleaned line. Returns "append", "replace" (a progress bar
    updating in place) or None (exact repeat, nothing to show)."""
    buf = _job_logs.setdefault(job_id, deque(maxlen=400))
    # Collapse progress-bar spam ("Loading weights: 37%|...") into one updating line.
    key = line.split(":")[0] if "%|" in line else None
    if buf and key and buf[-1].split(":")[0] == key and "%|" in buf[-1]:
        buf[-1] = line
        return "replace"
    if buf and buf[-1] == line:
        return None
    buf.append(line)
    while len(_job_logs) > 50:
        _job_logs.pop(next(iter(_job_logs)))
    return "append"


def job_logs(job_id: str) -> list[str]:
    return list(_job_logs.get(job_id, []))


# ── Timing history and estimates ─────────────────────────────────────────────

_timings_lock = threading.Lock()


def _timings_path() -> Path:
    from model_registry import MODELS_DIR
    return MODELS_DIR.parent / "timings.json"


def _load_timings() -> dict:
    p = _timings_path()
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except ValueError:
        return {}


def record_timing(model_id: str, settings: dict, seconds: float) -> None:
    entry = {k: settings.get(k) for k in ("width", "height", "num_frames", "steps")}
    entry.update(seconds=round(seconds, 1), at=int(time.time()))
    with _timings_lock:
        data = _load_timings()
        runs = data.setdefault(model_id, [])
        runs.append(entry)
        data[model_id] = runs[-100:]
        try:
            _timings_path().write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass


def _cost(s: dict) -> float:
    return max(1.0, (s.get("width") or 1) * (s.get("height") or 1) * (s.get("num_frames") or 1) * (s.get("steps") or 1))


def estimate_seconds(model_id: str, settings: dict) -> dict | None:
    """Expected wall time on this PC, from past runs of the same model.
    Same settings: median of recent runs. Otherwise: nearest-cost run scaled by
    the work ratio (sub-linear, since loading and decoding don't scale fully)."""
    runs = _load_timings().get(model_id) or []
    if not runs:
        return None
    key = lambda r: (r.get("width"), r.get("height"), r.get("num_frames"), r.get("steps"))
    want = key(settings)
    same = [r["seconds"] for r in runs if key(r) == want]
    if same:
        return {"seconds": round(statistics.median(same[-5:])), "basis": "same settings", "samples": len(same)}
    c = _cost(settings)
    nearest = min(runs[-30:], key=lambda r: abs(_cost(r) / c - 1))
    ratio = c / _cost(nearest)
    return {"seconds": round(nearest["seconds"] * ratio ** 0.85), "basis": "similar runs", "samples": 1}
