"""Helpers shared by runners: progress reporting, memory hygiene, video export."""
import gc
import os
import json
import time
from pathlib import Path


class Reporter:
    """Emits job_status events on stdout (one JSON object per line), the
    protocol main.py relays to the UI over the WebSocket."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.t0 = time.time()

    def status(self, status: str, progress: float, eta: int = 0, message: str = ""):
        print(json.dumps({
            "type": "job_status", "job_id": self.job_id, "status": status,
            "progress": round(float(progress), 1), "eta": int(max(0, eta)),
            "message": message, "outputPath": None, "error": None,
        }), flush=True)

    def log(self, msg: str):
        print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def vram_gb() -> tuple[float, float]:
    """(free, total) VRAM in GB, or (0, 0) without CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / 1024 ** 3, total / 1024 ** 3
    except Exception:
        pass
    return 0.0, 0.0


def export_mp4(frames, path: Path, fps: int, audio=None, audio_rate: int | None = None):
    """frames: (F, H, W, 3) float in [0,1] or uint8. Writes H.264 MP4 (+AAC audio if given)."""
    import numpy as np
    import imageio

    arr = np.asarray(frames)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    with imageio.get_writer(str(tmp), fps=fps, codec="libx264", quality=8,
                            macro_block_size=16, ffmpeg_log_level="error") as w:
        for f in arr:
            w.append_data(f)
    tmp.replace(path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Video export produced an empty file")
    return path


# ── Prompt embedding cache ───────────────────────────────────────────────────
# Re-rolling a prompt with a new seed is the most common action, and encoding
# with a 5-11GB text encoder costs ~30s. Cache the result, capped in size.

EMBED_CACHE_LIMIT_BYTES = 512 * 1024 ** 2


def _embed_cache_dir() -> Path:
    from model_registry import MODELS_DIR
    d = MODELS_DIR.parent / "cache" / "embeds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _embed_key(namespace: str, *parts: str) -> str:
    import hashlib
    h = hashlib.sha256(namespace.encode())
    for p in parts:
        h.update(b"\0" + p.encode("utf-8"))
    return h.hexdigest()[:32]


def load_cached_embeds(namespace: str, *parts: str):
    """Tensors saved by save_cached_embeds, moved to CUDA, or None."""
    path = _embed_cache_dir() / f"{_embed_key(namespace, *parts)}.safetensors"
    if not path.exists():
        return None
    try:
        from safetensors.torch import load_file
        tensors = load_file(str(path), device="cuda")
        os.utime(path)  # mark recently used
        return tensors
    except Exception:
        path.unlink(missing_ok=True)
        return None


def save_cached_embeds(namespace: str, tensors: dict, *parts: str) -> None:
    try:
        from safetensors.torch import save_file
        d = _embed_cache_dir()
        save_file({k: v.detach().contiguous().cpu() for k, v in tensors.items()},
                  str(d / f"{_embed_key(namespace, *parts)}.safetensors"))
        files = sorted(d.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
        total = 0
        for f in files:
            total += f.stat().st_size
            if total > EMBED_CACHE_LIMIT_BYTES:
                f.unlink(missing_ok=True)
    except Exception as e:
        print(f"[cache] could not save embeddings: {e}", flush=True)
