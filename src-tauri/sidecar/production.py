"""Productions: multi-shot projects the agent plans and the engine executes.

The agent writes a plan and hands it over with one tool call, then sleeps. The
engine works alone, saving state after every step (resumes after a restart):

mode "image" (default: fast, sharp, consistent):
  keyframe image per shot + one voice-over  ->  shot lengths fitted to the voice
  ->  photo-motion camera move per shot  ->  assemble with voice, logo watermark
  and an exact end card.
mode "video" (real motion, slow):
  keyframe image per shot  ->  video model animates each keyframe  ->  assemble.

When done (or failed), a message is posted in the chat that started it.
"""
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from model_registry import MODELS_DIR, OUTPUT_DIR

PROD_DIR = MODELS_DIR.parent / "productions"
_lock = threading.RLock()


class Hooks:
    """Set by main.py."""
    queue_image: Callable[[str, dict], str]                  # (prompt, opts) -> job_id
    queue_video: Callable[[str, Optional[str], dict], str]   # (prompt, start_image, opts) -> job_id
    queue_voice: Callable[[str, str], str]                   # (script, voice_description) -> job_id
    queue_motion: Callable[[str, str, float], str]           # (image, move, seconds) -> job_id
    audio_duration: Callable[[str], float]                   # audio file name -> seconds
    assemble: Callable[[dict], str]                          # production -> output file name
    notify: Callable[[str, str], None]                       # (conv_id, text)
    broadcast: Callable[[dict], None]


hooks = Hooks()


def _path(pid: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{10}", pid):
        raise ValueError("Invalid production id")
    return PROD_DIR / f"{pid}.json"


def save(prod: dict):
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    prod["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = _path(prod["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(prod, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_path(prod["id"]))
    try:
        hooks.broadcast({"type": "production", "production": summary(prod)})
    except Exception:
        pass


def load(pid: str) -> dict:
    return json.loads(_path(pid).read_text(encoding="utf-8"))


def summary(prod: dict) -> dict:
    shots = prod["shots"]
    return {
        "id": prod["id"], "title": prod["title"], "status": prod["status"], "mode": prod.get("mode", "video"),
        "created_at": prod["created_at"], "conv_id": prod.get("conv_id"),
        "shots_total": len(shots), "shots_done": sum(1 for s in shots if s.get("video")),
        "shots_failed": sum(1 for s in shots if s.get("error")), "final": prod.get("final"),
        "error": prod.get("error"), "voice": (prod.get("voiceover") or {}).get("audio"),
        "shots": [{k: s.get(k) for k in ("index", "title", "status", "image", "video", "error")} for s in shots],
    }


def list_all() -> list[dict]:
    out = []
    for p in sorted(PROD_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            out.append(summary(json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return out


def create(title: str, style: str, shots: list[dict], options: dict, mode: str = "image",
           voiceover: Optional[dict] = None, end_card: Optional[dict] = None, conv_id: Optional[str] = None) -> dict:
    prod = {
        "id": uuid.uuid4().hex[:10], "title": title.strip()[:120] or "Untitled", "style": style.strip(),
        "mode": mode if mode in ("image", "video") else "image",
        "status": "running", "created_at": datetime.now().isoformat(timespec="seconds"),
        "options": options, "final": None, "error": None, "conv_id": conv_id,
        "voiceover": ({"script": voiceover.get("script", "").strip(), "voice": voiceover.get("voice_description", "").strip(),
                       "job": None, "audio": None, "error": None} if voiceover and voiceover.get("script") else None),
        "end_card": end_card or None,
        "shots": [{
            "index": i + 1, "title": (s.get("title") or f"Shot {i + 1}")[:80],
            "keyframe_prompt": s.get("keyframe_prompt", ""), "motion_prompt": s.get("motion_prompt", ""),
            "move": s.get("move") or "push_in", "seconds": float(s.get("seconds") or 4), "status": "pending",
            "image": s.get("image"), "image_job": None, "video": None, "video_job": None, "error": None,
        } for i, s in enumerate(shots)],
    }
    save(prod)
    advance(prod["id"])
    return prod


def _finish(prod: dict, ok: bool):
    if not prod.get("conv_id"):
        return
    if ok:
        msg = (f"**{prod['title']}** is ready. Watch it on the **Video** page (file `{prod['final']}`), "
               "or open it in the **Editor** to tweak cuts, titles or the voice-over.")
    else:
        msg = f"I couldn't finish **{prod['title']}**: {prod.get('error') or 'unknown error'}. Tell me how you'd like to proceed."
    try:
        hooks.notify(prod["conv_id"], msg)
    except Exception:
        pass


def advance(pid: str):
    """Queue whatever can start now. Safe to call repeatedly."""
    with _lock:
        prod = load(pid)
        if prod["status"] != "running":
            return
        style = prod.get("style", "")
        shots = prod["shots"]
        try:
            # 1. Keyframes (both modes) and the voice-over (image mode).
            for s in shots:
                if not s.get("error") and not s.get("image") and not s.get("image_job"):
                    s["image_job"] = hooks.queue_image(f"{s['keyframe_prompt']} {style}".strip(), prod["options"])
                    s["status"] = "keyframe"
            vo = prod.get("voiceover")
            if vo and not vo.get("audio") and not vo.get("job") and not vo.get("error"):
                vo["job"] = hooks.queue_voice(vo["script"], vo["voice"])

            live = [s for s in shots if not s.get("error")]
            keyframes_done = all(s.get("image") for s in live)
            voice_done = not vo or vo.get("audio") or vo.get("error")

            # 2. Motion for each shot.
            if prod["mode"] == "video":
                for s in live:
                    if s.get("image") and not s.get("video") and not s.get("video_job"):
                        s["video_job"] = hooks.queue_video(f"{s['motion_prompt']} {style}".strip(), s["image"],
                                                           {**prod["options"], "seconds": s["seconds"]})
                        s["status"] = "animating"
            elif keyframes_done and voice_done and live:
                # Fit shot lengths to the voice-over so pictures and words end together.
                if vo and vo.get("audio") and not prod.get("timed"):
                    total = hooks.audio_duration(vo["audio"]) + 1.2 + 0.4 * (len(live) - 1)
                    weight = sum(s["seconds"] for s in live)
                    for s in live:
                        s["seconds"] = round(max(2.2, total * s["seconds"] / weight), 2)
                    prod["timed"] = True
                for s in live:
                    if not s.get("video") and not s.get("video_job"):
                        s["video_job"] = hooks.queue_motion(s["image"], s.get("move") or "push_in", s["seconds"])
                        s["status"] = "animating"

            # 3. Assemble.
            live = [s for s in shots if not s.get("error")]
            if live and all(s.get("video") for s in live) and voice_done and not prod.get("final"):
                prod["status"] = "assembling"
                save(prod)
                prod["final"] = hooks.assemble(prod)
                prod["status"] = "done"
                save(prod)
                _finish(prod, True)
                return
            if not live:
                prod["status"], prod["error"] = "failed", "Every shot failed."
                save(prod)
                _finish(prod, False)
                return
        except Exception as e:
            prod["status"], prod["error"] = "failed", str(e)[:400]
            save(prod)
            _finish(prod, False)
            return
        save(prod)


def on_job_finished(job_id: str, output_path: Optional[str], error: Optional[str]):
    """Called by the engine for every finished job; updates whichever production owns it."""
    for p in PROD_DIR.glob("*.json"):
        try:
            prod = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hit = False
        vo = prod.get("voiceover")
        if vo and vo.get("job") == job_id:
            hit = True
            if output_path:
                vo["audio"] = Path(output_path).name
            else:
                vo["error"] = f"Voice-over failed: {error or 'unknown error'}"[:300]  # carry on without it
        for s in prod.get("shots", []):
            if s.get("image_job") == job_id:
                hit = True
                if output_path:
                    s["image"], s["status"] = Path(output_path).name, "keyframe_ready"
                else:
                    s["error"], s["status"] = f"Keyframe failed: {error or 'unknown error'}"[:300], "failed"
            elif s.get("video_job") == job_id:
                hit = True
                if output_path:
                    s["video"], s["status"] = Path(output_path).name, "done"
                else:
                    s["error"], s["status"] = f"Shot failed: {error or 'unknown error'}"[:300], "failed"
        if hit:
            with _lock:
                save(prod)
            advance(prod["id"])
            return


def resume_all(job_exists: Callable[[str], bool]):
    """After a restart: jobs that were queued are gone, so clear them and re-queue."""
    for p in PROD_DIR.glob("*.json"):
        try:
            prod = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if prod.get("status") not in ("running", "assembling"):
            continue
        prod["status"] = "running"
        for s in prod["shots"]:
            if s.get("image_job") and not s.get("image") and not job_exists(s["image_job"]):
                done = OUTPUT_DIR / "images" / f"image_{s['image_job']}.png"
                s["image"] = done.name if done.exists() else None
                s["image_job"] = s["image_job"] if s["image"] else None
            if s.get("video_job") and not s.get("video") and not job_exists(s["video_job"]):
                done = OUTPUT_DIR / f"video_{s['video_job']}.mp4"
                s["video"] = done.name if done.exists() else None
                s["video_job"] = s["video_job"] if s["video"] else None
        vo = prod.get("voiceover")
        if vo and vo.get("job") and not vo.get("audio") and not job_exists(vo["job"]):
            done = OUTPUT_DIR / "audio" / f"voice_{vo['job']}.wav"
            vo["audio"] = done.name if done.exists() else None
            vo["job"] = vo["job"] if vo["audio"] else None
        with _lock:
            save(prod)
        advance(prod["id"])


def cancel(pid: str) -> dict:
    with _lock:
        prod = load(pid)
        prod["status"] = "cancelled"
        save(prod)
    return prod
