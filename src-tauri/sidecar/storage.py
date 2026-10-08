"""Disk accounting for models and outputs.

Shared resources (text encoders, VAEs) are reference-counted: deleting a model
removes a shared folder only when no other installed model still needs it.
Anything in the models folder that no active model claims is reported as an
orphan (old models, abandoned partial downloads) so the user can reclaim it.
"""
import os
import re
import shutil
from pathlib import Path

from model_registry import MODEL_CONFIG, SHARED_RESOURCES, MODELS_DIR, OUTPUT_DIR


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass  # folder vanished mid-scan (e.g. being deleted); count what's left
    return total


def _shared_keys(model_cfg: dict) -> list[str]:
    key = model_cfg.get("distribution", {}).get("shared_resources")
    if not key:
        return []
    return [key] if isinstance(key, str) else list(key)


def _model_dirs(model_id: str) -> list[str]:
    """Folder names under MODELS_DIR owned by a model (not shared)."""
    cfg = MODEL_CONFIG[model_id]
    return list(cfg.get("distribution", {}).get("extra_dirs", [])) + [model_id]


def report(downloaded: dict[str, bool]) -> dict:
    usage = shutil.disk_usage(MODELS_DIR)
    claimed: set[str] = set()

    models = []
    for mid, cfg in MODEL_CONFIG.items():
        own = sum(dir_size(MODELS_DIR / d) for d in _model_dirs(mid))
        claimed.update(d.lower() for d in _model_dirs(mid))
        models.append({
            "id": mid,
            "name": cfg.get("identity", {}).get("display_name", mid),
            "own_bytes": own,
            "shared": _shared_keys(cfg),
            "downloaded": bool(downloaded.get(mid)),
        })

    shared = []
    for key, res in SHARED_RESOURCES.items():
        local = res["local_dir"]
        users = [m["id"] for m in models if key in m["shared"]]
        installed_users = [m["id"] for m in models if key in m["shared"] and m["downloaded"]]
        if users:
            claimed.add(local.lower())
        shared.append({
            "key": key, "dir": local, "bytes": dir_size(MODELS_DIR / local),
            "used_by": users, "used_by_installed": installed_users,
        })

    orphans = []
    if MODELS_DIR.exists():
        for entry in sorted(MODELS_DIR.iterdir()):
            if entry.name.lower() in claimed or entry.name.startswith("."):
                continue
            orphans.append({"name": entry.name, "bytes": dir_size(entry)})

    outputs_bytes = dir_size(OUTPUT_DIR)
    return {
        "models_dir": str(MODELS_DIR),
        "output_dir": str(OUTPUT_DIR),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "models": models,
        "shared": shared,
        "orphans": orphans,
        "outputs_bytes": outputs_bytes,
        "models_bytes": sum(m["own_bytes"] for m in models) + sum(s["bytes"] for s in shared)
                        + sum(o["bytes"] for o in orphans),
    }


def _rmtree(path: Path):
    path = path.resolve()
    if not path.is_relative_to(MODELS_DIR.resolve()) or path == MODELS_DIR.resolve():
        raise ValueError(f"Refusing to delete outside models dir: {path}")
    if path.exists():
        shutil.rmtree(path)


def delete_model(model_id: str, downloaded: dict[str, bool]) -> list[str]:
    """Delete a model's own files plus any shared resource no other installed
    model uses. Returns the folder names removed."""
    removed = []
    for d in _model_dirs(model_id):
        if (MODELS_DIR / d).exists():
            _rmtree(MODELS_DIR / d)
            removed.append(d)
    for key in _shared_keys(MODEL_CONFIG[model_id]):
        still_needed = any(
            key in _shared_keys(cfg) and downloaded.get(mid)
            for mid, cfg in MODEL_CONFIG.items() if mid != model_id
        )
        if not still_needed and key in SHARED_RESOURCES:
            local = SHARED_RESOURCES[key]["local_dir"]
            if (MODELS_DIR / local).exists():
                _rmtree(MODELS_DIR / local)
                removed.append(local)
    return removed


def delete_orphan(name: str, downloaded: dict[str, bool]) -> None:
    valid = {o["name"] for o in report(downloaded)["orphans"]}
    if name not in valid:
        raise ValueError("Not an orphaned folder")
    _rmtree(MODELS_DIR / name)


# ── Outputs (gallery) ────────────────────────────────────────────────────────

OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}\.mp4$")


def output_path(name: str) -> Path:
    if not OUTPUT_NAME_RE.match(name):
        raise ValueError("Invalid output name")
    return OUTPUT_DIR / name


IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}\.(png|jpg|jpeg|webp)$")
IMAGE_DIR = OUTPUT_DIR / "images"


def image_path(name: str) -> Path:
    if not IMAGE_NAME_RE.match(name):
        raise ValueError("Invalid image name")
    return IMAGE_DIR / name


AUDIO_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,80}\.(wav|mp3|m4a)$")
AUDIO_DIR = OUTPUT_DIR / "audio"


def audio_path(name: str) -> Path:
    if not AUDIO_NAME_RE.match(name):
        raise ValueError("Invalid audio name")
    return AUDIO_DIR / name
