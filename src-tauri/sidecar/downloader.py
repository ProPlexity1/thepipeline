"""Model downloads: pinned revisions, integrity checks, disk-space guard, resume.

Every Hugging Face file is fetched at the exact revision the model JSON pins
and checked against the hash the Hub publishes for it (sha256 for LFS weights,
git blob sha1 for small files) before it is moved into place, so a truncated
or tampered file can never be loaded.
"""
import hashlib
import io
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable

import requests

from model_registry import (MODELS_DIR, SHARED_RESOURCES, get_download_manifest, get_model,
                            shared_resource_keys, validate_runtime_assets)
from optimize import converted_files, run_post_download

CHUNK = 1024 * 1024
SEGMENTS = 6                          # parallel connections for big files
SEGMENTED_MIN_BYTES = 256 * 1024 ** 2
RETRIES = 6
DISK_MARGIN_BYTES = 3 * 1024 ** 3  # never fill the drive completely
_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


class DownloadCancelled(Exception):
    pass


class _FileLock:
    """OS-level exclusive lock on <file>.lock. Another process (a second app
    window, a leftover engine) can't download the same file at the same time;
    the OS releases the lock automatically if the holder dies."""

    def __init__(self, target: Path):
        self.path = target.with_name(target.name + ".lock")
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError("This model is already being downloaded by another ThePipeline process.")
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    os.lseek(self.fd, 0, 0)
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(self.fd)
            try:
                self.path.unlink()
            except OSError:
                pass


def _lock_for(path: Path) -> threading.Lock:
    with _file_locks_guard:
        return _file_locks.setdefault(str(path).lower(), threading.Lock())


def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _fetch_file_info(repo: str, revision: str, paths: list[str]) -> dict[str, dict]:
    """size + expected hash for each path, from the Hub's paths-info API."""
    from huggingface_hub import HfApi
    infos = HfApi(token=os.environ.get("HF_TOKEN") or None).get_paths_info(
        repo, paths, revision=revision, expand=False
    )
    out = {}
    for info in infos:
        lfs = getattr(info, "lfs", None)
        out[info.path] = {
            "size": info.size,
            "sha256": lfs.sha256 if lfs else None,
            "git_sha1": None if lfs else getattr(info, "blob_id", None),
        }
    missing = set(paths) - set(out)
    if missing:
        raise RuntimeError(f"{repo}@{revision[:10]} is missing files: {sorted(missing)}")
    return out


RECOVER_BLOCK = 1024 * 1024


def _recover_segments(part: Path, size: int) -> dict | None:
    """Rebuild segment progress from a preallocated .part file by finding the
    regions that still hold only zeros (never written). Segments don't start on
    block boundaries, so the blocks on *both* sides of each such region may be
    part data, part zeros: they are re-fetched too."""
    zero = bytes(RECOVER_BLOCK)
    holes: list[list[int]] = []
    with open(part, "rb") as f:
        pos = 0
        while pos < size:
            block = f.read(RECOVER_BLOCK)
            if not block:
                break
            if block == zero[:len(block)]:
                if holes and holes[-1][1] == pos - 1:
                    holes[-1][1] = pos + len(block) - 1
                else:
                    holes.append([max(0, pos - RECOVER_BLOCK), pos + len(block) - 1])
            pos += len(block)
    written = size - sum(b - a + 1 for a, b in holes)
    if written <= 0:
        return None
    if not holes:  # everything looks written; verification will decide
        return {"size": size, "segments": [[0, size - 1, size]]}
    # Pad each hole by one block after it as well, then merge overlaps.
    holes = [[a, min(size - 1, b + RECOVER_BLOCK)] for a, b in holes]
    merged: list[list[int]] = []
    for a, b in sorted(holes):
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    segments, cursor = [], 0
    for a, b in merged:
        if a > cursor:
            segments.append([cursor, a - 1, a - cursor])  # recovered data
        segments.append([a, b, 0])
        cursor = b + 1
    if cursor < size:
        segments.append([cursor, size - 1, size - cursor])
    return {"size": size, "segments": segments, "recovered": True}


def _write_atomic(path: Path, text: str) -> bool:
    """Best-effort atomic write of a resume checkpoint. Windows antivirus and
    indexers briefly lock freshly written files, so retry a little, and never
    fail the download over it: resuming from an older checkpoint only means
    re-fetching a few bytes, which is always safe."""
    tmp = path.with_name(path.name + ".tmp")
    for attempt in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())  # survive a power cut, not just a crash
            os.replace(tmp, path)
            return True
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    return False


def _verify(path: Path, info: dict, hasher_state=None) -> None:
    size = path.stat().st_size
    if info["size"] is not None and size != info["size"]:
        raise RuntimeError(f"{path.name}: size {size} != expected {info['size']}")
    if info["sha256"]:
        digest = hasher_state.hexdigest() if hasher_state else _hash_file(path, "sha256")
        if digest != info["sha256"]:
            raise RuntimeError(f"{path.name}: sha256 mismatch (file corrupted or tampered)")
    elif info["git_sha1"]:
        h = hashlib.sha1(f"blob {size}\0".encode())
        h.update(path.read_bytes())
        if h.hexdigest() != info["git_sha1"]:
            raise RuntimeError(f"{path.name}: checksum mismatch")


def _hash_file(path: Path, algo: str, h=None):
    h = h or hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK * 8), b""):
            h.update(chunk)
    return h.hexdigest() if algo else h


def _safe_extract(zf: zipfile.ZipFile, prefix: str, dest: Path) -> int:
    """Extract members under `prefix` into dest, refusing any path that escapes it."""
    dest = dest.resolve()
    count = 0
    for member in zf.namelist():
        if not member.startswith(prefix) or member.endswith("/"):
            continue
        rel = member[len(prefix):]
        target = (dest / rel).resolve()
        if not target.is_relative_to(dest):
            raise RuntimeError(f"Refusing unsafe path in archive: {member}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        count += 1
    return count


class ModelDownload:
    """One model download; runs in a worker thread and reports via callback."""

    def __init__(self, model_id: str, on_progress: Callable[[dict], None]):
        self.model_id = model_id
        self.on_progress = on_progress
        self.cancelled = threading.Event()
        self.done_bytes = 0
        self.total_bytes = 0
        self._last_emit = 0.0
        self._speed_window: list[tuple[float, int]] = []

    def cancel(self):
        self.cancelled.set()

    def _emit(self, force=False, **extra):
        now = time.time()
        if not force and now - self._last_emit < 0.5:
            return
        self._last_emit = now
        self._speed_window.append((now, self.done_bytes))
        self._speed_window = [(t, b) for t, b in self._speed_window if now - t <= 5]
        t0, b0 = self._speed_window[0]
        speed = (self.done_bytes - b0) / (now - t0) if now > t0 else 0.0
        remaining = max(self.total_bytes - self.done_bytes, 0)
        payload = {
            "type": "download_progress",
            "model_id": self.model_id,
            "progress": round(min(self.done_bytes / self.total_bytes * 100, 99.9), 1) if self.total_bytes else 0.0,
            "downloaded_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "speed_mbps": round(speed / 1024 ** 2, 1),
            "eta_seconds": int(remaining / speed) if speed > 0 else 0,
            "downloading": True,
            "downloaded": False,
        }
        payload.update(extra)
        self.on_progress(payload)

    # ── HF files ────────────────────────────────────────────────────────────
    def _download_hf_file(self, entry: dict, info: dict):
        final = MODELS_DIR / entry["local_path"]
        final.parent.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_url
        url = hf_hub_url(entry["repo_id"], entry["filename"], revision=entry["revision"])

        if (info["size"] or 0) >= SEGMENTED_MIN_BYTES:
            self._download_segmented(url, final, info)
            return

        for attempt in range(RETRIES):
            try:
                self._download_stream(url, final, info)
                return
            except (requests.RequestException, OSError) as e:
                if self.cancelled.is_set() or attempt == RETRIES - 1:
                    raise
                self._emit(force=True, stage="retrying", error=str(e)[:200])
                time.sleep(min(2 ** attempt, 30))

    def _download_segmented(self, url: str, final: Path, info: dict):
        """Parallel ranged download for large files. Per-segment progress is
        checkpointed in <file>.part.json so any interruption resumes exactly."""
        import json
        part = final.with_name(final.name + ".part")
        state_path = final.with_name(final.name + ".part.json")
        size = info["size"]
        counted = part.stat().st_size if part.exists() else 0  # what run() already added

        state = None
        if state_path.exists() and part.exists():
            try:
                state = json.loads(state_path.read_text())
                if state.get("size") != size:
                    state = None
            except ValueError:
                state = None
        if state is None and part.exists() and part.stat().st_size == size:
            # Full-size file but the checkpoint is gone or garbage (e.g. power cut):
            # rebuild progress from the data itself. The final sha256 check makes
            # this safe even if a guess is wrong.
            state = _recover_segments(part, size)
            if state:
                _write_atomic(state_path, json.dumps(state))
        if state is None:
            # A plain sequential .part (older downloader) counts as a done prefix;
            # a full-size file with nothing recoverable is just preallocated space.
            prefix = part.stat().st_size if part.exists() else 0
            if prefix >= size:
                prefix = 0
            with open(part, "ab" if prefix else "wb") as f:
                f.truncate(size)
            rest = size - prefix
            n = max(1, min(SEGMENTS, rest // (64 * 1024 ** 2) or 1))
            bounds = [prefix + rest * i // n for i in range(n + 1)]
            segs = [[bounds[i], bounds[i + 1] - 1, 0] for i in range(n)]
            if prefix:
                segs.insert(0, [0, prefix - 1, prefix])
            state = {"size": size, "segments": segs}
            _write_atomic(state_path, json.dumps(state))

        # Bytes already present were counted as `part` size by run(); recount precisely.
        self.done_bytes -= counted
        self.done_bytes += sum(s[2] for s in state["segments"])
        lock = threading.Lock()
        errors: list[BaseException] = []

        def worker(seg):
            start, end, _ = seg
            for attempt in range(RETRIES):
                if seg[2] >= end - start + 1:
                    return
                try:
                    headers = {**_hf_headers(), "Range": f"bytes={start + seg[2]}-{end}"}
                    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                        if r.status_code != 206:
                            raise requests.RequestException(f"Range not honoured (HTTP {r.status_code})")
                        with open(part, "r+b") as f:
                            f.seek(start + seg[2])
                            for chunk in r.iter_content(chunk_size=CHUNK):
                                if self.cancelled.is_set():
                                    return
                                f.write(chunk)
                                # Count bytes only once they reach the OS, so the
                                # checkpoint never claims data a crash could lose.
                                f.flush()
                                with lock:
                                    seg[2] += len(chunk)
                                    self.done_bytes += len(chunk)
                    return
                except (requests.RequestException, OSError) as e:
                    if self.cancelled.is_set() or attempt == RETRIES - 1:
                        errors.append(e)
                        return
                    time.sleep(min(2 ** attempt, 30))

        threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in state["segments"]]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            with lock:
                snapshot = json.dumps(state)
            _write_atomic(state_path, snapshot)
            self._emit()
        _write_atomic(state_path, json.dumps(state))
        if self.cancelled.is_set():
            raise DownloadCancelled()
        if errors:
            raise errors[0]

        self._emit(force=True, stage="verifying", file=final.name)
        try:
            _verify(part, info)
        except Exception:
            part.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
            raise
        state_path.unlink(missing_ok=True)
        part.replace(final)

    def _download_stream(self, url: str, final: Path, info: dict):
        part = final.with_name(final.name + ".part")
        hasher = hashlib.sha256() if info["sha256"] else None
        have = part.stat().st_size if part.exists() else 0
        if have and hasher:
            _hash_file(part, None, hasher)  # resume: hash what we already have
        headers = _hf_headers() if "huggingface.co" in url else {}  # never send the HF token elsewhere
        if have:
            headers["Range"] = f"bytes={have}-"

        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            if have and r.status_code != 206:  # server ignored the range: restart
                self.done_bytes -= have
                have, hasher = 0, (hashlib.sha256() if info["sha256"] else None)
            r.raise_for_status()
            with open(part, "ab" if have else "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if self.cancelled.is_set():
                        raise DownloadCancelled()
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    self.done_bytes += len(chunk)
                    self._emit()

        self._emit(force=True, stage="verifying", file=final.name)
        try:
            _verify(part, info, hasher)
        except Exception:
            part.unlink(missing_ok=True)  # corrupt: never resume from it
            raise
        part.replace(final)

    # ── GitHub archives (pinned commit, safe extraction) ────────────────────
    def _download_github(self, entry: dict):
        marker = MODELS_DIR / entry["local_path"]
        if marker.exists():
            return
        repo, rev = entry["repo"], entry["revision"]
        if len(rev) != 40:
            raise RuntimeError(f"GitHub source {repo} must be pinned to a full commit sha, got '{rev}'")
        r = requests.get(f"https://github.com/{repo}/archive/{rev}.zip", timeout=120)
        r.raise_for_status()
        install_dir = MODELS_DIR / entry["install_dir"]
        if install_dir.exists():
            shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True)
        sub = entry["extract_path"].strip("/")
        prefix = f"{repo.split('/')[-1]}-{rev}/" + (f"{sub}/" if sub else "")
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            if _safe_extract(zf, prefix, install_dir) == 0:
                raise RuntimeError(f"Archive had no files under '{entry['extract_path']}'")
        if not marker.exists():
            raise RuntimeError(f"Marker {marker.name} missing after extraction")

    # ── GitHub release assets (prebuilt binaries, sha256-verified) ─────────
    def _download_release(self, entry: dict):
        marker = MODELS_DIR / entry["local_path"]
        if marker.exists():
            return
        if not entry.get("sha256") or len(entry["sha256"]) != 64:
            raise RuntimeError(f"Release asset {entry['asset']} must be pinned to a sha256 digest")
        install_dir = MODELS_DIR / entry["install_dir"]
        install_dir.mkdir(parents=True, exist_ok=True)
        zip_path = install_dir / entry["asset"]
        url = f"https://github.com/{entry['repo']}/releases/download/{entry['tag']}/{entry['asset']}"
        info = {"size": entry["size"], "sha256": entry["sha256"], "git_sha1": None}
        if not zip_path.exists():
            for attempt in range(RETRIES):
                try:
                    self._download_stream(url, zip_path, info)
                    break
                except (requests.RequestException, OSError) as e:
                    if self.cancelled.is_set() or attempt == RETRIES - 1:
                        raise
                    self._emit(force=True, stage="retrying", error=str(e)[:200])
                    time.sleep(min(2 ** attempt, 30))
        else:
            _verify(zip_path, info)
        self._emit(force=True, stage="installing", file=entry["asset"])
        with zipfile.ZipFile(zip_path) as zf:
            if _safe_extract(zf, "", install_dir) == 0:
                raise RuntimeError(f"{entry['asset']} was empty")
        zip_path.unlink(missing_ok=True)
        marker.write_text(entry["sha256"], encoding="utf-8")

    # ── Orchestration ────────────────────────────────────────────────────────
    def run(self):
        manifest = get_download_manifest(self.model_id)
        hf = [e for e in manifest if e["provider"] == "huggingface"]
        gh = [e for e in manifest if e["provider"] == "github_archive"]
        releases = [e for e in manifest if e["provider"] == "github_release"]

        self._emit(force=True, stage="checking")
        infos: dict[str, dict] = {}
        groups: dict[tuple, list] = {}
        for e in hf:
            groups.setdefault((e["repo_id"], e["revision"]), []).append(e)
        for (repo, rev), entries in groups.items():
            fetched = _fetch_file_info(repo, rev, [e["filename"] for e in entries])
            for e in entries:
                infos[e["local_path"]] = fetched[e["filename"]]

        converted = {}
        pending = []
        for e in hf:
            final = MODELS_DIR / e["local_path"]
            info = infos[e["local_path"]]
            self.total_bytes += info["size"] or 0
            top, _, rel = e["local_path"].partition("/")
            if top not in converted:
                converted[top] = converted_files(MODELS_DIR / top)
            if final.exists() and rel in converted[top]:
                self.done_bytes += info["size"] or 0  # already optimised in place
                continue
            if final.exists() and (info["size"] is None or final.stat().st_size == info["size"]):
                self.done_bytes += final.stat().st_size
                continue
            part = final.with_name(final.name + ".part")
            if part.exists():
                self.done_bytes += part.stat().st_size
            pending.append(e)

        for e in releases:
            if not (MODELS_DIR / e["local_path"]).exists():
                self.total_bytes += e["size"]
                z = MODELS_DIR / e["install_dir"] / e["asset"]
                if z.exists():
                    self.done_bytes += z.stat().st_size
        needed = self.total_bytes - self.done_bytes
        free = shutil.disk_usage(MODELS_DIR).free
        if needed + DISK_MARGIN_BYTES > free:
            raise RuntimeError(
                f"Not enough disk space: this model needs {needed / 1024 ** 3:.1f} GB more, "
                f"but only {free / 1024 ** 3:.1f} GB is free. Delete unused models in Storage."
            )

        for e in pending:
            final = MODELS_DIR / e["local_path"]
            with _lock_for(final), _FileLock(final):  # no races within or across processes
                if final.exists():
                    continue
                self._download_hf_file(e, infos[e["local_path"]])
        for e in gh:
            if self.cancelled.is_set():
                raise DownloadCancelled()
            self._download_github(e)
        for e in releases:
            if self.cancelled.is_set():
                raise DownloadCancelled()
            with _lock_for(MODELS_DIR / e["local_path"]), _FileLock(MODELS_DIR / e["local_path"]):
                self._download_release(e)

        for key in shared_resource_keys(get_model(self.model_id)):
            res = SHARED_RESOURCES.get(key, {})
            if res.get("convert"):
                self._emit(force=True, stage="optimizing")
                run_post_download(res, MODELS_DIR / res["local_dir"])

        ok, msg = validate_runtime_assets(self.model_id, MODELS_DIR)
        if not ok:
            raise RuntimeError(msg)


def estimated_size_gb(model_id: str) -> float:
    return float(get_model(model_id).get("distribution", {}).get("estimated_download_size_gb", 0))
