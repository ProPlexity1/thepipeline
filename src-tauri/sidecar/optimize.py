"""Post-download storage optimisation.

Some upstream repos ship weights in float32 even though inference runs in
bfloat16 (e.g. LTX's 18GB T5 text encoder). Converting those shards once to
bf16 halves disk use and load time with no change in output, because the
loader casts to bf16 anyway.

Conversion happens shard by shard, writing each new file next to the old one
and swapping it in, so at most one extra shard of disk is ever needed. A
marker file records what was converted so the downloader doesn't mistake the
smaller files for corrupt downloads.
"""
import json
from pathlib import Path

MARKER = ".thepipeline_converted.json"
LEGACY_MARKER = ".neuralcut_converted.json"  # written before the rename


def read_marker(resource_dir: Path) -> dict:
    p = resource_dir / MARKER
    if not p.exists():
        p = resource_dir / LEGACY_MARKER
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def converted_files(resource_dir: Path) -> set[str]:
    """Relative paths (posix) whose content was rewritten by a conversion."""
    out = set()
    for info in read_marker(resource_dir).values():
        out.update(info.get("files", []))
    return out


def convert_dir_to_bf16(resource_dir: Path, subdir: str, log=print) -> int:
    """Convert every float32 tensor in <resource_dir>/<subdir>/*.safetensors to
    bfloat16. Returns bytes saved. Idempotent."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    marker = read_marker(resource_dir)
    if marker.get(subdir, {}).get("dtype") == "bfloat16":
        return 0

    folder = resource_dir / subdir
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        return 0
    saved = 0
    touched = []
    for shard in shards:
        tmp = shard.with_suffix(".bf16.tmp")
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            meta = f.metadata() or {}
            for key in f.keys():
                t = f.get_tensor(key)
                tensors[key] = t.to(torch.bfloat16) if t.dtype == torch.float32 else t
        before = shard.stat().st_size
        save_file(tensors, str(tmp), metadata={**meta, "format": "pt"})
        del tensors
        tmp.replace(shard)
        saved += before - shard.stat().st_size
        touched.append(f"{subdir}/{shard.name}")
        log(f"[optimize] {subdir}/{shard.name}: {before / 1e9:.1f}GB -> {shard.stat().st_size / 1e9:.1f}GB")

    index = folder / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text(encoding="utf-8"))
        idx.setdefault("metadata", {})["total_size"] = sum(s.stat().st_size for s in shards)
        index.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        touched.append(f"{subdir}/{index.name}")
    cfg = folder / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text(encoding="utf-8"))
        c["torch_dtype"] = "bfloat16"
        cfg.write_text(json.dumps(c, indent=2), encoding="utf-8")
        touched.append(f"{subdir}/{cfg.name}")

    marker[subdir] = {"dtype": "bfloat16", "files": touched}
    (resource_dir / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return saved


def run_post_download(resource: dict, resource_dir: Path, log=print) -> int:
    saved = 0
    for subdir, dtype in (resource.get("convert") or {}).items():
        if dtype != "bfloat16":
            raise ValueError(f"Unsupported conversion target {dtype}")
        saved += convert_dir_to_bf16(resource_dir, subdir, log)
    return saved
