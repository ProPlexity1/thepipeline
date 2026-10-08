"""SeedVR2 3B one-step video restoration/upscale, on the private ComfyUI engine.

Mirrors ComfyUI's official "Upscale Video (SeedVR2 3B Int8)" workflow: resize
with Lanczos to the target size, encode, restore in one diffusion step per
temporal chunk (chunk length chosen automatically to fit free VRAM), decode,
colour-match to the source, and keep the source's audio and frame rate.
"""
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from model_registry import resolve_asset_path, MODELS_DIR, OUTPUT_DIR
from runners.common import Reporter
from runners.comfy_engine import ComfyEngine, remove_work_dir, run_graph


def source_size(path) -> tuple[int, int]:
    import imageio.v3 as iio
    frame = iio.imread(str(path), index=0, plugin="pyav")
    return frame.shape[1], frame.shape[0]


# Clips longer than this are upscaled in segments: a 10s clip at 1080p needs
# more VRAM/RAM than an 8GB card has and slows ~4x (measured on an RTX 5060).
SEGMENT_FRAMES = 125


def frame_count(path) -> int:
    import av
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        return v.frames or sum(1 for _ in c.demux(v))


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run_ffmpeg(args: list[str]):
    r = subprocess.run([_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", *args], capture_output=True,
                       text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-300:]}")


def split_video(src: Path, out_dir: Path, frames: int) -> list[Path]:
    """Cut into near-equal segments of at most SEGMENT_FRAMES frames (video only,
    near-lossless so the enhancer sees the original detail)."""
    n = math.ceil(frames / SEGMENT_FRAMES)
    size = math.ceil(frames / n)
    parts = []
    for i in range(n):
        a, b = i * size, min(frames, (i + 1) * size)
        out = out_dir / f"segment_{i:02d}.mp4"
        _run_ffmpeg(["-i", str(src), "-vf", f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS",
                     "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p", str(out)])
        parts.append(out)
    return parts


def join_segments(parts: list[Path], audio_src: Path, out: Path):
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio_src),
                 "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", str(out)])
    listing.unlink(missing_ok=True)


def build_graph(rt: dict, src_name: str, multiplier: float, seed: int, with_audio: bool = True) -> dict:
    f = rt["files"]
    # Measured on an RTX 5060, 81 frames to 1080p. Encode: 768px tiles beat the
    # template's 512/128 (147s vs 175s). Decode: 768px tiles spill VRAM on full
    # clips (896s vs 342s), so decode keeps 512px tiles with a lighter overlap.
    enc_tiles = {"tile_size": 768, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}
    dec_tiles = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}
    return {
        "load": {"class_type": "LoadVideo", "inputs": {"file": src_name}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["load", 0]}},
        "resize": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["parts", 0], "resize_type": "scale by multiplier",
            "resize_type.multiplier": round(multiplier, 4), "scale_method": "lanczos"}},
        "pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["resize", 0]}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": f["unet"], "weight_dtype": "default"}},
        "encode": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0], **enc_tiles}},
        "chunk": {"class_type": "SeedVR2TemporalChunk", "inputs": {
            "latent": ["encode", 0], "temporal_overlap": rt.get("temporal_overlap", 2), "chunking_mode": "auto"}},
        "cond": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["unet", 0], "vae_conditioning": ["chunk", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "seed": seed, "steps": 1, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["cond", 0], "negative": ["cond", 1],
            "latent_image": ["chunk", 0], "denoise": 1.0}},
        "merge": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["sample", 0], "temporal_overlap": ["chunk", 1]}},
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0], **dec_tiles}},
        "post": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "images": ["decode", 0], "original_resized_images": ["resize", 0],
            "color_correction_method": rt.get("color_correction", "lab")}},
        "video": {"class_type": "CreateVideo", "inputs": {
            "images": ["post", 0], "fps": ["parts", 2], **({"audio": ["parts", 1]} if with_audio else {})}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["video", 0], "filename_prefix": "pipeline_enhanced", "format": "mp4", "format.codec": "h264"}},
    }


STAGES = {
    "load": (6, "Reading the video"), "encode": (12, "Encoding frames"),
    "sample": (25, "Restoring detail"), "decode": (78, "Decoding the upscaled frames"),
    "post": (90, "Matching colours to the original"), "save": (96, "Saving video"),
}


def run(params: dict, model_cfg: dict, report: Reporter) -> str:
    rt = model_cfg["runtime"]
    engine_dir = resolve_asset_path(rt["engine"])
    model_root = resolve_asset_path(rt["model_root"])
    src = OUTPUT_DIR / params["source"]
    if not src.exists():
        raise RuntimeError("The original video is no longer there.")

    w, h = source_size(src)
    target = int(params["target_short_edge"])
    multiplier = max(1.0, target / min(w, h))
    out_w, out_h = round(w * multiplier), round(h * multiplier)
    seed = int.from_bytes(os.urandom(6), "little")
    report.log(f"Source {w}x{h} -> about {out_w}x{out_h} ({multiplier:.2f}x)")

    work = MODELS_DIR.parent / "engine-work" / params["job_id"]
    engine = ComfyEngine(engine_dir, model_root, work, rt.get("engine_args", []), report,
                         folders=("diffusion_models", "vae"))
    report.status("loading_model", 2, message="Starting the enhancer")
    try:
        engine.start()
        frames = frame_count(src)
        if frames > SEGMENT_FRAMES:
            sources = split_video(src, work / "input", frames)
            report.log(f"{frames} frames: upscaling in {len(sources)} parts to fit in memory")
        else:
            shutil.copy2(src, work / "input" / "source.mp4")
            sources = [work / "input" / "source.mp4"]
        t0 = time.time()
        results = []
        for i, part in enumerate(sources):
            n = len(sources)
            label = f" (part {i + 1} of {n})" if n > 1 else ""
            batches = [0]

            def overall(pct: float) -> float:
                return 2 + (i + pct / 100) / n * 94

            def on_event(kind, body):
                node = body.get("node")
                if kind == "executing" and node in STAGES:
                    pct, msg = STAGES[node]
                    report.status("generating" if node == "sample" else "loading_model" if pct < 25 and i == 0
                                  else "post_processing" if pct > 75 and i == n - 1 else "generating",
                                  overall(pct), message=msg + label)
                    report.log(msg + label)
                elif kind == "progress" and node == "sample":
                    batches[0] += 1
                    report.status("generating", overall(min(75, 25 + batches[0] * 1.5)),
                                  message=f"Restoring detail{label}")
                    if batches[0] % 8 == 0:
                        report.log(f"Restored {batches[0]} batches{label} ({time.time() - t0:.0f}s so far)")

            produced = run_graph(engine, build_graph(rt, part.name, multiplier, seed, with_audio=n == 1),
                                 report, on_event)
            if not produced:
                raise RuntimeError(f"The enhancer finished without a video. {engine.tail()}")
            results.append(produced[0])

        out = OUTPUT_DIR / f"video_{params['job_id']}.mp4"
        if len(results) == 1:
            shutil.move(str(results[0]), out)
        else:
            report.status("post_processing", 97, message="Joining the parts")
            join_segments(results, src, out)
        report.log(f"Saved {out} ({out.stat().st_size / 1e6:.1f}MB)")
        return str(out)
    finally:
        engine.stop()
        remove_work_dir(work)
