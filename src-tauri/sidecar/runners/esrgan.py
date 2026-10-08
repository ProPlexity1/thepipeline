"""Real-ESRGAN x4 "quick sharpen": frame-by-frame, streamed.

Frames are read, upscaled in tiles, resized down to the target and written one
at a time, so memory stays flat regardless of clip length (a whole 5s clip at
4x would be ~9GB of frames if held at once). The source audio is copied over.
"""
import subprocess
import time

from model_registry import resolve_asset_path, OUTPUT_DIR
from runners.common import Reporter, free_memory

TILE, PAD = 512, 16  # measured: 384px 1.26s/frame, 512px 1.18s, larger no faster


def _upscale_tiled(model, img, scale: int, torch):
    """img: (1,3,H,W) in [0,1] on CUDA. Tiles keep VRAM bounded at any size."""
    _, _, h, w = img.shape
    out = torch.zeros((1, 3, h * scale, w * scale), device=img.device, dtype=img.dtype)
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y0, x0 = max(0, y - PAD), max(0, x - PAD)
            y1, x1 = min(h, y + TILE + PAD), min(w, x + TILE + PAD)
            tile = model(img[:, :, y0:y1, x0:x1])
            ty, tx = (y - y0) * scale, (x - x0) * scale
            th, tw = (min(y + TILE, h) - y) * scale, (min(x + TILE, w) - x) * scale
            out[:, :, y * scale:y * scale + th, x * scale:x * scale + tw] = tile[:, :, ty:ty + th, tx:tx + tw]
    return out


def _mux_audio(video_only, source, out):
    """Copy the source's audio (if any) onto the upscaled video."""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video_only), "-i", str(source),
                        "-map", "0:v", "-map", "1:a?", "-c", "copy", "-shortest", str(out)],
                       capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.returncode == 0


def run(params: dict, model_cfg: dict, report: Reporter) -> str:
    import imageio.v3 as iio
    import imageio
    import numpy as np
    import torch
    import torch.nn.functional as F
    from spandrel import ModelLoader

    src = OUTPUT_DIR / params["source"]
    if not src.exists():
        raise RuntimeError("The original video is no longer there.")
    meta = iio.immeta(str(src), plugin="pyav")
    fps = float(meta.get("fps") or 24)
    total = int(meta.get("frames") or 0) or sum(1 for _ in iio.imiter(str(src), plugin="pyav"))

    report.status("loading_model", 5, message="Loading the sharpening model")
    desc = ModelLoader().load_from_file(str(resolve_asset_path(model_cfg["runtime"]["weights"])))
    model = desc.model.eval().cuda()
    use_half = desc.supports_half
    if use_half:
        model = model.half()
    scale = desc.scale

    first = iio.imread(str(src), index=0, plugin="pyav")
    h, w = first.shape[:2]
    target = int(params["target_short_edge"])
    factor = target / min(w, h)
    out_w, out_h = int(round(w * factor / 2) * 2), int(round(h * factor / 2) * 2)
    report.log(f"Source {w}x{h} -> {out_w}x{out_h} (x{scale} model, then resized)")

    tmp = OUTPUT_DIR / f"video_{params['job_id']}.noaudio.mp4"
    out = OUTPUT_DIR / f"video_{params['job_id']}.mp4"
    t0 = time.time()
    with torch.no_grad(), imageio.get_writer(str(tmp), fps=fps, codec="libx264", quality=9,
                                             macro_block_size=2, ffmpeg_log_level="error") as writer:
        for i, frame in enumerate(iio.imiter(str(src), plugin="pyav")):
            x = torch.from_numpy(np.ascontiguousarray(frame)).cuda().permute(2, 0, 1)[None].float() / 255.0
            if use_half:
                x = x.half()
            y = _upscale_tiled(model, x, scale, torch).float()
            y = F.interpolate(y, size=(out_h, out_w), mode="bicubic", antialias=True, align_corners=False)
            writer.append_data((y.clamp(0, 1)[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy())
            if i % 5 == 0 or i == total - 1:
                done = i + 1
                per = (time.time() - t0) / done
                report.status("generating", 10 + 85 * done / max(total, 1), eta=int(per * (total - done)),
                              message=f"Sharpening frame {done} of {total}")
                if i % 20 == 0:
                    report.log(f"Frame {done}/{total} ({per:.2f}s per frame)")

    del model
    free_memory()
    report.status("post_processing", 97, message="Saving video")
    if not _mux_audio(tmp, src, out):
        tmp.replace(out)
    else:
        tmp.unlink(missing_ok=True)
    report.log(f"Saved {out} ({out.stat().st_size / 1e6:.1f}MB)")
    return str(out)
