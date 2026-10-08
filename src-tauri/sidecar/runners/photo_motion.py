"""Photo motion: a cinematic camera move through a still image.

Two-layer 2.5D parallax, the way motion designers do it by hand:
1. A small depth model (Depth Anything V2 Small) finds the subject (the nearest
   large region) and cuts it out with a soft edge.
2. The background behind the subject is rebuilt once (inpainting), so when the
   layers move apart nothing is duplicated or smeared.
3. Subject and background move at different speeds; push-ins aim at the
   subject's face/top and the move is limited so the subject never leaves frame.
Wide scenes with no clear subject get a clean 2D camera move (no parallax, no
artefacts). Every frame keeps the photo's full sharpness. Seconds per shot.

CLI: python -m runners.photo_motion <params.json>
params: {image, out, move, seconds, strength, width, height, fps}
moves: push_in, pull_out, pan_left, pan_right, rise, orbit_left, orbit_right, drift
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

MOVES = ("push_in", "pull_out", "pan_left", "pan_right", "rise", "orbit_left", "orbit_right", "drift")


def model_dir() -> Path:
    from model_registry import MODEL_CONFIG, resolve_asset_path
    return resolve_asset_path(MODEL_CONFIG["depth-anything-v2-small"]["runtime"]["model_root"])


def estimate_disparity(img_rgb: np.ndarray) -> np.ndarray:
    """Relative inverse depth in [0, 1] (1 = nearest)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    root = str(model_dir())
    proc = AutoImageProcessor.from_pretrained(root)
    model = AutoModelForDepthEstimation.from_pretrained(root).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    with torch.no_grad():
        inputs = proc(images=Image.fromarray(img_rgb), return_tensors="pt").to(dev)
        pred = model(**inputs).predicted_depth[None]
        pred = torch.nn.functional.interpolate(pred, size=img_rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0]
    d = pred.float().cpu().numpy()
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def subject_mask(disp: np.ndarray):
    """Soft alpha of the nearest large region, or None if the photo has no clear subject."""
    import cv2
    h, w = disp.shape
    d8 = (disp * 255).astype(np.uint8)
    thr, mask = cv2.threshold(cv2.GaussianBlur(d8, (0, 0), 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Clear separation between near and far is needed for parallax to look right.
    near, far = disp[mask > 0], disp[mask == 0]
    if len(near) == 0 or len(far) == 0 or near.mean() - far.mean() < 0.25:
        return None, None
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None, None
    # Keep large components; the ground plane touching the whole bottom edge isn't a subject.
    keep = np.zeros_like(mask)
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < 0.01 * h * w:
            continue
        if bw > 0.9 * w and y + bh >= h - 2:
            continue  # floor/foreground ground band
        keep[lab == k] = 255
    frac = keep.mean() / 255
    if frac < 0.02 or frac > 0.55:
        return None, None
    ys, xs = np.where(keep > 0)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    keep = cv2.erode(keep, np.ones((3, 3), np.uint8))
    alpha = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (0, 0), 1.6)
    return alpha, bbox


def clean_plate(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Background with the subject removed and filled in, so layers can separate cleanly."""
    import cv2
    h, w = alpha.shape
    hole = (cv2.dilate((alpha > 0.05).astype(np.uint8) * 255, np.ones((1, 1), np.uint8),
                       iterations=1))
    hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, w // 60),) * 2))
    # Inpaint at reduced size (fast, and backgrounds are usually soft), then put it back at full size.
    s = 640 / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    small_hole = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    filled = cv2.inpaint(small, small_hole, 7, cv2.INPAINT_TELEA)
    filled = cv2.GaussianBlur(filled, (0, 0), 1.2)
    big = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(hole.astype(np.float32) / 255.0, (0, 0), 3)[..., None]
    return (img * (1 - m) + big * m).astype(np.uint8)


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def transforms(move: str, t: float, s: float, w: int, h: int, anchor, subject: bool):
    """(bg_scale, bg_dx, bg_dy, fg_scale, fg_dx, fg_dy) — scales about `anchor`, shifts in pixels."""
    e = ease(t)
    k = 1.0 if subject else 0.0  # parallax only when there is a subject layer
    if move in ("push_in", "pull_out"):
        p = e if move == "push_in" else 1 - e
        return 1.03 + 0.05 * s * p, 0, 0, 1.03 + (0.05 + 0.05 * k) * s * p, 0, 0
    base = 1.06 + 0.02 * s
    if move in ("pan_left", "pan_right", "orbit_left", "orbit_right"):
        d = -1 if move in ("pan_left", "orbit_left") else 1
        amt = 0.035 * s * w * (e - 0.5)
        if move.startswith("orbit"):  # layers slide in opposite directions around the subject
            return base, -d * amt * 0.6, 0, base, d * amt * (0.6 * k), 0
        return base, d * amt * 0.6, 0, base, d * amt * (1.0 if subject else 0.6), 0
    if move == "rise":
        amt = 0.03 * s * h * (e - 0.5)
        return base, 0, -amt * 0.6, base, 0, -amt * (1.0 if subject else 0.6)
    # drift: slow push with a gentle sideways slide
    amt = 0.015 * s * w * (e - 0.5)
    return base + 0.02 * s * e, amt * 0.5, 0, base + (0.02 + 0.03 * k) * s * e, amt * (1.0 if subject else 0.5), 0


def affine(scale: float, dx: float, dy: float, anchor) -> np.ndarray:
    ax, ay = anchor
    return np.array([[scale, 0, ax - scale * ax + dx], [0, scale, ay - scale * ay + dy]], np.float32)


def render(image_path: str, out: str, move: str = "push_in", seconds: float = 4.0, strength: float = 1.0,
           width: int = 1920, height: int = 1080, fps: int = 24, on_progress=None) -> dict:
    import cv2
    import imageio_ffmpeg
    if move not in MOVES:
        raise ValueError(f"Unknown move {move}")
    img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]
    scale = max(width / iw, height / ih)
    if abs(scale - 1) > 1e-3:
        img = cv2.resize(img, (round(iw * scale), round(ih * scale)), interpolation=cv2.INTER_LANCZOS4)
    ih, iw = img.shape[:2]
    x0, y0 = (iw - width) // 2, (ih - height) // 2
    img = np.ascontiguousarray(img[y0:y0 + height, x0:x0 + width])
    h, w = img.shape[:2]

    alpha, bbox = subject_mask(estimate_disparity(img))
    subject = alpha is not None
    if subject:
        bx0, by0, bx1, by1 = bbox
        # Aim at the upper part of the subject (faces, product tops) and keep it in frame.
        anchor = ((bx0 + bx1) / 2, by0 + 0.3 * (by1 - by0))
        bg = clean_plate(img, alpha)
        fg = np.dstack([img, (alpha * 255).astype(np.uint8)])
    else:
        anchor = (w / 2, h / 2)
        bg, fg = img, None

    # Scale the whole move down if the subject would be pushed out of frame.
    if subject:
        for _ in range(8):
            _, _, _, fs, fdx, fdy = transforms(move, 1.0, strength, w, h, anchor, True)
            m = affine(fs, fdx, fdy, anchor)
            corners = np.array([[bx0, by0, 1], [bx1, by0, 1], [bx0, by1, 1], [bx1, by1, 1]], np.float32) @ m.T
            out_l, out_t = -corners[:, 0].min(), -corners[:, 1].min()
            out_r, out_b = corners[:, 0].max() - w, corners[:, 1].max() - h
            # Only the top and sides matter: subjects often continue past the bottom edge.
            orig_l, orig_t, orig_r = -bx0, -by0, bx1 - w
            if out_l <= max(orig_l, -8) + 2 and out_t <= max(orig_t, -8) + 2 and out_r <= max(orig_r, -8) + 2:
                break
            strength *= 0.75

    n = max(2, int(round(seconds * fps)))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                             "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    for k in range(n):
        bs, bdx, bdy, fs, fdx, fdy = transforms(move, k / (n - 1), strength, w, h, anchor, subject)
        frame = cv2.warpAffine(bg, affine(bs, bdx, bdy, anchor), (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        if subject:
            layer = cv2.warpAffine(fg, affine(fs, fdx, fdy, anchor), (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            a = layer[..., 3:4].astype(np.float32) / 255.0
            frame = (frame * (1 - a) + layer[..., :3] * a).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if on_progress and k % 12 == 0:
            on_progress(k / n)
    proc.stdin.close()
    proc.wait()
    return {"frames": n, "width": w, "height": h, "seconds": round(n / fps, 2), "subject_layer": subject}


if __name__ == "__main__":
    p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(render(p["image"], p["out"], p.get("move", "push_in"), p.get("seconds", 4.0), p.get("strength", 1.0),
                            p.get("width", 1920), p.get("height", 1080), p.get("fps", 24))), flush=True)
