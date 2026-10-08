"""Track a patch on clothing through a generated video and composite a real logo onto it.

Video models can't keep small text crisp in motion, so the logo is added after
generation, like a VFX "logo replacement": feature points on the fabric around
the patch are tracked frame to frame (Lucas-Kanade optical flow), a similarity
transform (position, scale, rotation) is estimated with RANSAC, the whole path
is smoothed, and the exact logo file is warped onto every frame with the
footage's brightness and softness.

usage: logo_track.py <video> <logo.png> <out.mp4> <cx> <cy> <height> [--keyframe W H]
  cx, cy, height: the logo's centre and height in the first frame (pixels). With
  --keyframe, they are given in the keyframe image's coordinates (W x H) and are
  mapped to the video like ComfyUI's centre-crop resize.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def read_video(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def track(frames: list[np.ndarray], cx: float, cy: float, size: float) -> np.ndarray:
    """Per-frame (cx, cy, scale, angle) of the patch relative to frame 0."""
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = gray[0].shape

    def region_mask():
        m = np.zeros((h, w), np.uint8)
        r = int(size * 1.6)
        cv2.rectangle(m, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), 255, -1)
        return m

    pts = cv2.goodFeaturesToTrack(gray[0], 80, 0.01, 4, mask=region_mask())
    state = [np.array([cx, cy, 1.0, 0.0])]
    M_total = np.eye(3)
    for k in range(1, len(frames)):
        if pts is None or len(pts) < 10:
            # Re-seed points around the current patch estimate.
            ccx, ccy, s = state[-1][:3]
            m = np.zeros((h, w), np.uint8)
            r = int(size * s * 1.6)
            cv2.rectangle(m, (int(ccx - r), int(ccy - r)), (int(ccx + r), int(ccy + r)), 255, -1)
            pts = cv2.goodFeaturesToTrack(gray[k - 1], 80, 0.01, 4, mask=m)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[k - 1], gray[k], pts, None, winSize=(21, 21), maxLevel=3)
        good_old, good_new = pts[st == 1], nxt[st == 1]
        M = None
        if len(good_old) >= 6:
            M, inl = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.0)
            if M is not None and inl is not None:
                good_new = good_new[inl.ravel() == 1]
        if M is None:
            M = np.array([[1, 0, 0], [0, 1, 0]], float)
        M_total = np.vstack([M, [0, 0, 1]]) @ M_total
        c = M_total @ np.array([cx, cy, 1.0])
        scale = float(np.hypot(M_total[0, 0], M_total[1, 0]))
        angle = float(np.degrees(np.arctan2(M_total[1, 0], M_total[0, 0])))
        state.append(np.array([c[0], c[1], scale, angle]))
        pts = good_new.reshape(-1, 1, 2) if len(good_new) else None
    return np.array(state)


def smooth(path: np.ndarray, radius: int = 3) -> np.ndarray:
    """Centred moving average (offline, so no lag)."""
    out = path.copy()
    n = len(path)
    for i in range(n):
        a, b = max(0, i - radius), min(n, i + radius + 1)
        out[i] = path[a:b].mean(axis=0)
    return out


def composite(frames, path, logo_rgba: np.ndarray, height0: float, ring_ref: float):
    lh, lw = logo_rgba.shape[:2]
    base_scale = height0 / lh
    out = []
    for f, (cx, cy, s, ang) in zip(frames, path):
        sc = base_scale * s
        M = cv2.getRotationMatrix2D((lw / 2, lh / 2), -ang, sc)
        M[0, 2] += cx - lw / 2
        M[1, 2] += cy - lh / 2
        warped = cv2.warpAffine(logo_rgba, M, (f.shape[1], f.shape[0]), flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)).astype(np.float32)
        # Match the scene: brightness of the fabric around the patch vs the first frame, plus lens softness.
        r = int(height0 * s)
        y0, y1 = max(0, int(cy - r)), min(f.shape[0], int(cy + r))
        x0, x1 = max(0, int(cx - r)), min(f.shape[1], int(cx + r))
        ring = float(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).mean()) if y1 > y0 and x1 > x0 else ring_ref
        gain = np.clip(ring / max(ring_ref, 1.0), 0.75, 1.25)
        warped[..., :3] *= gain
        warped = cv2.GaussianBlur(warped, (0, 0), 0.6)
        a = warped[..., 3:4] / 255.0
        comp = f.astype(np.float32) * (1 - a) + warped[..., :3] * a
        out.append(np.clip(comp, 0, 255).astype(np.uint8))
    return out


def main():
    video, logo_path, out_path = sys.argv[1:4]
    cx, cy, height = map(float, sys.argv[4:7])
    frames, fps = read_video(video)
    H, W = frames[0].shape[:2]
    if "--keyframe" in sys.argv:
        i = sys.argv.index("--keyframe")
        kw, kh = float(sys.argv[i + 1]), float(sys.argv[i + 2])
        s = max(W / kw, H / kh)
        ox, oy = (kw * s - W) / 2, (kh * s - H) / 2
        cx, cy, height = cx * s - ox, cy * s - oy, height * s
    path = smooth(track(frames, cx, cy, height))
    logo = Image.open(logo_path).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    rgba = cv2.cvtColor(np.asarray(logo), cv2.COLOR_RGBA2BGRA)
    r = int(height)
    ring_ref = float(cv2.cvtColor(frames[0][int(cy - r):int(cy + r), int(cx - r):int(cx + r)], cv2.COLOR_BGR2GRAY).mean())
    comp = composite(frames, path, rgba, height, ring_ref)
    # Encode losslessly-ish; keep the source's audio if it has any.
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-i", video, "-map", "0:v", "-map", "1:a?",
           "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in comp:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()
    drift = path[-1] - path[0]
    print(f"{Path(video).name}: {len(frames)} frames, logo moved {drift[0]:.0f},{drift[1]:.0f}px, "
          f"scale x{path[-1][2]:.2f}, angle {path[-1][3]:.1f} deg -> {out_path}")


if __name__ == "__main__":
    main()
