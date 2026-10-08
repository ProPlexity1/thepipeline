"""Video editor: stitch gallery clips into one video with ffmpeg.

Each clip is trimmed, scaled to fit a common canvas (letterboxed if its shape
differs), resampled to a common frame rate, and given a stereo audio track
(silence for clips without sound) so clips from different models and
resolutions can be joined. Clips are joined with hard cuts or crossfades, with
optional fades from and to black. Runs on the CPU, so it never waits behind GPU
jobs.
"""
import re
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

FPS = 24
AUDIO_RATE = 48000
MAX_CLIPS = 100

_probe_cache: dict[tuple[str, float], dict] = {}
_probe_lock = threading.Lock()


def probe(path: Path) -> dict:
    """Duration, size and whether the file has sound (cached by mtime)."""
    key = (str(path), path.stat().st_mtime)
    with _probe_lock:
        if key in _probe_cache:
            return _probe_cache[key]
    import av
    info = {"duration": 0.0, "width": 0, "height": 0, "has_audio": False}
    try:
        with av.open(str(path)) as c:
            v = c.streams.video[0] if c.streams.video else None
            if v is not None:
                info["width"], info["height"] = v.codec_context.width, v.codec_context.height
                if v.duration is not None and v.time_base:
                    info["duration"] = float(v.duration * v.time_base)
                elif v.frames and v.average_rate:
                    info["duration"] = v.frames / float(v.average_rate)
            if not info["duration"] and c.duration:
                info["duration"] = c.duration / 1_000_000
            info["has_audio"] = bool(c.streams.audio)
    except Exception:
        pass
    info["duration"] = round(info["duration"], 3)
    with _probe_lock:
        _probe_cache[key] = info
        if len(_probe_cache) > 2000:
            _probe_cache.pop(next(iter(_probe_cache)))
    return info


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def canvas_size(clips: list[dict], resolution: str) -> tuple[int, int]:
    """Output size: the first clip's shape, at the requested height (or the
    tallest clip's height for 'auto')."""
    first = clips[0]["info"]
    aspect = first["width"] / max(1, first["height"])
    if resolution == "1080p":
        h = 1080
    elif resolution == "720p":
        h = 720
    else:
        h = max(c["info"]["height"] for c in clips)
    return _even(h * aspect), _even(h)


def _atempo(speed: float) -> str:
    """atempo only accepts 0.5-2.0 per stage; chain stages for bigger changes."""
    parts, s = [], speed
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


FONT_CANDIDATES = [r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]


def render_text_png(text: str, height_px: int, color: str, out: Path, background: bool) -> Path:
    """Titles are drawn with Pillow into a transparent PNG and overlaid like a logo.
    (More reliable than ffmpeg drawtext: no font-path or escaping pitfalls.)"""
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for f in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(f, max(8, height_px))
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    lines = text.splitlines() or [""]
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    boxes = [probe.textbbox((0, 0), ln or " ", font=font) for ln in lines]
    w = max(b[2] - b[0] for b in boxes)
    line_h = int(height_px * 1.25)
    pad = int(height_px * 0.45) if background else int(height_px * 0.1)
    img = Image.new("RGBA", (w + pad * 2, line_h * len(lines) + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if background:
        d.rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=int(height_px * 0.3), fill=(0, 0, 0, 150))
    for i, ln in enumerate(lines):
        x = pad + (w - (boxes[i][2] - boxes[i][0])) // 2
        y = pad + i * line_h
        if not background:  # soft shadow keeps text readable on any footage
            d.text((x + 2, y + 2), ln, font=font, fill=(0, 0, 0, 160))
        d.text((x, y), ln, font=font, fill=color)
    img.save(out)
    return out


def build_command(ffmpeg: str, clips: list[dict], out: Path, transition: str, fade: float,
                  fade_edges: bool, size: tuple[int, int], overlays: Optional[list[dict]] = None,
                  workdir: Optional[Path] = None, audio_tracks: Optional[list[dict]] = None) -> tuple[list[str], float]:
    """Returns the ffmpeg command and the expected output duration.

    clips: {path, start, end, info, speed?, volume?}
    overlays: {kind: image|text, path? (image), text?, start, end, x, y (0-1, centre), width (0-1 of canvas,
               images) or size (0-1 of canvas height, text), opacity, color?, background?, fade?}
    """
    W, H = size
    args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
    filters, durations = [], []
    n_inputs = 0
    for i, c in enumerate(clips):
        start, end = c["start"], c["end"]
        src_dur = end - start
        speed = max(0.25, min(4.0, float(c.get("speed") or 1.0)))
        volume = max(0.0, min(2.0, float(c.get("volume", 1.0))))
        dur = src_dur / speed
        durations.append(dur)
        args += ["-ss", f"{start:.3f}", "-t", f"{src_dur:.3f}", "-i", str(c["path"])]
        vi = n_inputs
        n_inputs += 1
        filters.append(
            f"[{vi}:v]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,format=yuv420p,"
            f"trim=duration={src_dur:.3f},setpts=(PTS-STARTPTS)/{speed:.4f},"
            f"fps={FPS}[v{i}]")  # fps last: xfade needs a constant rate
        if c["info"]["has_audio"] and volume > 0:
            filters.append(
                f"[{vi}:a]aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad,atrim=duration={src_dur:.3f},asetpts=PTS-STARTPTS,{_atempo(speed)},"
                f"volume={volume:.3f},atrim=duration={dur:.3f}[a{i}]")
        else:
            args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
            filters.append(f"[{n_inputs}:a]aformat=sample_fmts=fltp:channel_layouts=stereo,"
                           f"asetpts=PTS-STARTPTS[a{i}]")
            n_inputs += 1

    n = len(clips)
    # A crossfade can't be longer than either clip it joins.
    if transition == "fade" and n > 1:
        fade = max(0.1, min(fade, min(durations) / 2))
    if n == 1:
        vlast, alast, total = "v0", "a0", durations[0]
    elif transition == "fade":
        vlast, alast, offset = "v0", "a0", 0.0
        for i in range(1, n):
            offset += durations[i - 1] - fade
            filters.append(f"[{vlast}][v{i}]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f}[vx{i}]")
            filters.append(f"[{alast}][a{i}]acrossfade=d={fade:.3f}:c1=tri:c2=tri[ax{i}]")
            vlast, alast = f"vx{i}", f"ax{i}"
        total = sum(durations) - fade * (n - 1)
    else:
        pairs = "".join(f"[v{i}][a{i}]" for i in range(n))
        filters.append(f"{pairs}concat=n={n}:v=1:a=1[vc][ac]")
        vlast, alast, total = "vc", "ac", sum(durations)

    # Overlay layers (logos, images, titles), in order: later ones draw on top.
    for k, ov in enumerate(overlays or []):
        s = max(0.0, min(total, float(ov.get("start", 0))))
        e = max(s + 0.1, min(total, float(ov.get("end", total))))
        if ov["kind"] == "text":
            px = int(H * max(0.02, min(0.3, float(ov.get("size", 0.07)))))
            png = render_text_png(ov.get("text", ""), px, ov.get("color", "#ffffff"),
                                  (workdir or out.parent) / f"_title_{k}.png", bool(ov.get("background")))
            scale = ""
        else:
            png = ov["path"]
            wpx = _even(W * max(0.02, min(1.0, float(ov.get("width", 0.2)))))
            scale = f"scale={wpx}:-2:flags=lanczos,"
        args += ["-loop", "1", "-t", f"{e:.3f}", "-i", str(png)]
        idx = n_inputs
        n_inputs += 1
        opacity = max(0.0, min(1.0, float(ov.get("opacity", 1.0))))
        fd = min(float(ov.get("fade", 0.3)), (e - s) / 3)
        fades = (f",fade=t=in:st={s:.3f}:d={fd:.3f}:alpha=1,fade=t=out:st={e - fd:.3f}:d={fd:.3f}:alpha=1"
                 if fd > 0.01 else "")
        filters.append(f"[{idx}:v]{scale}format=rgba,colorchannelmixer=aa={opacity:.3f}{fades}[ov{k}]")
        x = f"(W*{float(ov.get('x', 0.5)):.4f})-w/2"
        y = f"(H*{float(ov.get('y', 0.5)):.4f})-h/2"
        filters.append(f"[{vlast}][ov{k}]overlay=x='{x}':y='{y}':enable='between(t,{s:.3f},{e:.3f})':"
                       f"eof_action=pass,format=yuv420p[vo{k}]")
        vlast = f"vo{k}"

    # Extra audio tracks (voice-over, music, ambience): each starts at its time with its own volume,
    # then everything is mixed with the clips' own sound. Music can duck under the voice.
    if audio_tracks:
        mix_inputs = [f"[{alast}]"]
        for j, tr in enumerate(audio_tracks):
            args += ["-i", str(tr["path"])]
            idx = n_inputs
            n_inputs += 1
            delay = int(max(0.0, float(tr.get("start", 0))) * 1000)
            vol = max(0.0, min(2.0, float(tr.get("volume", 1.0))))
            chain = (f"[{idx}:a]aresample={AUDIO_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,"
                     f"volume={vol:.3f},adelay={delay}|{delay},apad,atrim=duration={total:.3f}")
            filters.append(chain + f"[tr{j}]")
            mix_inputs.append(f"[tr{j}]")
        filters.append(f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:normalize=0[amx]")
        alast = "amx"

    if fade_edges:
        edge = min(0.6, total / 4)
        filters.append(f"[{vlast}]fade=t=in:st=0:d={edge:.3f},fade=t=out:st={total - edge:.3f}:d={edge:.3f}[vf]")
        filters.append(f"[{alast}]afade=t=in:st=0:d={edge:.3f},afade=t=out:st={total - edge:.3f}:d={edge:.3f}[af]")
        vlast, alast = "vf", "af"

    args += ["-filter_complex", ";".join(filters), "-map", f"[{vlast}]", "-map", f"[{alast}]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-r", str(FPS),
             "-t", f"{total:.3f}", str(out)]
    return args, total


_OUT_TIME = re.compile(r"^out_time_(?:us|ms)=(\d+)")


def render(clips: list[dict], out: Path, transition: str, fade: float, fade_edges: bool,
           resolution: str, on_progress: Callable[[float, str], None],
           on_proc: Optional[Callable[[subprocess.Popen], None]] = None,
           overlays: Optional[list[dict]] = None, audio_tracks: Optional[list[dict]] = None) -> dict:
    """Render `clips` ({path, start, end, info, speed?, volume?}) plus overlay layers to `out`."""
    import imageio_ffmpeg
    import tempfile
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    size = canvas_size(clips, resolution)
    workdir = Path(tempfile.mkdtemp(prefix="pipeline_edit_"))
    try:
        cmd, total = build_command(ffmpeg, clips, out, transition, fade, fade_edges, size, overlays, workdir, audio_tracks)
        on_progress(1, f"Joining {len(clips)} clip{'s' if len(clips) != 1 else ''} at {size[0]}x{size[1]}")
        tmp = out.with_name(out.stem + ".rendering.mp4")
        cmd[-1] = str(tmp)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace",
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if on_proc:
            on_proc(proc)
        err_lines: list[str] = []

        def drain():
            for line in proc.stderr:
                err_lines.append(line.rstrip())
                del err_lines[:-20]
        threading.Thread(target=drain, daemon=True).start()

        last = -1.0
        for line in proc.stdout:
            m = _OUT_TIME.match(line.strip())
            if m:
                secs = int(m.group(1)) / 1_000_000
                pct = min(99.0, 2 + 97 * secs / max(0.1, total))
                if pct - last >= 1:
                    last = pct
                    on_progress(pct, f"Rendering {secs:.1f}s of {total:.1f}s")
        code = proc.wait()
        if code != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            detail = " ".join(err_lines[-3:]) or f"exit code {code}"
            raise RuntimeError(f"Rendering failed: {detail[:400]}")
        tmp.replace(out)
        return {"width": size[0], "height": size[1], "duration": round(total, 2)}
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def make_end_card(lines: list[str], logo: Optional[Path], out_png: Path, size: tuple[int, int] = (1920, 1080),
                  accent: str = "#c49234") -> Path:
    """A clean closing card: logo (if any) above exact text lines. Text is drawn, never generated,
    so phone numbers and addresses are always spelled exactly."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    W, H = size
    bg = Image.new("RGB", (W, H), (14, 14, 16))
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse((W // 2 - W // 4, H // 18, W // 2 + W // 4, int(H * 0.7)), fill=80)
    glow = glow.filter(ImageFilter.GaussianBlur(W // 11))
    ar, ag, ab = (int(accent[i:i + 2], 16) for i in (1, 3, 5))
    bg = Image.composite(Image.new("RGB", (W, H), (ar // 2, ag // 2, ab // 2)), bg, glow)
    d = ImageDraw.Draw(bg)

    def font(bold: bool, px: int):
        for f in (FONT_CANDIDATES if bold else [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"]):
            try:
                return ImageFont.truetype(f, px)
            except OSError:
                continue
        return ImageFont.load_default()

    lines = [l for l in (lines or []) if l.strip()][:5]
    text_h = int(H * 0.09) * len(lines)
    y = int(H * 0.08)
    if logo and Path(logo).is_file():
        im = Image.open(logo).convert("RGBA")
        im = im.crop(im.getbbox() or (0, 0, im.width, im.height))
        lh = int(min(H * 0.5, H - text_h - H * 0.2))
        im = im.resize((max(1, round(im.width * lh / im.height)), lh), Image.LANCZOS)
        bg.paste(im, ((W - im.width) // 2, y), im)
        y += lh + int(H * 0.04)
    else:
        y = (H - text_h) // 2
    for i, line in enumerate(lines):
        big = i == 0 and not logo or (i == 1 and logo is not None)
        f = font(True, int(H * (0.072 if big else 0.036 if i == 0 else 0.042)))
        col = (ar, ag, ab) if (i == 0 and logo is not None) else (245, 245, 245) if big else (215, 215, 215)
        w = d.textlength(line, font=f)
        d.text(((W - w) / 2, y), line, font=f, fill=col)
        y += int(f.size * 1.45)
    bg.save(out_png)
    return out_png


def make_card_clip(png: Path, seconds: float, out: Path, size: tuple[int, int] = (1920, 1080)) -> Path:
    """Still image -> clip with a slow, gentle zoom (and a silent audio track for joining)."""
    import imageio_ffmpeg
    W, H = size
    n = max(2, int(seconds * FPS))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-loop", "1",
                    "-framerate", str(FPS), "-t", f"{seconds:.3f}", "-i", str(png), "-f", "lavfi", "-t", f"{seconds:.3f}",
                    "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo", "-filter_complex",
                    f"[0:v]scale={W * 2}:{H * 2},zoompan=z='1+0.035*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                    f"d={n}:s={W}x{H}:fps={FPS},format=yuv420p[v]", "-map", "[v]", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-c:a", "aac", "-shortest", str(out)],
                   check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out
