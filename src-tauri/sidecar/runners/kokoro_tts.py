"""Kokoro text-to-speech (82M). Runs on the CPU so voice-overs never wait behind
GPU jobs; it's still faster than real time.

CLI (used by the engine, one short-lived process per request):
  python -m runners.kokoro_tts <params.json>
params: {text, voice, speed, out}  ->  writes a 24 kHz WAV and prints JSON {duration}
"""
import json
import re
import sys
from pathlib import Path

SAMPLE_RATE = 24000

# Voice id prefix -> Kokoro language code
LANGS = {"a": "a", "b": "b", "e": "e", "f": "f", "h": "h", "i": "i", "j": "j", "p": "p", "z": "z"}
LANG_NAMES = {"a": "American English", "b": "British English", "e": "Spanish", "f": "French", "h": "Hindi",
              "i": "Italian", "j": "Japanese", "p": "Portuguese", "z": "Chinese"}


def model_dir() -> Path:
    from model_registry import MODEL_CONFIG, resolve_asset_path
    return resolve_asset_path(MODEL_CONFIG["kokoro-82m"]["runtime"]["model_root"])


def list_voices() -> list[dict]:
    d = model_dir() / "voices"
    out = []
    for p in sorted(d.glob("*.pt")):
        vid = p.stem
        out.append({"id": vid, "name": vid.split("_", 1)[1].title(), "language": LANG_NAMES.get(vid[0], vid[0]),
                    "gender": "female" if vid[1] == "f" else "male"})
    return out


_pipelines: dict = {}


def synth(text: str, voice: str, speed: float, out: Path) -> float:
    import numpy as np
    import torch
    from kokoro import KModel, KPipeline

    if not re.fullmatch(r"[a-z]{2}_[a-z]+", voice):
        raise ValueError("Unknown voice")
    root = model_dir()
    lang = LANGS.get(voice[0])
    if lang is None:
        raise ValueError("Unsupported voice language")
    key = lang
    if key not in _pipelines:
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(root / "config.json"),
                       model=str(root / "kokoro-v1_0.pth")).eval()
        _pipelines[key] = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model)
    pipe = _pipelines[key]
    pack = pipe.load_voice(str(root / "voices" / f"{voice}.pt"))
    chunks = []
    with torch.no_grad():
        for result in pipe(text, voice=pack, speed=float(speed), split_pattern=r"\n+"):
            if result.audio is not None:
                chunks.append(result.audio.cpu().numpy())
                chunks.append(np.zeros(int(SAMPLE_RATE * 0.25), dtype=np.float32))  # breath between lines
    if not chunks:
        raise RuntimeError("Nothing to say")
    audio = np.concatenate(chunks[:-1])
    out.parent.mkdir(parents=True, exist_ok=True)
    import wave
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return len(audio) / SAMPLE_RATE


if __name__ == "__main__":
    params = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    dur = synth(params["text"], params["voice"], params.get("speed", 1.0), Path(params["out"]))
    print(json.dumps({"duration": round(dur, 2)}), flush=True)
