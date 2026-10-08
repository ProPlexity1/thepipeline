"""Qwen3-TTS voice design: natural speech in a voice described in words.

Runs in its own environment (voice_venv: Qwen pins transformers 4.57.3) that
shares the main environment's PyTorch. One process per request; the whole text is
spoken in one call so the voice, pace and tone stay consistent across lines.

CLI: voice_venv\\Scripts\\python.exe runners\\qwen_tts_runner.py <params.json>
params: {model_dir, text, instruct, language, out, seed}
prints JSON {duration, sample_rate}
"""
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np


def main():
    p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    import torch
    from qwen_tts import Qwen3TTSModel

    torch.manual_seed(int(p.get("seed", 0)))
    t0 = time.time()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSModel.from_pretrained(p["model_dir"], device_map=dev,
                                          dtype=torch.bfloat16 if dev != "cpu" else torch.float32,
                                          attn_implementation="sdpa")
    load = time.time() - t0
    wavs, sr = model.generate_voice_design(text=p["text"], instruct=p.get("instruct", ""),
                                           language=p.get("language", "English"))
    audio = np.asarray(wavs[0], dtype=np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = audio / peak * 0.89  # consistent level, a little headroom
    out = Path(p["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    print(json.dumps({"duration": round(len(audio) / sr, 2), "sample_rate": int(sr),
                      "load_seconds": round(load, 1), "total_seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
