"""MiniMax H3 (FastVideo FastH3 8-step) runner, backed by a private ComfyUI.

ComfyUI is the only engine that can run this 33B model on 8GB VRAM / 16GB RAM:
it streams weights from disk (--fast-disk) instead of holding them in RAM. We
start it headless on a random loopback port for this one job (no web UI, no
custom nodes), submit an API-format graph equivalent to ComfyUI's official
"FastVideo FastH3 text to video" template, relay progress, then shut it down so
its memory is free for other models.
"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import requests

from model_registry import resolve_asset_path, OUTPUT_DIR, MODELS_DIR
from runners.common import Reporter
from runners.comfy_engine import ComfyEngine, remove_work_dir, _format_node_errors

SAMPLER_NODE = "sampler"


def snap_frames(n: int) -> int:
    """H3's video VAE decodes 17k+5 frames; round up like the template does."""
    n = max(5, n)
    return n + (5 - n % 17) % 17


def build_graph(rt: dict, prompt: str, width: int, height: int, frames: int, seed: int, steps: int,
                first_frame: str | None = None, last_frame: str | None = None) -> dict:
    """first_frame / last_frame: file names in the engine's input folder (image-to-video)."""
    f = rt["files"]
    sparse = rt.get("sparse_attention", {})
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": f["unet"], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": f["text_encoder"], "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": f["video_vae"]}},
        "audio_vae": {"class_type": "VAELoader", "inputs": {"vae_name": f["audio_vae"]}},
        "shift": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["unet", 0], "shift_video": rt.get("shift_video", 10.0), "shift_audio": rt.get("shift_audio", 3.0)}},
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["shift", 0], "attention": "comfy kitchen attention"}},
        "sparse": {"class_type": "BlockSparseAttention", "inputs": {
            "model": ["attn", 0], "selection": sparse.get("method", "vsa"),
            "selection.keep_percent": sparse.get("keep_percent", 10.0),
            "start_percent": sparse.get("start_percent", 0.2), "end_percent": 1.0,
            "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256,
            "sink_conditioning": "exact_kv_and_rows", "verbose": False}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
            "width": width, "height": height, "length": frames,
            **({"first_frame": ["first", 0]} if first_frame else {}),
            **({"last_frame": ["last", 0]} if last_frame else {})}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["sparse", 0], "conditioning": ["cond", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "ksampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": rt.get("sampler", "res_multistep")}},
        "sigmas": {"class_type": "BasicScheduler", "inputs": {
            "model": ["sparse", 0], "scheduler": rt.get("scheduler", "simple"), "steps": steps, "denoise": 1.0}},
        SAMPLER_NODE: {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ksampler", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["cond", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": [SAMPLER_NODE, 0], "vae": ["vae", 0]}},
        "decode_audio": {"class_type": "VAEDecodeAudio", "inputs": {"samples": [SAMPLER_NODE, 0], "vae": ["audio_vae", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "audio": ["decode_audio", 0], "fps": 24.0}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["video", 0], "filename_prefix": "pipeline_h3", "format": "mp4", "format.codec": "h264"}},
    }
    # Images are cropped to the video's shape so the model sees exactly the frame it must continue.
    for node, name in (("first", first_frame), ("last", last_frame)):
        if name:
            g[node + "_raw"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            g[node] = {"class_type": "ImageScale", "inputs": {"image": [node + "_raw", 0], "upscale_method": "lanczos",
                                                             "width": width, "height": height, "crop": "center"}}
    return g


def run(params: dict, model_cfg: dict, report: Reporter) -> str:
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed

    rt = model_cfg["runtime"]
    engine_dir = resolve_asset_path(rt["engine"])
    model_root = resolve_asset_path(rt["model_root"])
    if not (engine_dir / "main.py").exists():
        raise RuntimeError("The H3 engine isn't installed. Reinstall this model from the Models page.")

    prompt = (params.get("prompt") or "").strip()
    width, height = int(params["width"]), int(params["height"])
    frames = snap_frames(int(params["num_frames"]))
    steps = int(params.get("steps") or 8)
    seed = params.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(os.urandom(6), "little")

    work = MODELS_DIR.parent / "engine-work" / params["job_id"]
    engine = ComfyEngine(engine_dir, model_root, work, rt.get("engine_args", []), report)
    report.status("loading_model", 2, message="Starting the H3 engine")
    try:
        engine.start()
        names = {}
        for key in ("start_image", "end_image"):
            if params.get(key):
                src = Path(params[key])
                if not src.is_file():
                    raise RuntimeError("The starting image is missing.")
                names[key] = f"{key}{src.suffix.lower()}"
                shutil.copy2(src, engine.work / "input" / names[key])
        graph = build_graph(rt, prompt, width, height, frames, seed, steps,
                            names.get("start_image"), names.get("end_image"))
        client_id = uuid.uuid4().hex
        with connect(f"ws://127.0.0.1:{engine.port}/ws?clientId={client_id}", max_size=None,
                     open_timeout=30) as ws:
            r = requests.post(f"{engine.base}/prompt", json={"prompt": graph, "client_id": client_id}, timeout=60)
            if not r.ok:
                try:
                    raise RuntimeError(_format_node_errors(r.json()))
                except ValueError:
                    raise RuntimeError(f"Engine rejected the job: HTTP {r.status_code}")
            prompt_id = r.json()["prompt_id"]
            report.log(f"Queued graph {prompt_id}: {width}x{height}x{frames} seed={seed}")
            report.status("loading_model", 5, message="Loading H3 (streams from disk; first step is slowest)")

            step_t0, sampling_started = None, False
            decodes_seen = 0
            while True:
                engine.pump_log()
                try:
                    msg = ws.recv(timeout=2)
                except TimeoutError:
                    if engine.proc.poll() is not None:
                        raise RuntimeError(f"The H3 engine stopped unexpectedly. {engine.tail(6)}")
                    continue
                if isinstance(msg, bytes):
                    continue  # previews are disabled; ignore any binary frames
                data = json.loads(msg)
                kind, body = data.get("type"), data.get("data", {})
                if body.get("prompt_id") not in (None, prompt_id):
                    continue
                if kind == "executing":
                    node = body.get("node")
                    if node is None:
                        break  # graph finished
                    if node == "clip" or node == "cond":
                        report.status("loading_model", 8, message="Reading your prompt (large text model)")
                    elif node in ("decode", "decode_audio"):
                        # ComfyUI may run either decode first; keep the bar moving forward.
                        decodes_seen += 1
                        what = "video" if node == "decode" else "sound"
                        report.status("post_processing", 86 + 6 * decodes_seen, eta=60 if node == "decode" else 20,
                                      message=f"Decoding {what}")
                    elif node == "save":
                        report.status("post_processing", 97, eta=5, message="Saving video")
                elif kind == "progress" and body.get("node") == SAMPLER_NODE:
                    value, total = int(body.get("value", 0)), max(1, int(body.get("max", steps)))
                    if not sampling_started:
                        sampling_started, step_t0 = True, time.time()
                    elapsed = time.time() - step_t0
                    per_step = elapsed / max(1, value)
                    report.status("generating", 15 + 70 * value / total,
                                  eta=int(per_step * (total - value) + 90),
                                  message=f"Step {value} of {total}")
                    report.log(f"Sampling step {value}/{total} ({elapsed:.0f}s so far)")
                elif kind == "execution_error":
                    raise RuntimeError(f"{body.get('exception_type', 'Error')}: {body.get('exception_message', '')}".strip()[:800])
                elif kind == "execution_interrupted":
                    raise RuntimeError("Generation was interrupted.")

        engine.pump_log()
        hist = requests.get(f"{engine.base}/history/{prompt_id}", timeout=30).json().get(prompt_id, {})
        produced = None
        for node_out in (hist.get("outputs") or {}).values():
            for items in node_out.values():
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict) and str(it.get("filename", "")).endswith(".mp4"):
                            produced = work / "output" / it.get("subfolder", "") / it["filename"]
        if not produced or not produced.exists():
            raise RuntimeError(f"The engine finished without a video. {engine.tail()}")

        out = OUTPUT_DIR / f"video_{params['job_id']}.mp4"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), out)
        report.log(f"Saved {out} ({out.stat().st_size / 1e6:.1f}MB)")
        return str(out)
    except ConnectionClosed:
        raise RuntimeError(f"The H3 engine stopped unexpectedly (often: not enough memory). {engine.tail(6)}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError("Ran out of memory. Try the Fast preset or a shorter clip.") from e
        raise
    finally:
        engine.stop()
        remove_work_dir(work)
