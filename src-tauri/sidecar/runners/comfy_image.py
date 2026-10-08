"""Image models on the private ComfyUI engine (keyframes, character designs).

Graphs follow ComfyUI's official templates for each model family. Images are
saved as PNG in OUTPUT_DIR/images with the same job id naming as videos.
"""
import os
import shutil
import time

from model_registry import resolve_asset_path, OUTPUT_DIR, MODELS_DIR
from runners.common import Reporter
from runners.comfy_engine import ComfyEngine, remove_work_dir

IMAGE_DIR = OUTPUT_DIR / "images"


def z_image_graph(rt: dict, prompt: str, width: int, height: int, steps: int, seed: int) -> dict:
    f = rt["files"]
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": f["unet"], "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": rt.get("shift", 3.0)}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": f["text_encoder"], "type": "lumina2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": rt.get("sampler", "res_multistep"),
            "scheduler": rt.get("scheduler", "simple"), "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "pipeline_img"}},
    }


GRAPHS = {"z_image": z_image_graph}


def _run_image_graph(engine: ComfyEngine, graph: dict, report: Reporter, steps: int):
    """Like run_graph, but collects PNG outputs."""
    import json
    import uuid
    import requests
    from websockets.sync.client import connect

    client_id = uuid.uuid4().hex
    with connect(f"ws://127.0.0.1:{engine.port}/ws?clientId={client_id}", max_size=None, open_timeout=30) as ws:
        r = requests.post(f"{engine.base}/prompt", json={"prompt": graph, "client_id": client_id}, timeout=60)
        if not r.ok:
            from runners.comfy_engine import _format_node_errors
            raise RuntimeError(_format_node_errors(r.json()))
        prompt_id = r.json()["prompt_id"]
        while True:
            engine.pump_log()
            try:
                msg = ws.recv(timeout=2)
            except TimeoutError:
                if engine.proc.poll() is not None:
                    raise RuntimeError(f"The engine stopped unexpectedly. {engine.tail(6)}")
                continue
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            kind, body = data.get("type"), data.get("data", {})
            if body.get("prompt_id") not in (None, prompt_id):
                continue
            if kind == "executing" and body.get("node") is None:
                break
            if kind == "execution_error":
                raise RuntimeError(f"{body.get('exception_type', 'Error')}: {body.get('exception_message', '')}"[:800])
            if kind == "progress" and body.get("node") == "sample":
                v, mx = int(body.get("value", 0)), max(1, int(body.get("max", steps)))
                report.status("generating", 15 + 75 * v / mx, message=f"Step {v} of {mx}")
            if kind == "executing" and body.get("node") == "decode":
                report.status("post_processing", 92, message="Finishing the image")
    hist = requests.get(f"{engine.base}/history/{prompt_id}", timeout=30).json().get(prompt_id, {})
    out = []
    for node_out in (hist.get("outputs") or {}).values():
        for it in node_out.get("images", []):
            p = engine.work / "output" / it.get("subfolder", "") / it["filename"]
            if p.exists():
                out.append(p)
    return out


def run(params: dict, model_cfg: dict, report: Reporter) -> str:
    rt = model_cfg["runtime"]
    engine_dir = resolve_asset_path(rt["engine"])
    model_root = resolve_asset_path(rt["model_root"])
    prompt = (params.get("prompt") or "").strip()
    width, height = int(params["width"]) // 64 * 64, int(params["height"]) // 64 * 64
    steps = int(params.get("steps") or 8)
    seed = params.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(os.urandom(6), "little")

    work = MODELS_DIR.parent / "engine-work" / params["job_id"]
    engine = ComfyEngine(engine_dir, model_root, work, rt.get("engine_args", []), report)
    report.status("loading_model", 3, message="Starting the image engine")
    try:
        engine.start()
        t0 = time.time()
        report.log(f"{width}x{height}, {steps} steps, seed {seed}")
        produced = _run_image_graph(engine, GRAPHS[rt["graph"]](rt, prompt, width, height, steps, seed), report, steps)
        if not produced:
            raise RuntimeError(f"The engine finished without an image. {engine.tail()}")
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        out = IMAGE_DIR / f"image_{params['job_id']}.png"
        shutil.move(str(produced[0]), out)
        report.log(f"Saved {out.name} in {time.time() - t0:.0f}s")
        return str(out)
    finally:
        engine.stop()
        remove_work_dir(work)
