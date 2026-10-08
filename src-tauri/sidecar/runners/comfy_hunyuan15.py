"""HunyuanVideo 1.5 HD: fast 480p generation, then native 720p super-resolution.

Two base modes, picked by the model JSON (runtime.mode):
- "i2v": animate a starting image with the 8-step step-distilled model. This is
  the production path: the picture's look and the character's face carry into
  the video, so shots stay realistic and consistent.
- "t2v": straight from text with the CFG-distilled model plus the lightx2v
  4-step LoRA, for quick drafts.

The optional second stage mirrors ComfyUI's official HunyuanVideo 1.5
super-resolution workflow with Tencent's recommended settings (480p->720p:
shift 2, 6 steps): a learned latent upsampler takes the latent to 720p, then the
distilled SR model re-renders it. Detail comes from a video model, so it stays
consistent from frame to frame (per-frame upscalers shimmer).
"""
import os
import shutil
import time
from pathlib import Path

from model_registry import resolve_asset_path, OUTPUT_DIR, MODELS_DIR
from runners.common import Reporter
from runners.comfy_engine import ComfyEngine, remove_work_dir, run_graph

SR_SIZE = {"landscape": (1280, 720), "portrait": (720, 1280), "square": (960, 960)}


def frames_for(n: int) -> int:
    """The video VAE packs 4 frames per latent frame (+1)."""
    n = max(33, min(121, n))
    return (n - 1) // 4 * 4 + 1


def build_graph(rt: dict, prompt: str, width: int, height: int, frames: int, steps: int, seed: int,
                super_resolution: bool, start_image: str | None = None) -> dict:
    """start_image is a file name in the engine's input folder (image-to-video)."""
    f = rt["files"]
    i2v = rt.get("mode") == "i2v"
    if i2v and not start_image:
        raise RuntimeError("This model animates an image. Pick a starting image first.")
    g = {
        "clip": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": f["text_encoder"], "clip_name2": f["glyph_encoder"], "type": "hunyuan_video_15", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": ""}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": f["base"], "weight_dtype": "default"}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    model = ["unet", 0]
    if f.get("lora"):
        g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": f["lora"], "strength_model": 1.0}}
        model = ["lora", 0]
    g["shift"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": rt.get("base_shift", 5.0)}}

    if i2v:
        g.update({
            "image": {"class_type": "LoadImage", "inputs": {"image": start_image}},
            "cv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": f["clip_vision"]}},
            "cv_enc": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["cv", 0], "image": ["image", 0], "crop": "center"}},
            "cond": {"class_type": "HunyuanVideo15ImageToVideo", "inputs": {
                "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0], "width": width, "height": height,
                "length": frames, "batch_size": 1, "start_image": ["image", 0], "clip_vision_output": ["cv_enc", 0]}},
        })
        pos, neg, latent = ["cond", 0], ["cond", 1], ["cond", 2]
    else:
        g["latent"] = {"class_type": "EmptyHunyuanVideo15Latent", "inputs": {
            "width": width, "height": height, "length": frames, "batch_size": 1}}
        pos, neg, latent = ["pos", 0], ["neg", 0], ["latent", 0]

    g.update({
        "sched": {"class_type": "BasicScheduler", "inputs": {"model": ["shift", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "guider": {"class_type": "CFGGuider", "inputs": {"model": ["shift", 0], "positive": pos, "negative": neg, "cfg": 1.0}},
        "base": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0], "sigmas": ["sched", 0],
            "latent_image": latent}},
    })
    final = ["base", 0]

    if super_resolution:
        sr_w, sr_h = SR_SIZE["portrait" if height > width else "square" if height == width else "landscape"]
        sr_steps = rt.get("sr_steps", 6)
        g.update({
            "up_model": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": f["upsampler"]}},
            "upscaled": {"class_type": "HunyuanVideo15LatentUpscaleWithModel", "inputs": {
                "model": ["up_model", 0], "samples": ["base", 0], "upscale_method": "bilinear",
                "width": sr_w, "height": sr_h, "crop": "disabled"}},
            "sr_unet": {"class_type": "UNETLoader", "inputs": {"unet_name": f["sr"], "weight_dtype": "default"}},
            "sr_shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["sr_unet", 0], "shift": rt.get("sr_shift", 2.0)}},
            "sr_cond": {"class_type": "HunyuanVideo15SuperResolution", "inputs": {
                "positive": ["pos", 0], "negative": ["neg", 0], "latent": ["upscaled", 0],
                "noise_augmentation": rt.get("sr_noise_augmentation", 0.7),
                **({"vae": ["vae", 0], "start_image": ["image", 0], "clip_vision_output": ["cv_enc", 0]} if i2v else {})}},
            "sr_sched": {"class_type": "BasicScheduler", "inputs": {
                "model": ["sr_unet", 0], "scheduler": "simple", "steps": sr_steps, "denoise": 1.0}},
            "sr_split": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sr_sched", 0], "step": rt.get("sr_split", sr_steps // 2)}},
            "sr_noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1}},
            "sr_guider_a": {"class_type": "CFGGuider", "inputs": {
                "model": ["sr_shift", 0], "positive": ["sr_cond", 0], "negative": ["sr_cond", 1], "cfg": 1.0}},
            "sr_a": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["sr_noise", 0], "guider": ["sr_guider_a", 0], "sampler": ["sampler", 0],
                "sigmas": ["sr_split", 0], "latent_image": ["sr_cond", 2]}},
            "no_noise": {"class_type": "DisableNoise", "inputs": {}},
            "sr_guider_b": {"class_type": "CFGGuider", "inputs": {
                "model": ["sr_unet", 0], "positive": ["pos", 0], "negative": ["neg", 0], "cfg": 1.0}},
            "sr_b": {"class_type": "SamplerCustomAdvanced", "inputs": {
                "noise": ["no_noise", 0], "guider": ["sr_guider_b", 0], "sampler": ["sampler", 0],
                "sigmas": ["sr_split", 1], "latent_image": ["sr_a", 0]}},
        })
        final = ["sr_b", 0]

    g.update({
        "decode": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": final, "vae": ["vae", 0], "tile_size": 512, "overlap": 64, "temporal_size": 32, "temporal_overlap": 4}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": 24.0}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["video", 0], "filename_prefix": "pipeline_hy15", "format": "mp4", "format.codec": "h264"}},
    })
    return g


def run(params: dict, model_cfg: dict, report: Reporter) -> str:
    rt = model_cfg["runtime"]
    engine_dir = resolve_asset_path(rt["engine"])
    model_root = resolve_asset_path(rt["model_root"])
    common = resolve_asset_path(rt["common_root"])
    upsampler_dir = resolve_asset_path(rt["upsampler_root"])
    prompt = (params.get("prompt") or "").strip()
    width, height = int(params["width"]) // 16 * 16, int(params["height"]) // 16 * 16
    frames = frames_for(int(params["num_frames"]))
    steps = int(params.get("steps") or 8)
    profile = (model_cfg.get("profiles") or {}).get(params.get("profile") or "", {})
    sr = bool(params.get("super_resolution", profile.get("super_resolution", True)))
    seed = params.get("seed")
    seed = int(seed) if seed is not None and int(seed) >= 0 else int.from_bytes(os.urandom(6), "little")
    image_src = params.get("start_image")

    work = MODELS_DIR.parent / "engine-work" / params["job_id"]
    engine = ComfyEngine(engine_dir, model_root, work, rt.get("engine_args", []), report,
                         folders=("diffusion_models", "loras", "clip_vision"),
                         extra_folders={"text_encoders": common / "text_encoders", "vae": common / "vae",
                                        "diffusion_models": common / "diffusion_models",
                                        "latent_upscale_models": upsampler_dir})
    report.status("loading_model", 2, message="Starting the video engine")
    try:
        engine.start()
        start_name = None
        if image_src:
            src = Path(image_src)
            if not src.is_file():
                raise RuntimeError("The starting image is missing.")
            start_name = "start" + src.suffix.lower()
            shutil.copy2(src, engine.work / "input" / start_name)
        report.log(f"{width}x{height}x{frames}, {steps} steps, {'from image' if start_name else 'from text'}, "
                   f"super-resolution {'720p' if sr else 'off'}, seed {seed}")
        t0 = time.time()
        sr_total = rt.get("sr_steps", 6)
        sr_split = rt.get("sr_split", sr_total // 2)
        stages = {"pos": ("loading_model", 6, "Reading your prompt"),
                  "cond": ("loading_model", 9, "Reading the starting image"),
                  "base": ("generating", 12, "Generating at 480p"),
                  "upscaled": ("generating", 55, "Preparing 720p"),
                  "sr_a": ("generating", 58, "Rebuilding detail at 720p"),
                  "decode": ("post_processing", 90, "Decoding video"),
                  "save": ("post_processing", 97, "Saving video")}

        def on_event(kind, body):
            node = body.get("node")
            if kind == "executing" and node in stages:
                st, pct, msg = stages[node]
                report.status(st, pct, message=msg)
                report.log(f"{msg} ({time.time() - t0:.0f}s)")
            elif kind == "progress" and node in ("base", "sr_a", "sr_b"):
                v, mx = int(body.get("value", 0)), max(1, int(body.get("max", 1)))
                if node == "base":
                    lo, hi = 12, (50 if sr else 85)
                    pct, label = lo + (hi - lo) * v / mx, f"Generating at 480p: step {v} of {mx}"
                else:
                    done = v + (sr_split if node == "sr_b" else 0)
                    pct, label = 58 + 30 * done / sr_total, f"Rebuilding detail at 720p: step {done} of {sr_total}"
                report.status("generating", pct, message=label)
                report.log(f"{label} ({time.time() - t0:.0f}s)")

        graph = build_graph(rt, prompt, width, height, frames, steps, seed, sr, start_name)
        produced = run_graph(engine, graph, report, on_event)
        if not produced:
            raise RuntimeError(f"The engine finished without a video. {engine.tail()}")
        out = OUTPUT_DIR / f"video_{params['job_id']}.mp4"
        shutil.move(str(produced[0]), out)
        report.log(f"Saved {out} ({out.stat().st_size / 1e6:.1f}MB) in {time.time() - t0:.0f}s")
        return str(out)
    finally:
        engine.stop()
        remove_work_dir(work)
