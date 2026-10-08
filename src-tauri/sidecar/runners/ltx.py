"""LTX-Video 0.9.x (2B, single-file checkpoint) text-to-video runner.

The 0.9.5 checkpoint carries both transformer and VAE weights; they are built
with the official 0.9.5 diffusers configs (runtime.configs). Building the VAE
from the older repo-root config, or forcing dynamic timestep shifting, decodes
into colourful noise, which is what earlier versions of this app produced.

Memory plan (8GB VRAM / 16GB RAM):
  1. T5-XXL encoder loaded in bf16 (~9.5GB RAM), streamed through the GPU
     leaf by leaf for the prompt + negative prompt, then deleted.
  2. Transformer (~4GB bf16) and VAE (~1GB) go straight to the GPU.
"""
import time

from model_registry import resolve_asset_path, OUTPUT_DIR
from runners.common import (Reporter, free_memory, export_mp4, vram_gb,
                            load_cached_embeds, save_cached_embeds)


def _encode(paths: dict, prompt: str, negative: str, report: Reporter, torch):
    namespace = f"ltx-t5-128:{paths['text_encoder']}"
    cached = load_cached_embeds(namespace, prompt, negative)
    if cached:
        report.log("Prompt embeddings loaded from cache")
        return cached["pe"], cached["pm"], cached["ne"], cached["nm"]

    from transformers import T5EncoderModel, T5TokenizerFast
    from diffusers import LTXPipeline
    from diffusers.hooks import apply_group_offloading

    report.log("Loading text encoder")
    tokenizer = T5TokenizerFast.from_pretrained(paths["tokenizer"])
    text_encoder = T5EncoderModel.from_pretrained(
        paths["text_encoder"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    apply_group_offloading(
        text_encoder, onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"), offload_type="leaf_level",
    )
    text_pipe = LTXPipeline(scheduler=None, vae=None, text_encoder=text_encoder,
                            tokenizer=tokenizer, transformer=None)
    with torch.no_grad():
        pe, pm, ne, nm = text_pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative, do_classifier_free_guidance=True,
            max_sequence_length=128, device=torch.device("cuda"), dtype=torch.bfloat16,
        )
    del text_pipe, text_encoder, tokenizer
    free_memory()
    report.log("Prompt encoded; text encoder freed")
    save_cached_embeds(namespace, {"pe": pe, "pm": pm, "ne": ne, "nm": nm}, prompt, negative)
    return pe, pm, ne, nm


def run(params: dict, model_cfg: dict, report: Reporter) -> str:
    import torch
    from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler,
                           LTXPipeline, LTXVideoTransformer3DModel)

    if not torch.cuda.is_available():
        raise RuntimeError("This model needs an NVIDIA GPU with CUDA.")

    runtime = model_cfg["runtime"]
    paths = {k: str(resolve_asset_path(v)) for k, v in runtime["components"].items()}
    defaults = model_cfg.get("generation", {}).get("defaults", {})

    prompt = (params.get("prompt") or "").strip()
    negative = (params.get("negative_prompt") or "").strip() or defaults.get("negative_prompt", "")
    width, height = int(params["width"]), int(params["height"])
    num_frames, fps = int(params["num_frames"]), int(params["fps"])
    steps, guidance = int(params["steps"]), float(params["cfg_scale"])
    seed = params.get("seed")
    if seed is None or int(seed) < 0:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    seed = int(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    report.status("loading_model", 2, message="Reading your prompt")
    pe, pm, ne, nm = _encode(paths, prompt, negative, report, torch)

    report.status("loading_model", 12, message="Loading video model")
    report.log("Loading transformer + VAE from checkpoint")
    transformer = LTXVideoTransformer3DModel.from_single_file(
        paths["checkpoint"], config=paths["configs"], subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    # The VAE waits in system RAM: every GB of VRAM matters during denoising,
    # since overflowing makes Windows spill to RAM and run ~6x slower.
    vae = AutoencoderKLLTXVideo.from_single_file(
        paths["checkpoint"], config=paths["configs"], subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    # Tile in space *and* time: decoding 121 frames at 480p in one go needs more
    # than 8GB and Windows then silently spills into system RAM (GPU at 0%).
    vae.enable_tiling()
    vae.use_framewise_decoding = True
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(paths["configs"], subfolder="scheduler")
    # No VAE on the pipeline: diffusers would run everything on the VAE's device
    # (system RAM). Its defaults (32x spatial, 8x temporal) match LTX 0.9.x.
    pipe = LTXPipeline(scheduler=scheduler, vae=None, text_encoder=None, tokenizer=None, transformer=transformer)
    pipe.set_progress_bar_config(disable=True)
    free_memory()

    step_times: list[float] = []
    last = [time.time()]

    def on_step(_pipe, i, _t, kwargs):
        now = time.time()
        step_times.append(now - last[0])
        last[0] = now
        avg = sum(step_times[-5:]) / len(step_times[-5:])
        remaining = steps - i - 1
        report.status("generating", 15 + 70 * (i + 1) / steps, eta=int(remaining * avg + 20),
                      message=f"Step {i + 1} of {steps}")
        report.log(f"Denoising step {i + 1}/{steps} took {step_times[-1]:.1f}s")
        if i == 0:
            report.log(f"VRAM free during denoising: {vram_gb()[0]:.1f} GB")
        return kwargs

    report.log(f"Generating {width}x{height}x{num_frames} steps={steps} cfg={guidance} seed={seed} "
               f"(VRAM free {vram_gb()[0]:.1f} GB)")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        packed = pipe(
            prompt_embeds=pe, prompt_attention_mask=pm,
            negative_prompt_embeds=ne, negative_prompt_attention_mask=nm,
            width=width, height=height, num_frames=num_frames, frame_rate=fps,
            num_inference_steps=steps, guidance_scale=guidance,
            generator=generator, output_type="latent", callback_on_step_end=on_step,
        ).frames
    report.log(f"Denoised in {sum(step_times):.0f}s ({sum(step_times) / max(1, len(step_times)):.1f}s/step)")

    # Swap: transformer out, VAE in, then decode exactly as LTXPipeline does.
    transformer.to("cpu")
    free_memory()
    vae.to("cuda")
    report.status("post_processing", 88, eta=30, message="Decoding frames")
    report.log(f"VRAM free for decoding: {vram_gb()[0]:.1f} GB")
    with torch.no_grad():
        lat = pipe._unpack_latents(
            packed, (num_frames - 1) // 8 + 1, height // 32, width // 32,
            pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size,
        )
        lat = pipe._denormalize_latents(lat, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
        lat = lat.to(torch.bfloat16)
        timestep = None
        if vae.config.timestep_conditioning:
            dt = runtime.get("decode_timestep", 0.05)
            ns = runtime.get("decode_noise_scale", 0.025)
            noise = torch.randn(lat.shape, generator=generator, device="cuda", dtype=lat.dtype)
            lat = (1 - ns) * lat + ns * noise
            timestep = torch.tensor([dt], device="cuda", dtype=lat.dtype)
        video = vae.decode(lat.to(vae.dtype), timestep, return_dict=False)[0]
        frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]

    del pipe, transformer, vae
    free_memory()

    report.status("post_processing", 95, eta=3, message="Saving video")
    out = OUTPUT_DIR / f"video_{params['job_id']}.mp4"
    export_mp4(frames, out, fps=fps)
    report.log(f"Saved {out} ({out.stat().st_size / 1e6:.1f}MB)")
    return str(out)
