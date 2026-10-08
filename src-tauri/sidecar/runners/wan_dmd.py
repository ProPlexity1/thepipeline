"""Wan 2.2 TI2V-5B, few-step DMD-distilled (FastWan) text-to-video runner.

Memory plan for 8GB VRAM / 16GB RAM. Loading the whole pipeline at once would
need ~24GB of RAM, so components are staged and freed one at a time:

  1. UMT5 text encoder (11.4GB bf16) in RAM, leaf-level group-offloaded through
     the GPU for a single forward pass, then deleted.
  2. Transformer (10GB bf16) loaded, stored as fp8 (5GB) with bf16 compute on
     the GPU unless the card has room for bf16, run for the DMD steps, deleted.
  3. VAE (fp32) decodes with tiling.

DMD sampling (matches FastVideo's DmdDenoisingStage): at each training timestep
t the model's flow prediction v gives x0 = x_t - sigma_t * v with sigma = t/1000
(nearest entry of the shift-8 training schedule), and x0 is re-noised to the
next timestep with fresh noise. No classifier-free guidance.
"""
import time
from pathlib import Path

from model_registry import resolve_asset_path, OUTPUT_DIR
from runners.common import (Reporter, free_memory, vram_gb, export_mp4,
                            load_cached_embeds, save_cached_embeds)

DEFAULT_DMD_TIMESTEPS = [1000, 757, 522]


def _encode_prompt(paths: dict, prompt: str, report: Reporter, torch):
    namespace = f"wan-umt5-512:{paths['text_encoder']}"
    cached = load_cached_embeds(namespace, prompt)
    if cached:
        report.log("Prompt embeddings loaded from cache")
        return cached["embeds"]

    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers.hooks import apply_group_offloading

    report.log("Loading text encoder")
    tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer"])
    text_encoder = UMT5EncoderModel.from_pretrained(
        paths["text_encoder"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    # Streams each leaf module through the GPU: bf16 precision, ~no VRAM held.
    apply_group_offloading(
        text_encoder, onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"), offload_type="leaf_level",
    )

    max_len = 512
    inputs = tokenizer(
        [prompt], padding="max_length", max_length=max_len, truncation=True,
        add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
    )
    seq_len = int(inputs.attention_mask.gt(0).sum())
    with torch.no_grad():
        embeds = text_encoder(
            inputs.input_ids.to("cuda"), inputs.attention_mask.to("cuda")
        ).last_hidden_state
    # Same trimming + zero padding as WanPipeline._get_t5_prompt_embeds.
    embeds = embeds[0, :seq_len].to(torch.bfloat16)
    embeds = torch.cat([embeds, embeds.new_zeros(max_len - seq_len, embeds.size(1))]).unsqueeze(0)

    del text_encoder, tokenizer
    free_memory()
    report.log(f"Prompt encoded ({seq_len} tokens); text encoder freed")
    embeds = embeds.to("cuda")
    save_cached_embeds(namespace, {"embeds": embeds}, prompt)
    return embeds


# Measured on an RTX 5060: activations grow ~0.19 GB per 1k tokens; fp8 weights
# take ~4.8 GB, bf16 ~9.6 GB. Overshooting VRAM makes Windows spill into system
# RAM, which is ~3x slower than streaming blocks over PCIe on purpose.
ACT_GB_PER_1K_TOKENS = 0.19
FP8_WEIGHTS_GB, BF16_WEIGHTS_GB, MARGIN_GB = 4.8, 9.6, 0.6


def _load_transformer(path: str, tokens: int, report: Reporter, torch):
    from diffusers import WanTransformer3DModel

    report.log("Loading transformer")
    transformer = WanTransformer3DModel.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    free, _ = vram_gb()
    act = ACT_GB_PER_1K_TOKENS * tokens / 1000
    if free >= BF16_WEIGHTS_GB + act + MARGIN_GB:
        transformer.to("cuda")
        mode = "bf16, resident"
    else:
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        if free >= FP8_WEIGHTS_GB + act + MARGIN_GB:
            transformer.to("cuda")
            mode = "fp8, resident"
        else:
            # Weights stay in RAM; blocks stream to the GPU as they run.
            transformer.enable_group_offload(
                onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                offload_type="block_level", num_blocks_per_group=4, use_stream=False,
            )
            mode = "fp8, streamed from RAM"
    report.log(f"Transformer: {mode} ({free:.1f} GB VRAM free, ~{act:.1f} GB needed for activations)")
    free_memory()
    return transformer


def _encode_start_image(image_path: str, vae_path: str, width: int, height: int, z_dim: int, runtime: dict, torch):
    """Resize/crop the picture to the video size and encode it to a normalized latent frame."""
    import numpy as np
    from PIL import Image, ImageOps
    from diffusers import AutoencoderKLWan
    img = ImageOps.fit(Image.open(image_path).convert("RGB"), (width, height), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img).astype("float32") / 127.5 - 1.0).permute(2, 0, 1)[None, :, None]
    vae_dtype = getattr(torch, runtime.get("vae_dtype", "float32"))
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=vae_dtype).eval().to("cuda")
    with torch.no_grad():
        z = vae.encode(x.to("cuda", vae_dtype)).latent_dist.mode().float()
        mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to("cuda")
        std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to("cuda")
        z = (z - mean) / std
    del vae
    free_memory()
    return z


def run(params: dict, model_cfg: dict, report: Reporter) -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This model needs an NVIDIA GPU with CUDA.")

    runtime = model_cfg["runtime"]
    paths = {k: str(resolve_asset_path(v)) for k, v in runtime["components"].items()}
    timesteps = runtime.get("dmd_timesteps", DEFAULT_DMD_TIMESTEPS)

    prompt = (params.get("prompt") or "").strip()
    width, height = int(params["width"]), int(params["height"])
    num_frames, fps = int(params["num_frames"]), int(params["fps"])
    seed = params.get("seed")
    if seed is None or int(seed) < 0:
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    seed = int(seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    report.status("loading_model", 2, message="Reading your prompt")
    prompt_embeds = _encode_prompt(paths, prompt, report, torch)

    # Latent geometry from the Wan2.2 VAE config (16x spatial, 4x temporal, 48 channels).
    import json
    with open(Path(paths["vae"]) / "config.json", "r", encoding="utf-8") as f:
        vae_cfg = json.load(f)
    s_sp = vae_cfg.get("scale_factor_spatial", 16)
    s_t = vae_cfg.get("scale_factor_temporal", 4)
    z_dim = vae_cfg.get("z_dim", 48)
    lat_f = (num_frames - 1) // s_t + 1
    lat_h, lat_w = height // s_sp, width // s_sp
    seq_len = lat_f * (lat_h // 2) * (lat_w // 2)  # transformer patch is (1, 2, 2)

    # Image-to-video (TI2V): the starting picture becomes the first latent frame,
    # held fixed (timestep 0) while the model denoises the rest, as in diffusers'
    # WanImageToVideoPipeline with expand_timesteps.
    cond, mask = None, None
    if params.get("start_image"):
        report.status("loading_model", 8, message="Reading the starting image")
        cond = _encode_start_image(params["start_image"], paths["vae"], width, height, z_dim, runtime, torch)
        mask = torch.ones((1, 1, lat_f, lat_h, lat_w), device="cuda")
        mask[:, :, 0] = 0.0
        report.log("Starting image encoded as the first frame")

    report.status("loading_model", 12, message="Loading video model")
    transformer = _load_transformer(paths["transformer"], seq_len, report, torch)

    gen = torch.Generator(device="cuda").manual_seed(seed)
    latents = torch.randn((1, z_dim, lat_f, lat_h, lat_w), generator=gen,
                          device="cuda", dtype=torch.float32)
    if cond is not None:
        token_mask = mask[0, 0][:, ::2, ::2].flatten()  # one entry per (1, 2, 2) patch

    report.log(f"Denoising {width}x{height}x{num_frames} ({seq_len} tokens), seed={seed}, steps={timesteps}")
    step_times = []
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t0 = time.time()
            # TI2V expands the timestep per token; uniform for text-to-video.
            if cond is not None:
                latents = (1 - mask) * cond + mask * latents
                ts = (token_mask * float(t)).unsqueeze(0)
            else:
                ts = torch.full((1, seq_len), float(t), device="cuda")
            v = transformer(
                hidden_states=latents.to(torch.bfloat16), timestep=ts,
                encoder_hidden_states=prompt_embeds, return_dict=False,
            )[0]
            sigma = t / 1000.0
            x0 = latents.double() - sigma * v.double()
            if i < len(timesteps) - 1:
                s_next = timesteps[i + 1] / 1000.0
                noise = torch.randn(latents.shape, generator=gen, device="cuda", dtype=torch.float32)
                latents = ((1 - s_next) * x0 + s_next * noise.double()).float()
            else:
                latents = x0.float()
            del v, x0
            step_times.append(time.time() - t0)
            remaining = len(timesteps) - i - 1
            avg = sum(step_times) / len(step_times)
            report.status("generating", 20 + 60 * (i + 1) / len(timesteps),
                          eta=int(remaining * avg + 30), message=f"Step {i + 1} of {len(timesteps)}")
            report.log(f"step {i + 1}/{len(timesteps)} t={t} {step_times[-1]:.1f}s "
                       f"(peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB, free {vram_gb()[0]:.1f} GB)")

    if cond is not None:
        latents = (1 - mask) * cond + mask * latents
    if not torch.isfinite(latents).all():
        raise RuntimeError("Generation produced invalid values (NaN). Try a lower resolution.")

    del transformer, prompt_embeds
    free_memory()

    report.status("post_processing", 85, eta=30, message="Decoding frames")
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    vae_dtype = getattr(torch, runtime.get("vae_dtype", "float32"))
    t_dec = time.time()
    vae = AutoencoderKLWan.from_pretrained(paths["vae"], torch_dtype=vae_dtype).eval()
    vae.to("cuda")
    # Bigger tiles overlap less, so they decode faster; use the largest that fits.
    # Measured at 720p: 512px 58s / 4.5GB peak, 384px 67s / 3.4GB, 256px 84s / 3.2GB.
    free = vram_gb()[0]
    tile, stride = (512, 448) if free >= 5.0 else (384, 320) if free >= 4.0 else (256, 192)
    vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile,
                      tile_sample_stride_height=stride, tile_sample_stride_width=stride)
    with torch.no_grad():
        mean = torch.tensor(vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to("cuda")
        std = torch.tensor(vae.config.latents_std).view(1, z_dim, 1, 1, 1).to("cuda")
        video = vae.decode((latents * std + mean).to(vae_dtype), return_dict=False)[0].float()
        frames = VideoProcessor(vae_scale_factor=s_sp).postprocess_video(video, output_type="np")[0]
    del vae, video, latents
    free_memory()
    report.log(f"Decoded in {time.time() - t_dec:.0f}s ({vae_dtype}, {tile}px tiles, {free:.1f} GB free)")

    report.status("post_processing", 95, eta=3, message="Saving video")
    out = OUTPUT_DIR / f"video_{params['job_id']}.mp4"
    export_mp4(frames, out, fps=fps)
    report.log(f"Saved {out} ({out.stat().st_size / 1e6:.1f}MB)")
    return str(out)
