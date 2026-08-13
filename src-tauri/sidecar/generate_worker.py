import sys
import os
import json
import gc
import traceback
import inspect
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

# Force UTF-8 on stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress benign experimental/deprecated warnings
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.preferred_linalg_library is an experimental feature.*")
warnings.filterwarnings("ignore", message=".*The usage of MAGMA backend for linear algebra operations is deprecated.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*on_event is deprecated.*")

sys.path.insert(0, str(Path(__file__).parent))
from model_registry import (
    MODEL_CONFIG,
    SHARED_RESOURCES,
    MODELS_DIR,
    OUTPUT_DIR,
    get_model,
    resolve_generation_params,
    get_scheduler_config,
    get_quantization_config,
    get_validation_config,
)

print("WORKER_V4_STARTED", flush=True)


def emit(payload: dict) -> None:
    """Emit JSON status event to stdout for parent sidecar process."""
    print(json.dumps(payload), flush=True)


def _parse_version(value: str) -> tuple[int, ...]:
    parts = []
    for token in str(value).replace(".dev", ".").replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
    return tuple(parts)


def _load_model_index(model_path: Path) -> dict:
    index_path = model_path / "model_index.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_runtime_alignment(model_cfg: dict, model_index: dict, diffusers_version: str) -> None:
    runtime = model_cfg.get("runtime", {})
    expected_pipeline = runtime.get("pipeline_class")
    native_pipeline = model_index.get("_class_name")
    if native_pipeline and expected_pipeline and native_pipeline != expected_pipeline:
        raise RuntimeError(
            f"Downloaded model declares pipeline '{native_pipeline}', but registry expects "
            f"'{expected_pipeline}'. Update this model JSON before running generation."
        )

    min_version = runtime.get("min_diffusers_version")
    expected_version = runtime.get("expected_diffusers_version")
    expected_index_version = runtime.get("expected_model_index_diffusers_version")
    if min_version and _parse_version(diffusers_version) < _parse_version(min_version):
        raise RuntimeError(
            f"{expected_pipeline} needs diffusers >= {min_version}; installed version is "
            f"{diffusers_version}. Reinstall the sidecar requirements before generating."
        )
    if expected_version and expected_version != diffusers_version:
        print(
            f"[runtime_alignment] Model metadata expects diffusers {expected_version}; "
            f"installed version is {diffusers_version}",
            flush=True,
        )
    if expected_index_version and model_index.get("_diffusers_version") != expected_index_version:
        print(
            f"[runtime_alignment] Downloaded model_index was built with diffusers "
            f"{model_index.get('_diffusers_version')}; registry notes {expected_index_version}",
            flush=True,
        )


def get_free_vram_gb() -> float:
    """Utility to query free CUDA VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
    except Exception:
        pass
    return 0.0


# ═════════════════════════════════════════════════════════════════════
# FUNCTION REGISTRY: Each function is self-contained, reads from ctx
# ═════════════════════════════════════════════════════════════════════

class PipelineContext:
    """Mutable context passed through the pipeline."""
    def __init__(self):
        self.job_id = None
        self.model_id = None
        self.model_cfg = None
        self.raw_params = None
        self.resolved_params = None
        self.pipe = None
        self.video_output = None
        self.generation_args = None
        self.offload_strategy = None
        self.step_callback = None


# ── FUNCTION: Load Job ───────────────────────────────────────────────

def fn_load_job(ctx: PipelineContext, config: dict) -> None:
    """Load job file, validate input parameters, resolve VMR defaults."""
    params_path = Path(config.get("params_path"))
    
    if not params_path.exists():
        raise FileNotFoundError(f"Job parameter file not found: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        raw_params = json.load(f)

    job_id = raw_params.get("job_id")
    model_id = raw_params.get("model_id")

    if not job_id or not model_id:
        raise ValueError(f"Invalid job params: missing job_id or model_id")

    if model_id not in MODEL_CONFIG:
        raise ValueError(f"Unknown model_id: {model_id}")

    model_cfg = MODEL_CONFIG[model_id]
    resolved_params = resolve_generation_params(model_id, raw_params)

    # Apply model-level generation defaults
    generation_defaults = model_cfg.get("generation", {}).get("defaults", {})
    for key, default_value in generation_defaults.items():
        if resolved_params.get(key) is None and default_value is not None:
            resolved_params[key] = default_value
            print(f"[fn_load_job] Applied default: {key}={default_value}", flush=True)

    # Inject prompt fields
    resolved_params["prompt"] = raw_params.get("prompt", "")
    resolved_params["negative_prompt"] = raw_params.get("negative_prompt", "")

    # Store in context
    ctx.job_id = job_id
    ctx.model_id = model_id
    ctx.model_cfg = model_cfg
    ctx.raw_params = raw_params
    ctx.resolved_params = resolved_params

    print(
        f"[fn_load_job] Job loaded: {job_id} | Model: {model_id} | "
        f"steps={resolved_params.get('steps')}, "
        f"cfg={resolved_params.get('cfg_scale')}, "
        f"dims={resolved_params.get('width')}x{resolved_params.get('height')}",
        flush=True
    )


# ── FUNCTION: Load Pipeline ──────────────────────────────────────────

def fn_load_pipeline(ctx: PipelineContext, config: dict) -> None:
    """Instantiate PyTorch/Diffusers pipeline from disk."""
    import torch
    import diffusers

    model_cfg = ctx.model_cfg
    model_id = ctx.model_id
    runtime = model_cfg["runtime"]
    dtype_str = runtime.get("dtype", "float16")
    dtype = getattr(torch, dtype_str, torch.float16)
    pipeline_cls_name = runtime["pipeline_class"]
    load_kwargs = dict(runtime.get("load_kwargs", {}))

    if not hasattr(diffusers, pipeline_cls_name):
        raise AttributeError(f"Diffusers has no pipeline class '{pipeline_cls_name}'")

    pipeline_cls = getattr(diffusers, pipeline_cls_name)

    if runtime.get("load_mode") == "from_single_file":
        transformer_cls_name = runtime.get("transformer_class")
        transformer_file = runtime.get("transformer_file")
        if not transformer_cls_name or not transformer_file:
            raise ValueError(f"Model {model_id} requires transformer_class and transformer_file")

        transformer_cls = getattr(diffusers, transformer_cls_name)
        transformer_path = MODELS_DIR / model_id / transformer_file
        print(f"[fn_load_pipeline] Loading single-file transformer: {transformer_file}", flush=True)

        transformer = transformer_cls.from_single_file(
            str(transformer_path),
            torch_dtype=dtype,
            local_files_only=True,
        )

        shared_key = model_cfg.get("distribution", {}).get("shared_resources")
        if not shared_key or shared_key not in SHARED_RESOURCES:
            raise ValueError(f"Model {model_id} requires valid shared_resources definition")

        shared = SHARED_RESOURCES[shared_key]
        shared_path = MODELS_DIR / shared["local_dir"]
        print(f"[fn_load_pipeline] Loading shared components from: {shared['local_dir']}", flush=True)

        model_index = shared_path / "model_index.json"
        if not model_index.exists():
            print(f"[fn_load_pipeline] Shared components missing. Downloading from {shared['repo']}...", flush=True)
            from huggingface_hub import snapshot_download
            shared_path.mkdir(parents=True, exist_ok=True)
            # Note: local_dir_use_symlinks is deprecated in huggingface_hub, omitted
            snapshot_download(
                shared["repo"],
                revision=shared.get("revision", "main"),
                local_dir=str(shared_path),
                local_files_only=False,
            )

        model_index = _load_model_index(shared_path)
        _check_runtime_alignment(model_cfg, model_index, diffusers.__version__)

        pipe = pipeline_cls.from_pretrained(
            shared_path.as_posix(),
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
    else:
        model_path = MODELS_DIR / model_id
        print(f"[fn_load_pipeline] Loading from_pretrained: {model_path}", flush=True)
        model_index = _load_model_index(model_path)
        _check_runtime_alignment(model_cfg, model_index, diffusers.__version__)
        pipe = pipeline_cls.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )

    ctx.pipe = pipe
    print(
        f"[fn_load_pipeline] Pipeline initialized: {pipeline_cls_name} "
        f"(diffusers {diffusers.__version__})",
        flush=True,
    )


# ── FUNCTION: Apply Quantization ────────────────────────────────────

def fn_quantize(ctx: PipelineContext, config: dict) -> None:
    """Apply quantization (FP16, BF16, FP8, Quanto, BNB)."""
    import torch

    pipe = ctx.pipe
    target_quant = config.get("quantization") or ctx.resolved_params.get("quantization") or "none"

    print(f"[fn_quantize] Applying quantization: '{target_quant}'", flush=True)

    if target_quant in ("none", "auto"):
        return

    try:
        if target_quant == "fp8":
            if hasattr(pipe, "transformer") and hasattr(torch, "float8_e4m3fn"):
                pipe.transformer.to(torch.float8_e4m3fn)
                print("[fn_quantize] Transformer cast to float8_e4m3fn", flush=True)

        elif target_quant in ("quanto_int8", "quanto_fp8"):
            try:
                from optimum.quanto import quantize, qint8, qfloat8
                qtype = qint8 if "int8" in target_quant else qfloat8
                target_component = getattr(pipe, "transformer", getattr(pipe, "unet", None))
                if target_component is not None:
                    quantize(target_component, weights=qtype)
                    print(f"[fn_quantize] Applied Quanto: {target_quant}", flush=True)
            except ImportError:
                print("[fn_quantize] [WARNING] optimum.quanto not installed. Skipping.", flush=True)

        elif target_quant == "bnb_int8":
            try:
                import bitsandbytes as bnb
                print("[fn_quantize] bitsandbytes detected for int8 quantization.", flush=True)
            except ImportError:
                print("[fn_quantize] [WARNING] bitsandbytes not installed. Skipping.", flush=True)

    except Exception as e:
        print(f"[fn_quantize] [WARNING] Issue during quantization: {e}. Continuing.", flush=True)


# ── FUNCTION: Configure Scheduler ───────────────────────────────────

def fn_configure_scheduler(ctx: PipelineContext, config: dict) -> None:
    """Dynamically load and configure scheduler (DDIM, Euler, FlowMatch, etc.)."""
    import diffusers

    pipe = ctx.pipe
    model_cfg = ctx.model_cfg

    # Special case: LTX has built-in scheduler, don't override
    if model_cfg["runtime"].get("pipeline_class") == "LTXPipeline":
        print("[fn_configure_scheduler] LTXPipeline detected - using default scheduler", flush=True)
        return

    target_sched_name = config.get("scheduler") or ctx.resolved_params.get("scheduler")
    sched_cfg = get_scheduler_config(ctx.model_id)
    class_name = sched_cfg.get("class_name")

    if not target_sched_name or not class_name or target_sched_name == "Default":
        print("[fn_configure_scheduler] Using pipeline default scheduler", flush=True)
        return

    print(f"[fn_configure_scheduler] Configuring scheduler: '{target_sched_name}'", flush=True)

    try:
        scheduler_cls = getattr(diffusers, class_name)
        current_config = dict(pipe.scheduler.config)
        new_scheduler = scheduler_cls.from_config(current_config, **sched_cfg.get("params", {}))
        pipe.scheduler = new_scheduler
        print(f"[fn_configure_scheduler] Scheduler set to {type(pipe.scheduler).__name__}", flush=True)
    except Exception as e:
        print(f"[fn_configure_scheduler] [WARNING] Failed to set scheduler: {e}. Keeping default.", flush=True)


# ── FUNCTION: Configure Memory & Offloading ──────────────────────────

def fn_configure_memory(ctx: PipelineContext, config: dict) -> None:
    """Assess VRAM and apply offload strategy (GPU, Model Offload, Sequential, CPU)."""
    import torch
    from model_registry import resolve_offload_strategy

    pipe = ctx.pipe
    model_cfg = ctx.model_cfg
    model_id = ctx.model_id
    free_vram = get_free_vram_gb()

    runtime_constraints = model_cfg.get("runtime_constraints", {})
    forced_strategy = runtime_constraints.get("force_offload_strategy")
    gpu_unsafe_threshold = runtime_constraints.get("gpu_unsafe_at_vram")

    if gpu_unsafe_threshold and free_vram < gpu_unsafe_threshold:
        strategy = forced_strategy or "sequential"
        print(f"[fn_configure_memory] GPU unsafe at {gpu_unsafe_threshold}GB. Current={free_vram:.2f}GB. Using '{strategy}'", flush=True)
    elif forced_strategy:
        strategy = forced_strategy
        print(f"[fn_configure_memory] Forced strategy: '{strategy}'", flush=True)
    else:
        strategy = resolve_offload_strategy(model_id, free_vram)
        print(f"[fn_configure_memory] Auto-resolved strategy: '{strategy}'", flush=True)

    print(f"[fn_configure_memory] Free VRAM: {free_vram:.2f}GB | Strategy: '{strategy}'", flush=True)

    # Log known issues
    known_issues = runtime_constraints.get("known_issues", [])
    if known_issues:
        for issue in known_issues:
            if issue.get("status") == "confirmed":
                print(f"[fn_configure_memory] Known issue: {issue.get('issue')} → {issue.get('workaround')}", flush=True)

    # Set CUDA linalg preference
    try:
        torch.backends.cuda.preferred_linalg_library("magma")
        print("[fn_configure_memory] CUDA linalg set to 'magma'", flush=True)
    except Exception:
        pass

    if not torch.cuda.is_available():
        print("[fn_configure_memory] CUDA unavailable. Running on CPU.", flush=True)
        ctx.offload_strategy = "cpu"
        return

    if strategy == "sequential":
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            print("[fn_configure_memory] Enabled Sequential CPU Offloading", flush=True)
            ctx.offload_strategy = "sequential_offload"
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            print("[fn_configure_memory] Enabled Model CPU Offloading", flush=True)
            ctx.offload_strategy = "model_offload"

    elif strategy == "model":
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            print("[fn_configure_memory] Enabled Model CPU Offloading", flush=True)
            ctx.offload_strategy = "model_offload"
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            print("[fn_configure_memory] Enabled Sequential CPU Offloading", flush=True)
            ctx.offload_strategy = "sequential_offload"

    else:  # strategy == "gpu"
        try:
            pipe = pipe.to("cuda")
            ctx.pipe = pipe
            print("[fn_configure_memory] Model loaded to GPU VRAM", flush=True)
            ctx.offload_strategy = "gpu"
        except torch.cuda.OutOfMemoryError:
            print("[fn_configure_memory] [WARNING] OOM on GPU load. Falling back to Model CPU Offload.", flush=True)
            torch.cuda.empty_cache()
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                ctx.offload_strategy = "model_offload"
            elif hasattr(pipe, "enable_sequential_cpu_offload"):
                pipe.enable_sequential_cpu_offload()
                ctx.offload_strategy = "sequential_offload"


# ── FUNCTION: Configure Optimizations ────────────────────────────────

def fn_optimize(ctx: PipelineContext, config: dict) -> None:
    """Apply VAE Tiling, Slicing, Attention Slicing, FlashAttention, TF32."""
    import torch

    pipe = ctx.pipe
    opts = config.get("optimization", ctx.model_cfg.get("optimization", {}))

    print(f"[fn_optimize] Applying optimizations from config", flush=True)

    # TF32
    if opts.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[fn_optimize] TensorFloat32 (TF32) enabled", flush=True)

    # VAE Tiling
    if opts.get("vae_tiling", True):
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
            print("[fn_optimize] VAE Tiling enabled (pipeline level)", flush=True)
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
            print("[fn_optimize] VAE Tiling enabled (VAE level)", flush=True)

    # VAE Slicing
    if opts.get("vae_slicing", True):
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
            print("[fn_optimize] VAE Slicing enabled (pipeline level)", flush=True)
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
            print("[fn_optimize] VAE Slicing enabled (VAE level)", flush=True)

    # Attention Slicing
    if opts.get("attention_slicing", False):
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            print("[fn_optimize] Attention Slicing enabled", flush=True)

    # xFormers
    if opts.get("xformers", False):
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("[fn_optimize] xFormers memory efficient attention enabled", flush=True)
            except Exception as e:
                print(f"[fn_optimize] [WARNING] xFormers enabling failed: {e}", flush=True)


# ── FUNCTION: Prepare Generation Arguments ──────────────────────────

def fn_prepare_generation_args(ctx: PipelineContext, config: dict) -> None:
    """Build pipeline __call__ kwargs from params and model config."""
    import torch
    import inspect

    pipe = ctx.pipe
    params = ctx.resolved_params
    model_cfg = ctx.model_cfg

    prompt = params.get("prompt", "")
    negative_prompt = params.get("negative_prompt", "")
    steps = params.get("steps", 20)
    cfg_scale = params.get("cfg_scale", 4.0)
    width = params.get("width", 640)
    height = params.get("height", 352)
    num_frames = params.get("num_frames", 49)
    fps = params.get("fps", 16)
    seed = params.get("seed")

    # Handle guidance_rescale (default to 0.0 to prevent destroying CFG quality)
    guidance_rescale = params.get("guidance_rescale")
    if guidance_rescale is None or guidance_rescale == 0.7:
        guidance_rescale = 0.0

    print(
        f"[fn_prepare_generation_args] Building args: "
        f"steps={steps}, cfg={cfg_scale}, "
        f"{width}x{height}, {num_frames} frames @ {fps}fps",
        flush=True
    )

    generation_args = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "width": width,
        "height": height,
        "num_frames": num_frames,
    }
    if guidance_rescale > 0:
        generation_args["guidance_rescale"] = guidance_rescale
    # Add step callback if provided in config
    if config.get("enable_step_callback", True):
        generation_args["callback_on_step_end"] = ctx.step_callback

    # Handle seed
    if seed is not None and int(seed) >= 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(int(seed))
        generation_args["generator"] = generator
        print(f"[fn_prepare_generation_args] Seed: {seed}", flush=True)

    # Remap pipeline-specific parameters
    pipe_signature = inspect.signature(pipe.__call__)
    if "frame_rate" in pipe_signature.parameters and "fps" not in pipe_signature.parameters:
        generation_args["frame_rate"] = fps
        print("[fn_prepare_generation_args] Remapped fps → frame_rate", flush=True)
    else:
        generation_args["fps"] = fps

    # Filter to pipeline signature
    pipe_kwargs = {}
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in pipe_signature.parameters.values()
    )

    for k, v in generation_args.items():
        if k in pipe_signature.parameters or accepts_var_kwargs:
            pipe_kwargs[k] = v

    def supports_arg(arg_name: str) -> bool:
        return arg_name in pipe_signature.parameters or accepts_var_kwargs

    def add_supported_arg(arg_name: str, value, source: str) -> None:
        if supports_arg(arg_name):
            pipe_kwargs[arg_name] = value
            print(f"[fn_prepare_generation_args] Added {source}: {arg_name}={value}", flush=True)
        else:
            print(
                f"[fn_prepare_generation_args] Skipped unsupported {source}: {arg_name}",
                flush=True
            )

    # Inject model-specific defaults
    defaults = model_cfg.get("generation", {}).get("defaults", {})
    arg_aliases = model_cfg.get("generation", {}).get("arg_aliases", {})
    if model_cfg["runtime"].get("pipeline_class") == "LTXPipeline":
        ltx_args = (
            "output_type",
            "decode_timestep",
            "decode_noise_scale",
            "stg_scale",
            "skip_block_list",
        )
        if "output_type" not in defaults:
            defaults = {**defaults, "output_type": "pil"}

        for extra_key in ltx_args:
            if extra_key in defaults and extra_key not in pipe_kwargs:
                value = params.get(extra_key, defaults[extra_key])
                alias_key = arg_aliases.get(extra_key)
                if supports_arg(extra_key):
                    add_supported_arg(extra_key, value, "LTX parameter")
                elif alias_key and supports_arg(alias_key):
                    add_supported_arg(alias_key, value, f"LTX parameter alias for {extra_key}")
                else:
                    add_supported_arg(extra_key, value, "LTX parameter")

    ctx.generation_args = pipe_kwargs

    print("[fn_prepare_generation_args] Generation args prepared", flush=True)
    for key, value in pipe_kwargs.items():
        if key in ("generator", "callback_on_step_end", "prompt", "negative_prompt"):
            print(f"[fn_prepare_generation_args]   {key}=<...>", flush=True)
        else:
            print(f"[fn_prepare_generation_args]   {key}={value}", flush=True)


# ── FUNCTION: Generate Video ────────────────────────────────────────

def fn_generate(ctx: PipelineContext, config: dict) -> None:
    """Execute inference with OOM recovery."""
    import torch
    import time

    pipe = ctx.pipe
    generation_args = ctx.generation_args

    print("[fn_generate] Starting inference...", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        result = pipe(**generation_args)
    except torch.cuda.OutOfMemoryError as oom:
        print("[fn_generate] [WARNING] CUDA OOM. Performing cleanup and retry...", flush=True)

        try:
            if hasattr(pipe, "to"):
                pipe = pipe.to("cpu")
                ctx.pipe = pipe
        except Exception as e:
            print(f"[fn_generate] Could not move to CPU: {e}", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(0.5)

        if hasattr(pipe, "enable_sequential_cpu_offload"):
            try:
                pipe.enable_sequential_cpu_offload()
                print("[fn_generate] Enabled Sequential CPU Offload", flush=True)
            except Exception as e:
                print(f"[fn_generate] Could not enable sequential offload: {e}", flush=True)

        if hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                try:
                    pipe.vae.enable_tiling()
                    print("[fn_generate] Extra VAE tiling enabled", flush=True)
                except Exception:
                    pass

        print("[fn_generate] Retrying generation with CPU offload...", flush=True)
        result = pipe(**generation_args)

    # Extract frames
    if hasattr(result, "frames"):
        video = result.frames[0] if isinstance(result.frames, list) else result.frames
    else:
        video = result

    ctx.video_output = video
    print(f"[fn_generate] Inference complete. Output type: {type(video)}", flush=True)


# ── FUNCTION: Validate Output ───────────────────────────────────────

def fn_validate(ctx: PipelineContext, config: dict) -> None:
    """Validate output frame tensor/array/PIL integrity."""
    import torch
    import numpy as np
    from PIL import Image

    frames = ctx.video_output
    model_id = ctx.model_id
    val_cfg = get_validation_config(model_id)

    expected_count = ctx.resolved_params.get("num_frames", 49)

    print("[fn_validate] Validating output...", flush=True)

    try:
        actual_count = len(frames) if isinstance(frames, (list, tuple)) else 1
        if actual_count == 0:
            raise RuntimeError("Generated 0 frames (generation failed)")

        min_frames = val_cfg.get("min_frames", 8)
        if actual_count < min_frames:
            raise RuntimeError(f"Generated {actual_count} frames < minimum {min_frames}")

        first_frame = frames[0] if isinstance(frames, (list, tuple)) else frames

        if isinstance(first_frame, Image.Image):
            w, h = first_frame.size
            if w < val_cfg.get("min_width", 256) or h < val_cfg.get("min_height", 256):
                raise RuntimeError(f"Frame resolution {w}x{h} below minimum")
            print(f"[fn_validate] Valid: {actual_count} PIL frames ({w}x{h})", flush=True)

        elif isinstance(first_frame, np.ndarray):
            if first_frame.ndim < 2:
                raise RuntimeError(f"Invalid numpy shape: {first_frame.shape}")
            h, w = first_frame.shape[:2]
            min_val, max_val = float(np.min(first_frame)), float(np.max(first_frame))
            if max_val <= min_val:
                raise RuntimeError(f"Blank frame detected: [{min_val:.3f}, {max_val:.3f}]")
            print(f"[fn_validate] Valid: {actual_count} numpy frames ({w}x{h}), range=[{min_val:.3f}, {max_val:.3f}]", flush=True)

        elif isinstance(first_frame, torch.Tensor):
            if torch.isnan(first_frame).any() or torch.isinf(first_frame).any():
                raise RuntimeError("Tensor contains NaN or Infinity")
            min_val, max_val = first_frame.min().item(), first_frame.max().item()
            std_val = first_frame.std().item()
            if max_val < 0.01:
                raise RuntimeError(f"Suspiciously low tensor values: max={max_val:.4f}")
            if std_val < val_cfg.get("min_std_dev", 0.005):
                raise RuntimeError(f"Frame std too low: {std_val:.4f} (static image)")
            print(f"[fn_validate] Valid: {actual_count} Tensors, range=[{min_val:.3f}, {max_val:.3f}], std={std_val:.3f}", flush=True)

        else:
            raise RuntimeError(f"Unrecognized frame type: {type(first_frame)}")

    except Exception as e:
        raise RuntimeError(f"[fn_validate] Validation failed: {str(e)}")


# ── FUNCTION: Export Video ──────────────────────────────────────────

def fn_export(ctx: PipelineContext, config: dict) -> None:
    """Convert frames to PIL and write MP4 file."""
    import torch
    import numpy as np
    from PIL import Image
    from diffusers.utils import export_to_video

    frames = ctx.video_output
    fps = ctx.resolved_params.get("fps", 16)
    
    # Format output filename with job_id (handle {job_id} placeholder)
    output_filename = config.get("output_filename", "video_{job_id}.mp4")
    if "{job_id}" in output_filename:
        output_filename = output_filename.format(job_id=ctx.job_id)
    
    output_path = OUTPUT_DIR / output_filename

    print(f"[fn_export] Exporting {len(frames)} frames @ {fps}fps to {output_filename}", flush=True)

    processed_frames = []
    if len(frames) > 0 and isinstance(frames[0], torch.Tensor):
        for frame in frames:
            frame_np = frame.detach().cpu().numpy()
            if frame_np.dtype in (np.float32, np.float64):
                if frame_np.min() < 0:
                    frame_np = (frame_np + 1.0) / 2.0
                if frame_np.max() > 1.0:
                    frame_np = frame_np / (frame_np.max() + 1e-5)
                frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)

            if frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
                frame_np = np.transpose(frame_np, (1, 2, 0))

            processed_frames.append(Image.fromarray(frame_np))
    else:
        processed_frames = frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(processed_frames, str(output_path), fps=fps)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Export failed: {output_path} not created or empty")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[fn_export] Export succeeded: {output_filename} ({size_mb:.2f}MB)", flush=True)
    ctx.video_output_path = str(output_path)


# ── FUNCTION: Cleanup ───────────────────────────────────────────────

def fn_cleanup(ctx: PipelineContext, config: dict) -> None:
    """Purge VRAM and garbage collect."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    print("[fn_cleanup] Cleanup complete", flush=True)


# ═════════════════════════════════════════════════════════════════════
# FUNCTION REGISTRY: Maps function names to implementations
# ═════════════════════════════════════════════════════════════════════

FUNCTION_REGISTRY: Dict[str, Callable] = {
    "load_job": fn_load_job,
    "load_pipeline": fn_load_pipeline,
    "quantize": fn_quantize,
    "configure_scheduler": fn_configure_scheduler,
    "configure_memory": fn_configure_memory,
    "optimize": fn_optimize,
    "prepare_generation_args": fn_prepare_generation_args,
    "generate": fn_generate,
    "validate": fn_validate,
    "export": fn_export,
    "cleanup": fn_cleanup,
}


# ═════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR: Config-driven pipeline executor
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    """Execute pipeline defined in model config, not hardcoded stages."""
    if len(sys.argv) < 2:
        print("Usage: generate_worker_v4.py <params_json_path>", file=sys.stderr)
        sys.exit(1)

    params_path = Path(sys.argv[1])
    ctx = PipelineContext()

    try:
        # Load model config
        with open(params_path, "r") as f:
            raw_params = json.load(f)
        model_id = raw_params.get("model_id")
        if model_id not in MODEL_CONFIG:
            raise ValueError(f"Unknown model_id: {model_id}")

        model_cfg = MODEL_CONFIG[model_id]
        ctx.model_id = model_id
        ctx.model_cfg = model_cfg

        # Read pipeline from model config (or use default)
        pipeline_config = model_cfg.get("_pipeline_config", {})
        if not pipeline_config:
            print("[ORCHESTRATOR] No _pipeline_config in model. Using default pipeline.", flush=True)
            pipeline_config = _get_default_pipeline()

        job_id = raw_params.get("job_id")
        ctx.job_id = job_id

        # Setup step callback
        steps = raw_params.get("steps", 20)
        def step_callback(pipe_obj, step, timestep, kwargs=None):
            progress = 10.0 + (step / max(1, steps)) * 80.0
            emit({
                "type": "job_status",
                "job_id": job_id,
                "status": "generating",
                "progress": round(progress, 1),
                "eta": max(1, int((steps - step) * 1.5)),
                "outputPath": None,
                "error": None
            })
            return kwargs if kwargs is not None else {}

        ctx.step_callback = step_callback

        # Emit initial status
        emit({
            "type": "job_status",
            "job_id": job_id,
            "status": "loading_model",
            "progress": 0.0,
            "eta": 30,
            "outputPath": None,
            "error": None
        })

        # Execute pipeline steps
        print("[ORCHESTRATOR] Starting pipeline execution", flush=True)
        pipeline_steps = pipeline_config.get("steps", [])

        for i, step in enumerate(pipeline_steps):
            fn_name = step.get("fn")
            fn_config = step.get("config", {})
            fn_config["params_path"] = params_path  # Inject params_path for load_job

            if fn_name not in FUNCTION_REGISTRY:
                raise ValueError(f"Unknown function: {fn_name}")

            fn = FUNCTION_REGISTRY[fn_name]
            print(f"\n[ORCHESTRATOR] [{i+1}/{len(pipeline_steps)}] {fn_name}", flush=True)

            try:
                fn(ctx, fn_config)
            except Exception as e:
                print(f"[ORCHESTRATOR] [ERROR] Step '{fn_name}' failed: {e}", flush=True)
                raise

            # Update progress for long-running steps
            if fn_name in ("generate",):
                emit({
                    "type": "job_status",
                    "job_id": job_id,
                    "status": "post_processing",
                    "progress": 90.0,
                    "eta": 5,
                    "outputPath": None,
                    "error": None
                })

        # Final emit
        output_path = getattr(ctx, "video_output_path", None)
        
        # Debug: Log what we're about to emit
        print(f"[ORCHESTRATOR] Final emit: status=done, outputPath={output_path}", flush=True)
        
        # Keep native OS path format for Tauri's convertFileSrc compatibility
        
        emit({
            "type": "job_status",
            "job_id": job_id,
            "status": "done",
            "progress": 100.0,
            "eta": 0,
            "outputPath": output_path,
            "error": None
        })

        print(f"[ORCHESTRATOR] Pipeline completed successfully", flush=True)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        if hasattr(ctx, "job_id") and ctx.job_id:
            emit({
                "type": "job_status",
                "job_id": ctx.job_id,
                "status": "error",
                "progress": 0.0,
                "eta": 0,
                "outputPath": None,
                "error": str(e)
            })
        sys.exit(1)


def _get_default_pipeline() -> dict:
    """Fallback default pipeline (generic, all models)."""
    return {
        "steps": [
            {"fn": "load_job", "config": {}},
            {"fn": "load_pipeline", "config": {}},
            {"fn": "quantize", "config": {}},
            {"fn": "configure_scheduler", "config": {}},
            {"fn": "configure_memory", "config": {}},
            {"fn": "optimize", "config": {}},
            {"fn": "prepare_generation_args", "config": {"enable_step_callback": True}},
            {"fn": "generate", "config": {}},
            {"fn": "validate", "config": {}},
            {"fn": "export", "config": {}},
            {"fn": "cleanup", "config": {}},
        ]
    }


if __name__ == "__main__":
    main()
