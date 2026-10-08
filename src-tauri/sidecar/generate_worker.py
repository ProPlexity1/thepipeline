import sys
import os
import json
import gc
import time
import platform
import traceback
import inspect
import importlib
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", message=".*torch.backends.cuda.preferred_linalg_library is an experimental feature.*")
warnings.filterwarnings("ignore", message=".*The usage of MAGMA backend for linear algebra operations is deprecated.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*on_event is deprecated.*")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    load_runtime_profile,
    save_runtime_profile,
    save_error_report,
    get_strategy_ladder,
)

# ═════════════════════════════════════════════════════════════════════
# BUILD TAG — bump this whenever you change the worker meaningfully.
# Used for Option C: retry "failed_error" verdicts on new builds.
# ═════════════════════════════════════════════════════════════════════
SIDECAR_BUILD = "worker-v6-diffusers-restore-2026-08-17"



def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


# ═════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════

def _parse_version(value: str) -> tuple[int, ...]:
    parts = []
    for token in str(value).replace(".dev", ".").replace("-", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
    return tuple(parts)


def _resolve_path_directive(directive: str) -> Path:
    if not isinstance(directive, str):
        raise ValueError(f"Path directive must be str, got {type(directive)}")
    if directive.startswith("shared:"):
        rest = directive[len("shared:"):]
        parts = rest.split("/", 1)
        shared_key = parts[0]
        subpath = parts[1] if len(parts) > 1 else ""
        if shared_key not in SHARED_RESOURCES:
            raise ValueError(f"Unknown shared resource: {shared_key}")
        base = MODELS_DIR / SHARED_RESOURCES[shared_key]["local_dir"]
        return base / subpath if subpath else base
    if directive.startswith("model:"):
        rest = directive[len("model:"):]
        parts = rest.split("/", 1)
        model_id = parts[0]
        subpath = parts[1] if len(parts) > 1 else ""
        base = MODELS_DIR / model_id
        return base / subpath if subpath else base
    return Path(directive)


def _load_model_index(model_path: Path) -> dict:
    index_path = model_path / "model_index.json"
    if not index_path.exists():
        return {}
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_runtime_alignment(model_cfg: dict, model_index: dict, diffusers_version: str) -> None:
    runtime = model_cfg.get("runtime", {})
    if runtime.get("skip_model_index_validation"):
        print("[runtime_alignment] Skipping model_index validation", flush=True)
        return
    expected_pipeline = runtime.get("expected_model_index_class") or runtime.get("pipeline_class")
    native_pipeline = model_index.get("_class_name")
    if native_pipeline and expected_pipeline and native_pipeline != expected_pipeline:
        raise RuntimeError(
            f"Downloaded model declares '{native_pipeline}', registry expects '{expected_pipeline}'."
        )
    min_version = runtime.get("min_diffusers_version")
    if min_version and _parse_version(diffusers_version) < _parse_version(min_version):
        raise RuntimeError(f"Need diffusers >= {min_version}; got {diffusers_version}.")


def get_free_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return free / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def get_hardware_fingerprint() -> str:
    """Stable ID for this machine's GPU. Invalidates profile if GPU is swapped."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory // (1024 ** 3)
            return f"{props.name}|{vram_gb}GB|cuda{torch.version.cuda}"
    except Exception:
        pass
    return "cpu-only"


def _ensure_pipeline_source_on_path(runtime: dict) -> None:
    source = runtime.get("pipeline_source")
    if not source or source.get("type") != "local_package":
        return
    add_path = source.get("add_to_sys_path")
    if not add_path:
        return
    resolved = _resolve_path_directive(add_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Pipeline source path not found: {resolved}")
    path_str = str(resolved)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"[pipeline_import] Added to sys.path: {path_str}", flush=True)


def _resolve_pipeline_class(runtime: dict):
    pipeline_cls_name = runtime.get("pipeline_class")
    if not pipeline_cls_name:
        raise ValueError("runtime.pipeline_class not set in model JSON")
    pipeline_module = runtime.get("pipeline_module", "diffusers")
    module = importlib.import_module(pipeline_module)
    if not hasattr(module, pipeline_cls_name):
        raise AttributeError(f"Module '{pipeline_module}' has no attribute '{pipeline_cls_name}'")
    return getattr(module, pipeline_cls_name), pipeline_module


def _resolve_factory_kwargs(raw_kwargs: dict) -> dict:
    resolved = {}
    for key, value in raw_kwargs.items():
        if isinstance(value, str) and (value.startswith("shared:") or value.startswith("model:")):
            resolved[key] = str(_resolve_path_directive(value))
        else:
            resolved[key] = value
    return resolved


def _load_via_factory(runtime: dict):
    factory = runtime.get("factory")
    if not factory:
        raise ValueError("load_mode 'factory' requires runtime.factory")
    module = importlib.import_module(factory["module"])
    fn = getattr(module, factory["function"])
    kwargs = _resolve_factory_kwargs(dict(factory.get("kwargs", {})))
    sig = inspect.signature(fn)
    accepts_var = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not accepts_var:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    print(f"[fn_load_pipeline] Factory {factory['module']}.{factory['function']}(kwargs={list(kwargs.keys())})", flush=True)
    return fn(**kwargs)


def _resolve_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise AttributeError(f"Module '{module_name}' has no attribute '{class_name}'")
    return getattr(module, class_name)


# ═════════════════════════════════════════════════════════════════════
# ERROR CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════

def classify_error(error: BaseException) -> str:
    """
    Returns one of:
      'oom'         — CUDA OOM, retry with lighter strategy
      'cuda_crash'  — native crash (bnb on Blackwell, DLL missing, etc)
      'meta_tensor' — incompatible offload strategy
      'import'      — missing python package
      'timeout'     — subprocess/network timeout
      'error'       — other exception (may or may not be strategy-related)
    """
    import torch
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "oom"
    s = str(error).lower()
    if "out of memory" in s or "cuda out of memory" in s:
        return "oom"
    if "0xc0000005" in s or "access violation" in s or "illegal memory access" in s:
        return "cuda_crash"
    if "meta tensor" in s:
        return "meta_tensor"
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "import"
    if "timed out" in s or "timeout" in s:
        return "timeout"
    return "error"


def build_error_report(error: BaseException, ctx, strategy_name: str) -> dict:
    import torch
    report = {
        "timestamp": datetime.now().isoformat(),
        "sidecar_build": SIDECAR_BUILD,
        "model_id": ctx.model_id,
        "job_id": ctx.job_id,
        "strategy": strategy_name,
        "error_class": type(error).__name__,
        "error_message": str(error),
        "error_category": classify_error(error),
        "traceback": traceback.format_exc(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "hardware": {
            "gpu": None,
            "vram_gb": 0,
            "cuda": None,
        },
    }
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["hardware"]["gpu"] = props.name
            report["hardware"]["vram_gb"] = props.total_memory // (1024 ** 3)
            report["hardware"]["cuda"] = torch.version.cuda
    except Exception:
        pass
    return report


# ═════════════════════════════════════════════════════════════════════
# CONTEXT
# ═════════════════════════════════════════════════════════════════════

class PipelineContext:
    def __init__(self):
        self.job_id = None
        self.model_id = None
        self.model_cfg = None
        self.raw_params = None
        self.resolved_params = None
        self.pipe = None
        self.pipeline_module = None
        self.video_output = None
        self.video_output_path = None
        self.generation_args = None
        self.offload_strategy = None
        self.step_callback = None
        # Set by orchestrator each attempt
        self.active_strategy = None
        # Pre-computed prompt embeds (set by fn_encode_prompt_on_cpu)
        self.precomputed_embeds = None  # dict with prompt_embeds / prompt_attention_mask / negative_* / negative_*_mask


# ═════════════════════════════════════════════════════════════════════
# STEP FUNCTIONS
# ═════════════════════════════════════════════════════════════════════

def fn_load_job(ctx: PipelineContext, config: dict) -> None:
    params_path = Path(config.get("params_path"))
    if not params_path.exists():
        raise FileNotFoundError(f"Job parameter file not found: {params_path}")
    with open(params_path, "r", encoding="utf-8") as f:
        raw_params = json.load(f)

    job_id = raw_params.get("job_id")
    model_id = raw_params.get("model_id")
    if not job_id or not model_id:
        raise ValueError("Invalid job params")
    if model_id not in MODEL_CONFIG:
        raise ValueError(f"Unknown model_id: {model_id}")

    model_cfg = MODEL_CONFIG[model_id]
    resolved_params = resolve_generation_params(model_id, raw_params)

    generation_defaults = model_cfg.get("generation", {}).get("defaults", {})
    for key, default_value in generation_defaults.items():
        if resolved_params.get(key) is None and default_value is not None:
            resolved_params[key] = default_value
            print(f"[fn_load_job] Applied default: {key}={default_value}", flush=True)

    # Prompt is taken from user input as-is (may be empty for text-free generation)
    resolved_params["prompt"] = raw_params.get("prompt", "")

    # Negative prompt: use user's if non-empty, otherwise fall back to model default.
    # This lets models declare a recommended negative prompt in their JSON without
    # hardcoding it in the worker. Models that don't need one simply omit the field.
    user_negative = raw_params.get("negative_prompt", "") or ""
    if user_negative.strip():
        resolved_params["negative_prompt"] = user_negative
        print(f"[fn_load_job] Using user-supplied negative_prompt ({len(user_negative)} chars)", flush=True)
    else:
        default_negative = generation_defaults.get("negative_prompt", "")
        resolved_params["negative_prompt"] = default_negative
        if default_negative:
            print(f"[fn_load_job] Applied model default negative_prompt ({len(default_negative)} chars)", flush=True)
        else:
            print(f"[fn_load_job] No negative_prompt provided or defaulted", flush=True)

    ctx.job_id = job_id
    ctx.model_id = model_id
    ctx.model_cfg = model_cfg
    ctx.raw_params = raw_params
    ctx.resolved_params = resolved_params

    print(
        f"[fn_load_job] {job_id} | {model_id} | "
        f"steps={resolved_params.get('steps')}, cfg={resolved_params.get('cfg_scale')}, "
        f"{resolved_params.get('width')}x{resolved_params.get('height')}",
        flush=True,
    )


def fn_disable_dynamic_shifting(ctx: PipelineContext, config: dict) -> None:
    """
    LTX-Video 0.9.5 relies on resolution-dependent dynamic timestep shifting (`use_dynamic_shifting=True`).
    `diffusers` 0.39.0's `LTXConditionPipeline` invokes `retrieve_timesteps` without passing `mu`,
    causing `FlowMatchEulerDiscreteScheduler` to raise a `ValueError` if `mu` is missing.

    Disabling dynamic shifting breaks the noise schedule and leads to corrupted rainbow static output.
    Instead, this function ensures `use_dynamic_shifting` remains `True` and patches `scheduler.set_timesteps`
    to automatically compute the exact dynamic `mu` for the target video resolution and frame count.
    """
    pipe = ctx.pipe
    if not hasattr(pipe, "scheduler") or pipe.scheduler is None:
        print("[fn_disable_dynamic_shifting] No scheduler on pipe. Skipping.", flush=True)
        return

    scheduler = pipe.scheduler
    if not hasattr(scheduler, "config"):
        print("[fn_disable_dynamic_shifting] Scheduler has no config. Skipping.", flush=True)
        return

    # Ensure use_dynamic_shifting is True in scheduler config
    if not getattr(scheduler.config, "use_dynamic_shifting", False):
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            new_config = dict(scheduler.config)
            new_config["use_dynamic_shifting"] = True
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(new_config)
            scheduler = pipe.scheduler
            print("[fn_disable_dynamic_shifting] Enabled use_dynamic_shifting=True on scheduler", flush=True)
        except Exception as e:
            print(f"[fn_disable_dynamic_shifting] [WARN] Could not enable dynamic shifting: {e}", flush=True)

    orig_set_timesteps = scheduler.set_timesteps

    def set_timesteps_with_dynamic_mu(num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None, **kwargs):
        if getattr(scheduler.config, "use_dynamic_shifting", False) and mu is None:
            params = getattr(ctx, "resolved_params", {}) or {}
            height = params.get("height", 480)
            width = params.get("width", 704)
            num_frames = params.get("num_frames", 121)

            # Spatial compression = 32 (VAE factor 8 * spatial patch 4, or 32 for tokens)
            # Temporal compression = 8
            vae_spatial = getattr(pipe, "vae_spatial_compression_ratio", 32)
            vae_temporal = getattr(pipe, "vae_temporal_compression_ratio", 8)

            latent_h = height // vae_spatial
            latent_w = width // vae_spatial
            latent_f = (num_frames - 1) // vae_temporal + 1

            n_tokens = latent_f * latent_h * latent_w

            min_tokens = scheduler.config.get("base_image_seq_len", 1024)
            max_tokens = scheduler.config.get("max_image_seq_len", 4096)
            min_shift = scheduler.config.get("base_shift", 0.95)
            max_shift = scheduler.config.get("max_shift", 2.05)

            m = (max_shift - min_shift) / (max_tokens - min_tokens)
            b = min_shift - m * min_tokens
            mu = m * n_tokens + b
            print(
                f"[fn_disable_dynamic_shifting] Calculated dynamic shift mu={mu:.4f} "
                f"for n_tokens={n_tokens} ({width}x{height}, {num_frames}f)",
                flush=True,
            )

        return orig_set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
            **kwargs,
        )

    scheduler.set_timesteps = set_timesteps_with_dynamic_mu
    print("[fn_disable_dynamic_shifting] Patched scheduler.set_timesteps with dynamic mu calculation", flush=True)

def fn_ensure_dependencies(ctx: PipelineContext, config: dict) -> None:
    import subprocess
    deps = ctx.model_cfg.get("runtime", {}).get("python_dependencies", [])
    if not deps:
        return
    missing = []
    for pkg in deps:
        import_name = pkg.replace("-", "_").split(">=")[0].split("==")[0]
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg)
    if not missing:
        print("[fn_ensure_dependencies] All satisfied", flush=True)
        return
    # Never pip-install from model config at runtime: packages ship with the app's
    # environment (requirements.txt), so a config entry can't pull arbitrary code.
    raise RuntimeError(f"Missing Python packages for this model: {missing}. Reinstall NeuralCut.")


def fn_load_pipeline(ctx: PipelineContext, config: dict) -> None:
    import torch
    import diffusers

    
    # A strategy can override the entire runtime block by specifying a key path
    # into the model_cfg, e.g. config["runtime_key"] = "runtime_lightricks".
    # Defaults to the base "runtime" block. This lets us keep multiple loader
    # profiles (diffusers vs Lightricks factory) side-by-side in the same JSON.
    runtime_key = config.get("runtime_key", "runtime")
    runtime = ctx.model_cfg.get(runtime_key)
    if not runtime:
        raise ValueError(f"Model config has no '{runtime_key}' block")
    print(f"[fn_load_pipeline] Using runtime block: '{runtime_key}'", flush=True)

    load_mode = runtime.get("load_mode", "from_pretrained")
    dtype = getattr(torch, runtime.get("dtype", "float16"), torch.float16)
    load_kwargs = dict(runtime.get("load_kwargs", {}))

    _ensure_pipeline_source_on_path(runtime)

    if load_mode == "factory":
        pipe = _load_via_factory(runtime)
        ctx.pipe = pipe
        print(f"[fn_load_pipeline] Built via factory: {type(pipe).__name__}", flush=True)
        return

    pipeline_cls, pipeline_module = _resolve_pipeline_class(runtime)
    ctx.pipeline_module = pipeline_module

    if load_mode == "from_pretrained":
        model_path = MODELS_DIR / ctx.model_id
        model_index = _load_model_index(model_path)
        _check_runtime_alignment(ctx.model_cfg, model_index, diffusers.__version__)
        pipe = pipeline_cls.from_pretrained(
            str(model_path), torch_dtype=dtype,
            local_files_only=True, low_cpu_mem_usage=True, **load_kwargs,
        )
    elif load_mode == "from_single_file":
        transformer_cls = _resolve_class(
            runtime.get("transformer_module", "diffusers"),
            runtime["transformer_class"],
        )
        transformer_path = MODELS_DIR / ctx.model_id / runtime["transformer_file"]
        transformer = transformer_cls.from_single_file(
            str(transformer_path), torch_dtype=dtype, local_files_only=True,
        )
        shared_key = ctx.model_cfg["distribution"]["shared_resources"]
        shared_path = MODELS_DIR / SHARED_RESOURCES[shared_key]["local_dir"]
        model_index = _load_model_index(shared_path)
        _check_runtime_alignment(ctx.model_cfg, model_index, diffusers.__version__)
        pipe = pipeline_cls.from_pretrained(
            shared_path.as_posix(), transformer=transformer, torch_dtype=dtype,
            local_files_only=True, low_cpu_mem_usage=True, **load_kwargs,
        )
    else:
        raise ValueError(f"Unknown load_mode: {load_mode}")

    ctx.pipe = pipe
    print(f"[fn_load_pipeline] Loaded: {pipeline_cls.__name__} from {pipeline_module}", flush=True)


def fn_post_load_cast(ctx: PipelineContext, config: dict) -> None:
    """Cast components to a target dtype. Int8 quantization is intentionally NOT supported
    here — it's fragile across GPUs. Use a separate strategy for that."""
    import torch
    pipe = ctx.pipe
    casts = config.get("casts", [])
    if not casts:
        return
    for cast in casts:
        component_name = cast.get("component")
        if not component_name or not hasattr(pipe, component_name):
            continue
        component = getattr(pipe, component_name)
        if component is None:
            continue
        dtype_str = cast.get("dtype")
        if not dtype_str:
            continue
        target_dtype = getattr(torch, dtype_str, None)
        if target_dtype is None:
            continue
        try:
            current_dtype = next(component.parameters()).dtype
        except StopIteration:
            current_dtype = "unknown"
        print(f"[fn_post_load_cast] {component_name}: {current_dtype} → {target_dtype}", flush=True)
        setattr(pipe, component_name, component.to(target_dtype))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("[fn_post_load_cast] Complete", flush=True)


def fn_quantize(ctx: PipelineContext, config: dict) -> None:
    import torch
    pipe = ctx.pipe
    target = config.get("quantization") or ctx.resolved_params.get("quantization") or "none"
    print(f"[fn_quantize] '{target}'", flush=True)
    if target not in ("none", "auto"):
        try:
            if target == "fp8":
                if hasattr(pipe, "transformer") and hasattr(torch, "float8_e4m3fn"):
                    pipe.transformer.to(torch.float8_e4m3fn)
            elif target in ("quanto_int8", "quanto_fp8"):
                from optimum.quanto import quantize, qint8, qfloat8
                qtype = qint8 if "int8" in target else qfloat8
                comp = getattr(pipe, "transformer", getattr(pipe, "unet", None))
                if comp is not None:
                    quantize(comp, weights=qtype)
        except Exception as e:
            print(f"[fn_quantize] [WARN] {e}", flush=True)

    # Independent of the transformer's own quantization target — lets a
    # strategy shrink just the text encoder (e.g. gpu_full needs T5 small
    # enough to coexist with transformer+VAE on an 8GB card).
    te_target = config.get("text_encoder_quantization")
    if te_target and hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        try:
            from optimum.quanto import quantize, qint8, qfloat8
            qtype = qint8 if te_target == "int8" else qfloat8
            quantize(pipe.text_encoder, weights=qtype)
            print(f"[fn_quantize] text_encoder quantized to {te_target}", flush=True)
        except Exception as e:
            print(f"[fn_quantize] [WARN] text_encoder quantization failed: {e}", flush=True)

def fn_configure_scheduler(ctx: PipelineContext, config: dict) -> None:
    sched_json = ctx.model_cfg.get("scheduler", {})
    if sched_json.get("override") is False:
        print("[fn_configure_scheduler] Override disabled by JSON", flush=True)
        return
    target = config.get("scheduler") or ctx.resolved_params.get("scheduler")
    class_name = sched_json.get("class_name")
    if not target or not class_name or target == "Default":
        print("[fn_configure_scheduler] Using default", flush=True)
        return
    import diffusers
    try:
        cls = getattr(diffusers, class_name)
        ctx.pipe.scheduler = cls.from_config(dict(ctx.pipe.scheduler.config), **sched_json.get("params", {}))
        print(f"[fn_configure_scheduler] Set to {class_name}", flush=True)
    except Exception as e:
        print(f"[fn_configure_scheduler] [WARN] {e}", flush=True)


def fn_configure_memory(ctx: PipelineContext, config: dict) -> None:
    """Apply offload strategy from config or JSON.
    Precedence: config.force_strategy > runtime_constraints.force_offload_strategy > auto."""
    import torch
    from model_registry import resolve_offload_strategy

    pipe = ctx.pipe
    model_id = ctx.model_id
    free_vram = get_free_vram_gb()

    forced = config.get("force_strategy")
    if not forced:
        forced = ctx.model_cfg.get("runtime_constraints", {}).get("force_offload_strategy")

    if forced:
        strategy = forced
    else:
        strategy = resolve_offload_strategy(model_id, free_vram)

    print(f"[fn_configure_memory] Free VRAM: {free_vram:.2f}GB | Strategy: '{strategy}'", flush=True)

    try:
        torch.backends.cuda.preferred_linalg_library("magma")
    except Exception:
        pass

    if not torch.cuda.is_available():
        ctx.offload_strategy = "cpu"
        return

    def try_model_offload():
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            ctx.offload_strategy = "model_offload"
            print("[fn_configure_memory] Model CPU offload enabled", flush=True)
            return True
        return False

    def try_sequential_offload():
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            ctx.offload_strategy = "sequential_offload"
            print("[fn_configure_memory] Sequential CPU offload enabled", flush=True)
            return True
        return False

    if strategy == "manual":
        ctx.offload_strategy = "manual"
        print("[fn_configure_memory] Manual — pipeline manages placement", flush=True)
        return
    if strategy == "sequential":
        try_sequential_offload() or try_model_offload()
        return
    if strategy == "model":
        try_model_offload() or try_sequential_offload()
        return
    # strategy == "gpu"
    try:
        ctx.pipe = pipe.to("cuda")
        ctx.offload_strategy = "gpu"
        print("[fn_configure_memory] Full GPU", flush=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try_model_offload() or try_sequential_offload()


def fn_set_execution_device(ctx: PipelineContext, config: dict) -> None:
    import torch
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[fn_set_execution_device] CUDA unavailable", flush=True)
        return
    device = torch.device(device_str)
    pipe = ctx.pipe
    components = config.get("components", ["transformer", "vae", "text_encoder"])
    skip = set(config.get("skip_components", []))

    for name in components:
        if name in skip:
            continue
        if not hasattr(pipe, name):
            continue
        comp = getattr(pipe, name)
        if comp is None or not hasattr(comp, "to"):
            continue
        try:
            current = next(comp.parameters()).device
            if current.type == device.type:
                continue
            print(f"[fn_set_execution_device] {name}: {current} → {device}", flush=True)
            setattr(pipe, name, comp.to(device))
        except torch.cuda.OutOfMemoryError as e:
            print(f"[fn_set_execution_device] [OOM] {name}", flush=True)
            torch.cuda.empty_cache()
            raise  # bubble up — the ladder handles this
        except Exception as e:
            print(f"[fn_set_execution_device] [WARN] {name}: {e}", flush=True)

    for name in ("transformer", "vae", "text_encoder"):
        if hasattr(pipe, name):
            comp = getattr(pipe, name)
            if comp is not None and hasattr(comp, "parameters"):
                try:
                    d = next(comp.parameters()).device
                    print(f"[fn_set_execution_device] Final: {name} on {d}", flush=True)
                except StopIteration:
                    pass

def fn_encode_prompt_on_gpu(ctx: PipelineContext, config: dict) -> None:
    """
    Encode the prompt on GPU using the pipeline's text encoder (T5), then
    evict the text encoder back to CPU (kept alive, not None — some
    pipelines assert it exists). Stores ctx.precomputed_embeds for
    prepare_generation_args to inject.
    """
    import torch

    pipe = ctx.pipe
    if not hasattr(pipe, "text_encoder") or pipe.text_encoder is None:
        print("[fn_encode_prompt_on_gpu] No text_encoder. Skipping.", flush=True)
        return
    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        print("[fn_encode_prompt_on_gpu] No tokenizer. Skipping.", flush=True)
        return
    if not torch.cuda.is_available():
        print("[fn_encode_prompt_on_gpu] CUDA unavailable. Skipping.", flush=True)
        return

    prompt = ctx.resolved_params.get("prompt", "")
    negative_prompt = ctx.resolved_params.get("negative_prompt", "")
    max_tokens = config.get("max_tokens", 256)

    print("[fn_encode_prompt_on_gpu] Moving text_encoder to cuda", flush=True)
    pipe.text_encoder = pipe.text_encoder.to("cuda")

    def encode(text: str):
        inputs = pipe.tokenizer(
            text if text else "",
            padding="max_length",
            max_length=max_tokens,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = pipe.text_encoder(
                input_ids=inputs.input_ids.to("cuda"),
                attention_mask=inputs.attention_mask.to("cuda"),
            )
        embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return embeds, inputs.attention_mask.to("cuda")

    print("[fn_encode_prompt_on_gpu] Encoding prompt...", flush=True)
    t0 = time.time()
    p_embeds, p_mask = encode(prompt)
    n_embeds, n_mask = encode(negative_prompt)
    elapsed = time.time() - t0
    print(f"[fn_encode_prompt_on_gpu] Encoded in {elapsed:.2f}s", flush=True)

    target_dtype = torch.bfloat16
    ctx.precomputed_embeds = {
        "prompt_embeds": p_embeds.to(dtype=target_dtype),
        "prompt_attention_mask": p_mask,
        "negative_prompt_embeds": n_embeds.to(dtype=target_dtype),
        "negative_prompt_attention_mask": n_mask,
    }
    print(f"[fn_encode_prompt_on_gpu] Embeds ready on cuda as {target_dtype}", flush=True)

    print("[fn_encode_prompt_on_gpu] Evicting text_encoder to cpu", flush=True)
    pipe.text_encoder = pipe.text_encoder.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

def fn_wrap_vae_decode(ctx: PipelineContext, config: dict) -> None:
    """
    Defers VAE onto GPU only for the moment it's actually needed. Patches the
    module-level 'vae_decode' name bound inside the pipeline's own module
    (not the defining module — pipelines import it by name, so patching the
    defining module wouldn't affect the already-bound reference). Moves
    pipe.vae to cuda BEFORE calling the real vae_decode — un_normalize_latents
    and similar helpers read buffers off vae before decode() runs internally,
    so the move must happen before entering vae_decode, not inside it.
    """
    import sys
    import torch

    pipe = ctx.pipe
    if not hasattr(pipe, "vae") or pipe.vae is None:
        print("[fn_wrap_vae_decode] No vae. Skipping.", flush=True)
        return

    module_name = type(pipe).__module__
    pipeline_module = sys.modules.get(module_name)
    if pipeline_module is None or not hasattr(pipeline_module, "vae_decode"):
        print(f"[fn_wrap_vae_decode] No 'vae_decode' found in {module_name}. Skipping.", flush=True)
        return

    original_vae_decode = pipeline_module.vae_decode

    # Move VAE off GPU now — it'll only come back for the actual decode call.
    pipe.vae = pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    print("[fn_wrap_vae_decode] vae moved to cpu, decode deferred", flush=True)

    def wrapped_vae_decode(latents, vae, *args, **kwargs):
        print("[fn_wrap_vae_decode] Moving vae to cuda for decode", flush=True)
        vae.to("cuda")
        try:
            return original_vae_decode(latents, vae, *args, **kwargs)
        finally:
            vae.to("cpu")
            torch.cuda.empty_cache()
            print("[fn_wrap_vae_decode] vae evicted back to cpu", flush=True)

    pipeline_module.vae_decode = wrapped_vae_decode
    print(f"[fn_wrap_vae_decode] Patched vae_decode in {module_name}", flush=True)

def fn_apply_accelerate_offload(ctx: PipelineContext, config: dict) -> None:
    """
    Applies accelerate's layer-level CPU offload to a specific component
    (default: transformer). Each nn.Module inside the target is hooked so
    that its weights are moved CPU->CUDA just before its forward runs,
    and back to CPU after.

    This is different from pipe.enable_model_cpu_offload() (which moves
    whole components) and pipe.enable_sequential_cpu_offload() (which
    the Lightricks custom pipeline doesn't handle well). It's also
    different from pipe.enable_sequential_cpu_offload's semantics — we
    apply hooks directly to the target module, bypassing any pipeline-
    level orchestration.

    Peak VRAM: ~1-2GB of weights (single largest layer) + activations,
    instead of the ~4-5GB full-transformer footprint. Trade-off is
    per-step latency: PCIe transfer overhead per layer per forward pass.
    """
    import torch

    try:
        from accelerate import cpu_offload
    except ImportError:
        print("[fn_apply_accelerate_offload] accelerate not installed. Skipping.", flush=True)
        return

    pipe = ctx.pipe
    component_name = config.get("component", "transformer")

    if not hasattr(pipe, component_name):
        print(f"[fn_apply_accelerate_offload] No '{component_name}' on pipe. Skipping.", flush=True)
        return

    component = getattr(pipe, component_name)
    if component is None:
        print(f"[fn_apply_accelerate_offload] '{component_name}' is None. Skipping.", flush=True)
        return

    # Ensure component is on CPU before installing hooks. If it's already on
    # GPU, hooks would still attach but the initial state would be wrong —
    # cpu_offload assumes the module starts on CPU and gets moved to GPU
    # per-forward. Moving to CPU first also frees the VRAM the component
    # was occupying, so the first forward has room to bring layers back.
    try:
        current_device = next(component.parameters()).device
        if current_device.type != "cpu":
            print(f"[fn_apply_accelerate_offload] Moving {component_name} from {current_device} to cpu first", flush=True)
            setattr(pipe, component_name, component.to("cpu"))
            component = getattr(pipe, component_name)
            torch.cuda.empty_cache()
    except StopIteration:
        pass

    execution_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[fn_apply_accelerate_offload] Installing layer-level offload hooks on {component_name} "
          f"(execution_device={execution_device})", flush=True)

    # cpu_offload attaches AlignDevicesHook to every submodule. Each hook
    # moves that submodule's parameters/buffers to execution_device on
    # pre-forward, and back to CPU on post-forward. Nested modules are
    # handled — a Linear inside an Attention inside a Block gets its own
    # hook and is moved independently. This gives finer-grained memory
    # control than model_cpu_offload (whole-component) but coarser than
    # per-parameter offload.
    cpu_offload(component, execution_device=execution_device, offload_buffers=False)

    print(f"[fn_apply_accelerate_offload] Offload hooks installed on {component_name}", flush=True)

    # Sanity check: verify the hook was actually installed on the top-level
    # module. If accelerate silently no-ops on a custom module class, we
    # want to know before the diffusion loop starts.
    has_hook = hasattr(component, "_hf_hook")
    print(f"[fn_apply_accelerate_offload] Verify: {component_name}._hf_hook exists = {has_hook}", flush=True)

def fn_neutralize_pipeline_device_moves(ctx: PipelineContext, config: dict) -> None:
    """
    The Lightricks LTXVideoPipeline manually calls self.transformer.to(execution_device)
    inside __call__ (line 1045 in pipeline_ltx_video.py) before the diffusion loop.
    This conflicts with any framework-level offload (accelerate hooks, meta tensors, etc)
    that expects to control device placement itself.

    We intercept .to() on the transformer instance: if a device move is requested and
    accelerate hooks are installed (indicated by _hf_hook), the move becomes a no-op.
    Dtype-only moves still pass through. This preserves the pipeline's ability to change
    numerical precision while preventing it from tearing down our offload plumbing.
    """
    import torch

    pipe = ctx.pipe
    component_name = config.get("component", "transformer")
    if not hasattr(pipe, component_name):
        print(f"[fn_neutralize_pipeline_device_moves] No '{component_name}' on pipe. Skipping.", flush=True)
        return
    component = getattr(pipe, component_name)
    if component is None:
        print(f"[fn_neutralize_pipeline_device_moves] '{component_name}' is None. Skipping.", flush=True)
        return

    if not hasattr(component, "_hf_hook"):
        print(f"[fn_neutralize_pipeline_device_moves] {component_name} has no _hf_hook — "
              f"offload not installed. Skipping patch.", flush=True)
        return

    if getattr(component, "_nc_to_neutralized", False):
        print(f"[fn_neutralize_pipeline_device_moves] Already patched. Skipping.", flush=True)
        return

    original_to = component.to

    def guarded_to(*args, **kwargs):
        # Detect whether a device change is being requested. PyTorch's Module.to accepts
        # (device), (dtype), (device, dtype), (tensor), or keyword forms — we check all
        # positional args plus the 'device' kwarg for anything that looks like a device.
        requested_device = kwargs.get("device")
        if requested_device is None:
            for a in args:
                if isinstance(a, (torch.device, str)) and a not in ("cpu",) and "cuda" in str(a):
                    requested_device = a
                    break
                if isinstance(a, torch.device):
                    requested_device = a
                    break

        if requested_device is not None:
            print(f"[fn_neutralize_pipeline_device_moves] Intercepted {component_name}.to("
                  f"device={requested_device}) — ignoring (offload hooks own placement)", flush=True)
            return component  # return self, as .to() would

        # No device change — this is dtype-only or a no-op, let it through
        return original_to(*args, **kwargs)

    component.to = guarded_to
    component._nc_to_neutralized = True
    print(f"[fn_neutralize_pipeline_device_moves] Guarded {component_name}.to() "
          f"against device moves while offload hooks are active", flush=True)

def fn_encode_prompt_on_cpu(ctx: PipelineContext, config: dict) -> None:
    """
    Encode the prompt on CPU using the pipeline's text encoder (T5), then
    inject prompt_embeds into generation args. Does NOT unload text_encoder
    because some pipelines assert it exists even when embeds are provided.
    """
    import torch

    pipe = ctx.pipe
    if not hasattr(pipe, "text_encoder") or pipe.text_encoder is None:
        print("[fn_encode_prompt_on_cpu] No text_encoder. Skipping.", flush=True)
        return
    if not hasattr(pipe, "tokenizer") or pipe.tokenizer is None:
        print("[fn_encode_prompt_on_cpu] No tokenizer. Skipping.", flush=True)
        return

    prompt = ctx.resolved_params.get("prompt", "")
    negative_prompt = ctx.resolved_params.get("negative_prompt", "")
    max_tokens = config.get("max_tokens", 256)

    original_threads = torch.get_num_threads()
    try:
        n_cpus = os.cpu_count() or 4
        torch.set_num_threads(n_cpus)
        print(f"[fn_encode_prompt_on_cpu] Using {n_cpus} CPU threads", flush=True)

        # T5EncoderModel must NOT be run in float16 on CPU as PyTorch CPU float16 produces NaNs.
        cpu_dtype = torch.bfloat16
        print(f"[fn_encode_prompt_on_cpu] Ensuring text_encoder on CPU in {cpu_dtype}", flush=True)
        pipe.text_encoder = pipe.text_encoder.to("cpu").to(cpu_dtype)

        def encode(text: str):
            inputs = pipe.tokenizer(
                text if text else "",
                padding="max_length",
                max_length=max_tokens,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = pipe.text_encoder(
                    input_ids=inputs.input_ids.to("cpu"),
                    attention_mask=inputs.attention_mask.to("cpu"),
                )
            embeds = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            return embeds, inputs.attention_mask.to("cpu")

        print("[fn_encode_prompt_on_cpu] Encoding prompt...", flush=True)
        t0 = time.time()
        p_embeds, p_mask = encode(prompt)
        n_embeds, n_mask = encode(negative_prompt)
        elapsed = time.time() - t0
        print(f"[fn_encode_prompt_on_cpu] Encoded in {elapsed:.1f}s", flush=True)

        # Move embeds to GPU in the transformer's dtype (bfloat16)
        target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        ctx.precomputed_embeds = {
            "prompt_embeds": p_embeds.to(device, dtype=target_dtype),
            "prompt_attention_mask": p_mask.to(device),
            "negative_prompt_embeds": n_embeds.to(device, dtype=target_dtype),
            "negative_prompt_attention_mask": n_mask.to(device),
        }
        print(f"[fn_encode_prompt_on_cpu] Embeds ready on {device} as {target_dtype}", flush=True)

        # Only unload if explicitly requested AND we've verified the pipeline doesn't assert
        if config.get("unload_text_encoder", False):
            print("[fn_encode_prompt_on_cpu] Unloading text_encoder from RAM", flush=True)
            try:
                pipe.text_encoder = None
                gc.collect()
            except Exception as e:
                print(f"[fn_encode_prompt_on_cpu] [WARN] Could not unload: {e}", flush=True)
        else:
            print("[fn_encode_prompt_on_cpu] Keeping text_encoder in RAM (pipeline asserts on None)", flush=True)

    finally:
        torch.set_num_threads(original_threads)

def fn_optimize(ctx: PipelineContext, config: dict) -> None:
    import torch
    pipe = ctx.pipe
    opts = config.get("optimization", ctx.model_cfg.get("optimization", {}))

    if opts.get("tf32", True) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[fn_optimize] TF32 enabled", flush=True)

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        print("[fn_optimize] SDP backends: flash=on, mem_efficient=on, math=off", flush=True)

    if opts.get("vae_tiling", True):
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        print("[fn_optimize] VAE Tiling", flush=True)

    if opts.get("vae_slicing", True):
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        print("[fn_optimize] VAE Slicing", flush=True)

    if opts.get("attention_slicing", False) and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        print("[fn_optimize] Attention Slicing", flush=True)

    if opts.get("xformers", False) and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[fn_optimize] xFormers enabled", flush=True)
        except Exception as e:
            print(f"[fn_optimize] [WARN] xFormers: {e}", flush=True)


def fn_prepare_generation_args(ctx: PipelineContext, config: dict) -> None:
    import torch
    pipe = ctx.pipe
    params = ctx.resolved_params
    gen_cfg = ctx.model_cfg.get("generation", {})
    arg_aliases: Dict[str, str] = gen_cfg.get("arg_aliases", {})
    extra_pipeline_args: List[str] = gen_cfg.get("extra_pipeline_args", [])
    defaults = gen_cfg.get("defaults", {})

    steps = params.get("steps", 20)
    cfg_scale = params.get("cfg_scale", 4.0)
    width = params.get("width", 640)
    height = params.get("height", 352)
    num_frames = params.get("num_frames", 49)
    fps = params.get("fps", 16)
    seed = params.get("seed")

    print(f"[fn_prepare_generation_args] steps={steps}, cfg={cfg_scale}, {width}x{height}, {num_frames}f @ {fps}fps", flush=True)

    pipe_sig = inspect.signature(pipe.__call__)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in pipe_sig.parameters.values())

    def supports(name: str) -> bool:
        return name in pipe_sig.parameters or accepts_var_kwargs

    def resolve_arg_name(canonical: str) -> Optional[str]:
        aliased = arg_aliases.get(canonical)
        if aliased:
            if supports(aliased):
                return aliased
            if canonical in pipe_sig.parameters:
                return canonical
            return None
        if canonical in pipe_sig.parameters:
            return canonical
        if accepts_var_kwargs:
            return canonical
        return None

    pipe_kwargs: Dict[str, Any] = {}

    # If we pre-encoded on CPU, use embeds instead of raw prompt
    if ctx.precomputed_embeds:
        for k, v in ctx.precomputed_embeds.items():
            if supports(k):
                pipe_kwargs[k] = v
                print(f"[fn_prepare_generation_args] Injected precomputed: {k}", flush=True)
        for text_key in ("prompt", "negative_prompt"):
            if supports(text_key):
                pipe_kwargs[text_key] = None
    else:
        pipe_kwargs["prompt"] = params.get("prompt", "")
        pipe_kwargs["negative_prompt"] = params.get("negative_prompt", "")

    core_map = {
        "num_inference_steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
    }
    for canonical, value in core_map.items():
        actual = resolve_arg_name(canonical)
        if actual:
            pipe_kwargs[actual] = value
            if actual != canonical:
                print(f"[fn_prepare_generation_args] Aliased {canonical} → {actual}", flush=True)

    if seed is not None and int(seed) >= 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe_kwargs["generator"] = torch.Generator(device=device).manual_seed(int(seed))

    if config.get("enable_step_callback", True):
        if supports("callback_on_step_end"):
            pipe_kwargs["callback_on_step_end"] = ctx.step_callback
        elif supports("callback"):
            pipe_kwargs["callback"] = ctx.step_callback

    coercions = gen_cfg.get("arg_coercions", {})

    def coerce(name: str, value):
        spec = coercions.get(name)
        if not spec:
            return value
        if spec.get("kind") == "enum" and isinstance(value, str):
            enum_cls = _resolve_class(spec["module"], spec["name"])
            return getattr(enum_cls, value)
        return value

    for arg_name in extra_pipeline_args:
        value = params.get(arg_name, defaults.get(arg_name))
        if value is None:
            continue
        value = coerce(arg_name, value)
        actual = resolve_arg_name(arg_name)
        if actual:
            pipe_kwargs[actual] = value
            if actual != arg_name:
                print(f"[fn_prepare_generation_args] Added (aliased): {arg_name} → {actual}={value}", flush=True)
            else:
                print(f"[fn_prepare_generation_args] Added: {arg_name}={value}", flush=True)

    # Strategy-level defaults (from strategy_ladder[i].extra_defaults)
    extra_defaults = config.get("extra_defaults", {})
    for k, v in extra_defaults.items():
        actual = resolve_arg_name(k)
        if actual:
            pipe_kwargs[actual] = v
            print(f"[fn_prepare_generation_args] Strategy default: {actual}={v}", flush=True)

    ctx.generation_args = pipe_kwargs
    print("[fn_prepare_generation_args] Final:", flush=True)
    for k, v in pipe_kwargs.items():
        if k in ("generator", "callback_on_step_end", "callback",
                 "prompt_embeds", "negative_prompt_embeds", "prompt_attention_mask", "negative_prompt_attention_mask"):
            print(f"[fn_prepare_generation_args]   {k}=<tensor>", flush=True)
        elif k in ("prompt", "negative_prompt"):
            print(f"[fn_prepare_generation_args]   {k}={repr(v)[:200]}", flush=True)
        else:
            print(f"[fn_prepare_generation_args]   {k}={v}", flush=True)


def fn_generate(ctx: PipelineContext, config: dict) -> None:
    import torch
    pipe = ctx.pipe
    gen_args = ctx.generation_args

    print("[fn_generate] Starting inference...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # No inline OOM recovery — the strategy ladder handles that at a higher level.
    # No inline OOM recovery — the strategy ladder handles that at a higher level.
    # Backend selection is handled globally in fn_optimize (flash off, math+mem_efficient on)
    result = pipe(**gen_args)

    if hasattr(result, "frames"):
        video = result.frames[0] if isinstance(result.frames, list) else result.frames
    elif hasattr(result, "images"):
        video = result.images
    elif isinstance(result, (list, tuple)):
        video = result[0] if len(result) == 1 else result
    else:
        video = result

    ctx.video_output = video
    print(f"[fn_generate] Done. Output: {type(video).__name__}", flush=True)


def fn_validate(ctx: PipelineContext, config: dict) -> None:
    import torch, numpy as np
    from PIL import Image
    frames = ctx.video_output
    val_cfg = get_validation_config(ctx.model_id)

    actual_count = len(frames) if isinstance(frames, (list, tuple)) else 1
    if actual_count == 0:
        raise RuntimeError("0 frames")
    if actual_count < val_cfg.get("min_frames", 8):
        raise RuntimeError(f"{actual_count} < min {val_cfg.get('min_frames', 8)}")

    first = frames[0] if isinstance(frames, (list, tuple)) else frames
    if isinstance(first, Image.Image):
        w, h = first.size
        if w < val_cfg.get("min_width", 256) or h < val_cfg.get("min_height", 256):
            raise RuntimeError(f"{w}x{h} below minimum")
        print(f"[fn_validate] {actual_count} PIL ({w}x{h})", flush=True)
    elif isinstance(first, np.ndarray):
        h, w = first.shape[:2]
        mn, mx = float(np.min(first)), float(np.max(first))
        if mx <= mn:
            raise RuntimeError(f"Blank: [{mn:.3f}, {mx:.3f}]")
        print(f"[fn_validate] {actual_count} numpy ({w}x{h})", flush=True)
    elif isinstance(first, torch.Tensor):
        if torch.isnan(first).any() or torch.isinf(first).any():
            raise RuntimeError("NaN/Inf")
        std_v = first.std().item()
        if std_v < val_cfg.get("min_std_dev", 0.005):
            raise RuntimeError(f"std {std_v:.4f} too low")
        print(f"[fn_validate] {actual_count} tensors, std={std_v:.3f}", flush=True)
    else:
        raise RuntimeError(f"Unknown type: {type(first)}")


def fn_export(ctx: PipelineContext, config: dict) -> None:
    import torch, numpy as np
    from PIL import Image
    from diffusers.utils import export_to_video

    frames = ctx.video_output
    fps = ctx.resolved_params.get("fps", 16)
    output_filename = config.get("output_filename", "video_{job_id}.mp4")
    if "{job_id}" in output_filename:
        output_filename = output_filename.format(job_id=ctx.job_id)
    output_path = OUTPUT_DIR / output_filename

    processed = []
    if len(frames) > 0 and isinstance(frames[0], torch.Tensor):
        for frame in frames:
            arr = frame.detach().cpu().numpy()
            if arr.dtype in (np.float32, np.float64):
                if arr.min() < 0:
                    arr = (arr + 1.0) / 2.0
                if arr.max() > 1.0:
                    arr = arr / (arr.max() + 1e-5)
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            processed.append(Image.fromarray(arr))
    else:
        processed = frames

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(processed, str(output_path), fps=fps)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Export failed")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[fn_export] {output_filename} ({size_mb:.2f}MB)", flush=True)
    ctx.video_output_path = str(output_path)


def fn_cleanup(ctx: PipelineContext, config: dict) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    print("[fn_cleanup] Done", flush=True)

def fn_patch_prepare_attention_mask(ctx: PipelineContext, config: dict) -> None:
    """
    Patches ltx_video.models.transformers.attention.Attention.prepare_attention_mask
    to fix an upstream bug where F.pad(mask, (0, target_length), value=0.0) pads BY
    target_length instead of TO target_length. The original code even has a TODO
    comment acknowledging this is wrong for cross-attn masks.

    The fix: pad by (target_length - current_length) when current_length < target_length,
    which is the semantically correct behavior for cross-attention padding masks.
    """
    import torch
    import torch.nn.functional as F
    import importlib

    try:
        attention_module = importlib.import_module("ltx_video.models.transformers.attention")
    except ImportError as e:
        print(f"[fn_patch_prepare_attention_mask] Could not import attention module: {e}", flush=True)
        return

    Attention = getattr(attention_module, "Attention", None)
    if Attention is None:
        print("[fn_patch_prepare_attention_mask] Attention class not found. Skipping.", flush=True)
        return

    if getattr(Attention, "_nc_prepare_mask_patched", False):
        print("[fn_patch_prepare_attention_mask] Already patched. Skipping.", flush=True)
        return

    def prepare_attention_mask_fixed(
        self,
        attention_mask,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length - current_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                remaining_length = target_length - current_length
                if remaining_length > 0:
                    attention_mask = F.pad(attention_mask, (0, remaining_length), value=0.0)
                elif remaining_length < 0:
                    attention_mask = attention_mask[..., :target_length]

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    Attention.prepare_attention_mask = prepare_attention_mask_fixed
    Attention._nc_prepare_mask_patched = True
    print("[fn_patch_prepare_attention_mask] Patched Attention.prepare_attention_mask", flush=True)

# ═════════════════════════════════════════════════════════════════════
# FUNCTION REGISTRY
# ═════════════════════════════════════════════════════════════════════

FUNCTION_REGISTRY: Dict[str, Callable] = {
    "load_job": fn_load_job,
    "ensure_dependencies": fn_ensure_dependencies,
    "load_pipeline": fn_load_pipeline,
    "encode_prompt_on_gpu": fn_encode_prompt_on_gpu,
    "wrap_vae_decode": fn_wrap_vae_decode,
    "apply_accelerate_offload": fn_apply_accelerate_offload,
    "neutralize_pipeline_device_moves": fn_neutralize_pipeline_device_moves,
    "disable_dynamic_shifting": fn_disable_dynamic_shifting,
    "post_load_cast": fn_post_load_cast,
    "quantize": fn_quantize,
    "patch_prepare_attention_mask": fn_patch_prepare_attention_mask,
    "configure_scheduler": fn_configure_scheduler,
    "configure_memory": fn_configure_memory,
    "set_execution_device": fn_set_execution_device,
    "encode_prompt_on_cpu": fn_encode_prompt_on_cpu,
    "optimize": fn_optimize,
    "prepare_generation_args": fn_prepare_generation_args,
    "generate": fn_generate,
    "validate": fn_validate,
    "export": fn_export,
    "cleanup": fn_cleanup,
}


# ═════════════════════════════════════════════════════════════════════
# STRATEGY LADDER + ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════

def _validate_pipeline_config(pipeline_config: dict) -> None:
    steps = pipeline_config.get("steps", [])
    if not steps:
        raise ValueError("Empty pipeline")
    for i, step in enumerate(steps):
        fn_name = step.get("fn")
        if not fn_name:
            raise ValueError(f"Step {i} missing 'fn'")
        if fn_name not in FUNCTION_REGISTRY:
            raise ValueError(f"Step {i} unknown fn '{fn_name}'")


def build_pipeline_steps_for_strategy(base_pipeline_config: dict, strategy: dict) -> list:
    """
    Merge a strategy definition into the base pipeline steps.

    A strategy can:
      1. Override the config of an existing step (steps_override[fn_name])
      2. Inject new steps before generation is set up (extra_steps_before_generate)

    IMPORTANT: extras are inserted before 'prepare_generation_args', not before
    'generate'. 'prepare_generation_args' is what actually reads ctx.precomputed_embeds
    (and any other context an extra step sets up) and bakes it into ctx.generation_args.
    If an extra step (e.g. encode_prompt_on_cpu) ran between prepare_generation_args and
    generate instead, its output would never make it into the pipe() call — the args are
    already frozen by then. We fall back to anchoring on 'generate' only if a given
    pipeline config doesn't define a 'prepare_generation_args' step at all, so this stays
    generic across any model JSON.
    """
    base_steps = base_pipeline_config.get("steps", [])
    overrides: dict = strategy.get("steps_override", {}) or {}
    extra_before_gen: list = strategy.get("extra_steps_before_generate", []) or []

    anchor_fn = "prepare_generation_args" if any(
        s["fn"] == "prepare_generation_args" for s in base_steps
    ) else "generate"

    merged = []
    for step in base_steps:
        fn_name = step["fn"]
        # Insert extras just before the anchor step (see docstring above)
        if fn_name == anchor_fn and extra_before_gen:
            for extra in extra_before_gen:
                merged.append(dict(extra))
        step_copy = dict(step)
        if fn_name in overrides:
            step_copy["config"] = {**step.get("config", {}), **overrides[fn_name]}
        merged.append(step_copy)

    return merged


def should_retry_verdict(verdict: dict, current_build: str) -> bool:
    """
    Option C: retry failed strategies if the sidecar build has changed since last test.
    """
    status = verdict.get("status")
    if status == "working":
        return True  # always trust the known-good
    if status == "untested":
        return True
    # failed_* — retry only if build changed
    return verdict.get("sidecar_build_last_tested") != current_build


def order_strategies(ladder: list, profile: dict, current_build: str) -> list:
    """
    Reorder the ladder based on prior knowledge:
      1. If a strategy is 'working' and hardware matches → try it FIRST, alone
      2. Otherwise walk the ladder in declared order, skipping strategies known
         to have failed on this exact build (unless build has since changed).
    """
    verdicts = profile.get("verdicts", {})
    preferred = profile.get("preferred_strategy")

    # Fast path: known-good strategy
    if preferred:
        v = verdicts.get(preferred, {})
        if v.get("status") == "working":
            for s in ladder:
                if s["name"] == preferred:
                    return [s] + [x for x in ladder if x["name"] != preferred]

    # General path: try each in declared order, skip if failed on this build
    ordered = []
    for s in ladder:
        v = verdicts.get(s["name"], {})
        if should_retry_verdict(v, current_build):
            ordered.append(s)
        else:
            print(f"[ORCHESTRATOR] Skipping '{s['name']}' — failed on this build ({v.get('status')})", flush=True)
    if not ordered:
        # All strategies previously failed on this build — retry them all anyway
        print("[ORCHESTRATOR] All strategies previously failed — retrying full ladder", flush=True)
        ordered = list(ladder)
    return ordered


def run_pipeline_with_strategy(ctx: PipelineContext, strategy: dict, base_pipeline_config: dict) -> None:
    """Execute the pipeline for a single strategy. Raises on failure."""
    strategy_name = strategy["name"]
    ctx.active_strategy = strategy_name
    print(f"\n{'='*70}", flush=True)
    print(f"[ORCHESTRATOR] Running strategy: '{strategy_name}'", flush=True)
    print(f"[ORCHESTRATOR] {strategy.get('description', '')}", flush=True)
    print(f"{'='*70}", flush=True)

    steps = build_pipeline_steps_for_strategy(base_pipeline_config, strategy)
    _validate_pipeline_config({"steps": steps})

    emit({
        "type": "job_status", "job_id": ctx.job_id,
        "status": "loading_model", "progress": 0.0, "eta": 30,
        "outputPath": None, "error": None,
        "strategy": strategy_name,
    })

    params_path = Path(sys.argv[1])
    for i, step in enumerate(steps):
        fn_name = step["fn"]
        fn_config = dict(step.get("config", {}))
        fn_config["params_path"] = params_path
        fn = FUNCTION_REGISTRY[fn_name]
        print(f"\n[ORCHESTRATOR] [{i+1}/{len(steps)}] {fn_name}", flush=True)
        fn(ctx, fn_config)

        if fn_name == "generate":
            emit({
                "type": "job_status", "job_id": ctx.job_id,
                "status": "post_processing", "progress": 90.0, "eta": 5,
                "outputPath": None, "error": None,
            })

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: generate_worker.py <params_json> [--strategy <name>]", file=sys.stderr)
        sys.exit(1)

    params_path = Path(sys.argv[1])

    # Parse --strategy arg
    strategy_name = None
    if "--strategy" in sys.argv:
        i = sys.argv.index("--strategy")
        if i + 1 < len(sys.argv):
            strategy_name = sys.argv[i + 1]

    ctx = PipelineContext()

    try:
        with open(params_path, "r") as f:
            raw_params = json.load(f)
        model_id = raw_params.get("model_id")
        if model_id not in MODEL_CONFIG:
            raise ValueError(f"Unknown model_id: {model_id}")

        model_cfg = MODEL_CONFIG[model_id]
        ctx.model_id = model_id
        ctx.model_cfg = model_cfg
        ctx.job_id = raw_params.get("job_id")

        runner_name = model_cfg.get("runtime", {}).get("runner")
        if runner_name:
            from runners import get_runner
            from runners.common import Reporter
            if model_cfg.get("identity", {}).get("kind") == "enhancer":
                params = dict(raw_params)  # source video + target size, nothing to resolve
            else:
                params = resolve_generation_params(model_id, raw_params)
            params["prompt"] = raw_params.get("prompt", "")
            params["negative_prompt"] = raw_params.get("negative_prompt", "")
            params["job_id"] = ctx.job_id
            for k in ("image_path", "seed", "start_image", "end_image", "super_resolution"):
                if raw_params.get(k) is not None:
                    params[k] = raw_params[k]
            output_path = get_runner(runner_name).run(params, model_cfg, Reporter(ctx.job_id))
            emit({
                "type": "job_status", "job_id": ctx.job_id,
                "status": "done", "progress": 100.0, "eta": 0,
                "outputPath": output_path, "error": None, "strategy": runner_name,
            })
            return

        base_pipeline_config = model_cfg.get("_pipeline_config") or {"steps": []}
        if not base_pipeline_config.get("steps"):
            raise ValueError(f"Model {model_id} has no _pipeline_config.steps")

        # Determine which strategy to run
        ladder = get_strategy_ladder(model_id)
        if strategy_name:
            strategy = next((s for s in ladder if s["name"] == strategy_name), None)
            if not strategy:
                raise ValueError(f"Unknown strategy: {strategy_name}. Available: {[s['name'] for s in ladder]}")
        else:
            # No --strategy given: use first in ladder
            strategy = ladder[0]

        # Step callback
        steps_count = raw_params.get("steps", 20)
        job_id = ctx.job_id

        def step_callback(pipe_obj, step, timestep, kwargs=None):
            progress = 10.0 + (step / max(1, steps_count)) * 80.0
            emit({
                "type": "job_status", "job_id": job_id,
                "status": "generating", "progress": round(progress, 1),
                "eta": max(1, int((steps_count - step) * 1.5)),
                "outputPath": None, "error": None,
            })
            return kwargs if kwargs is not None else {}
        ctx.step_callback = step_callback

        # Run the single strategy
        run_pipeline_with_strategy(ctx, strategy, base_pipeline_config)

        # Success
        emit({
            "type": "job_status", "job_id": job_id,
            "status": "done", "progress": 100.0, "eta": 0,
            "outputPath": ctx.video_output_path, "error": None,
            "strategy": strategy["name"],
        })
        print(f"[ORCHESTRATOR] Pipeline completed via '{strategy['name']}'", flush=True)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        if ctx.job_id:
            emit({
                "type": "job_status", "job_id": ctx.job_id,
                "status": "error", "progress": 0.0, "eta": 0,
                "outputPath": None, "error": str(e),
                "strategy": strategy_name,
            })
        sys.exit(1)

def serve() -> None:
    """Warm mode: pay the slow torch/diffusers import cost while idle, then
    wait for exactly one job on stdin ({"params_path": ..., "strategy": ...}).
    The process still exits after the job, so every job keeps a clean process."""
    import torch  # noqa: F401
    import diffusers  # noqa: F401
    from diffusers import (AutoencoderKLLTXVideo, AutoencoderKLWan, LTXPipeline,  # noqa: F401
                           LTXVideoTransformer3DModel, WanTransformer3DModel)
    import transformers  # noqa: F401
    emit({"type": "worker_ready"})
    line = sys.stdin.readline()
    if not line.strip():
        return  # parent closed stdin: shut down quietly
    job = json.loads(line)
    sys.argv = [sys.argv[0], job["params_path"], "--strategy", job["strategy"]]
    main()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        serve()
    else:
        main()