import json
import os
from pathlib import Path
from typing import Dict, Any

# ── Registry loading (VMR v2 folder-based) ────────────────────────────────────
# Each model and shared resource is its own JSON file in video_models/
# Nothing hardcoded — everything comes from folder structure.

_REGISTRY_DIR = Path(__file__).parent / "video_models"
_METADATA_PATH = _REGISTRY_DIR / "metadata.json"
_MODELS_DIR = _REGISTRY_DIR / "models"
_SHARED_DIR = _REGISTRY_DIR / "shared_resources"

# Load metadata
with open(_METADATA_PATH, "r", encoding="utf-8") as f:
    _METADATA = json.load(f)

VMR_METADATA = _METADATA.get("vmr_metadata", {})
FAMILIES = _METADATA.get("families", {})

# Load all models from models/ folder
MODEL_CONFIG: Dict[str, Any] = {}
if _MODELS_DIR.exists():
    for model_file in sorted(_MODELS_DIR.glob("*.json")):
        with open(model_file, "r", encoding="utf-8") as f:
            model_data = json.load(f)
            model_id = model_data.get("identity", {}).get("id")
            if model_id:
                MODEL_CONFIG[model_id] = model_data
                print(f"[REGISTRY] Loaded model: {model_id}", flush=True)
            else:
                print(f"[REGISTRY] WARNING: {model_file.name} missing identity.id", flush=True)

# Load all shared resources from shared_resources/ folder
SHARED_RESOURCES: Dict[str, Any] = {}
if _SHARED_DIR.exists():
    for shared_file in sorted(_SHARED_DIR.glob("*.json")):
        with open(shared_file, "r", encoding="utf-8") as f:
            shared_data = json.load(f)
            # Key by filename stem (e.g. ltx-shared-0.9.5.json -> ltx-shared-0.9.5)
            shared_key = shared_file.stem
            SHARED_RESOURCES[shared_key] = shared_data
            print(f"[REGISTRY] Loaded shared resource: {shared_key}", flush=True)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_APPDATA = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
DEFAULT_MODELS_DIR = Path(LOCAL_APPDATA) / "NeuralCut" / "models"

if os.environ.get("MODELS_DIR"):
    MODELS_DIR = Path(os.environ["MODELS_DIR"])
elif DEFAULT_MODELS_DIR.exists():
    MODELS_DIR = DEFAULT_MODELS_DIR
elif (PROJECT_ROOT / "models").exists():
    MODELS_DIR = PROJECT_ROOT / "models"
else:
    MODELS_DIR = DEFAULT_MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUTPUT_DIR = Path(LOCAL_APPDATA) / "NeuralCut" / "outputs"
if os.environ.get("OUTPUT_DIR"):
    OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
elif (PROJECT_ROOT / "outputs").exists():
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
else:
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_model(model_id: str) -> dict:
    if model_id not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_CONFIG.keys())}")
    return MODEL_CONFIG[model_id]


def get_profiles(model_id: str) -> dict:
    model = get_model(model_id)
    return model.get("profiles", {})


def get_scheduler_config(model_id: str) -> dict:
    model = get_model(model_id)
    return model.get("scheduler", {
        "default": "FlowMatch",
        "class_name": "FlowMatchEulerDiscreteScheduler",
        "params": {}
    })


def get_quantization_config(model_id: str) -> dict:
    model = get_model(model_id)
    return model.get("quantization", {
        "default": "none",
        "supported": ["none", "fp16", "bf16", "fp8", "auto"]
    })


def get_validation_config(model_id: str) -> dict:
    model = get_model(model_id)
    return model.get("validation", {
        "min_frames": 8,
        "min_width": 256,
        "min_height": 256,
        "check_nan": True,
        "check_black_frames": True,
        "min_std_dev": 0.005,
        "max_std_dev": 0.45
    })


# ── Settings clamping & parameter resolution ─────────────────────────────────

def clamp_settings(model_id: str, requested: dict) -> dict:
    """Take whatever the frontend sent and force every value onto the
    model's own allowed slider range (generation.limits). A modified
    client, a stale UI, or a raw API call can never push a model outside
    settings it actually supports."""
    limits = get_model(model_id)["generation"]["limits"]
    resolved = {}
    for key, spec in limits.items():
        value = requested.get(key)
        if value is None:
            value = spec["default"]
        step = spec.get("step", 1)
        lo, hi = spec["min"], spec["max"]
        # snap to nearest valid step, then clamp to range
        if step > 0:
            snapped = round((value - lo) / step) * step + lo
        else:
            snapped = value
        resolved[key] = max(lo, min(hi, snapped))
    return resolved


def resolve_generation_params(model_id: str, requested: dict) -> dict:
    """Resolves parameter dictionary taking into account preset profiles
    ('fast', 'balanced', 'detailed') and advanced overrides."""
    model = get_model(model_id)
    profiles = model.get("profiles", {})
    selected_profile_name = requested.get("profile") or "balanced"

    # Start with profile defaults if present, else fallback to model defaults
    if selected_profile_name in profiles:
        base_params = dict(profiles[selected_profile_name])
    else:
        base_params = dict(model.get("generation", {}).get("defaults", {}))

    # Apply explicit overrides from request if provided and not None
    for key in ("steps", "cfg_scale", "width", "height", "num_frames", "fps", "seed", "scheduler", "quantization"):
        if requested.get(key) is not None:
            base_params[key] = requested[key]

    # Clamp numeric parameters to model limits
    clamped = clamp_settings(model_id, base_params)
    base_params.update(clamped)
    base_params["profile"] = selected_profile_name
    return base_params


# ── Offload strategy ──────────────────────────────────────────────────────────

def resolve_offload_strategy(model_id: str, free_vram_gb: float) -> str:
    """'auto' picks sequential vs model-level offload vs GPU based on what's
    actually free on this GPU right now."""
    model = get_model(model_id)
    strategy = model.get("optimization", {}).get("offload_strategy", "auto")
    if strategy != "auto":
        return strategy
    # If free VRAM is 5.5GB or higher, run directly on GPU for maximum speed
    # (fn_generate will automatically fall back to model offloading if CUDA OOM occurs)
    if free_vram_gb >= 5.5:
        return "gpu"
    elif free_vram_gb >= 4.0:
        return "model"
    else:
        return "sequential"


# ── Download manifest (handles shared_resources, e.g. LTX's shared bundle) ────

def get_download_manifest(model_id: str) -> list[dict]:
    model = get_model(model_id)
    dist = model["distribution"]
    manifest = []

    for filename in dist["files"]["required"]:
        manifest.append({
            "repo_id": dist["repo"],
            "revision": dist.get("revision", "main"),
            "filename": filename,
            "local_path": f"{model_id}/{filename}",
            "shared": False,
        })

    shared_key = dist.get("shared_resources")
    if shared_key:
        if shared_key not in SHARED_RESOURCES:
            raise ValueError(
                f"Model {model_id} references unknown shared_resources '{shared_key}'"
            )
        shared = SHARED_RESOURCES[shared_key]
        local_dir = shared["local_dir"]
        for filename in shared["files"]:
            manifest.append({
                "repo_id": shared["repo"],
                "revision": shared.get("revision", "main"),
                "filename": filename,
                "local_path": f"{local_dir}/{filename}",
                "shared": True,
            })

    return manifest


def check_downloaded(model_id: str, models_dir: Path) -> bool:
    manifest = get_download_manifest(model_id)
    if not all((models_dir / entry["local_path"]).exists() for entry in manifest):
        return False
    ok, _ = validate_runtime_assets(model_id, models_dir)
    return ok


def get_runtime_model_dir(model_id: str, models_dir: Path = MODELS_DIR) -> Path:
    model = get_model(model_id)
    shared_key = model.get("distribution", {}).get("shared_resources")
    if shared_key:
        shared = SHARED_RESOURCES.get(shared_key)
        if shared:
            return models_dir / shared["local_dir"]
    return models_dir / model_id


def validate_runtime_assets(model_id: str, models_dir: Path = MODELS_DIR) -> tuple[bool, str]:
    model = get_model(model_id)
    runtime = model.get("runtime", {})
    expected_pipeline = runtime.get("pipeline_class")
    runtime_dir = get_runtime_model_dir(model_id, models_dir)
    model_index_path = runtime_dir / "model_index.json"

    if not model_index_path.exists():
        if runtime.get("requires_model_index", True):
            return False, f"Missing runtime metadata: {model_index_path}"
        return True, "Runtime metadata not required"

    try:
        with open(model_index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
    except Exception as e:
        return False, f"Could not read runtime metadata: {e}"

    native_pipeline = model_index.get("_class_name")
    if expected_pipeline and native_pipeline and native_pipeline != expected_pipeline:
        return (
            False,
            f"Runtime pipeline mismatch: registry={expected_pipeline}, downloaded={native_pipeline}",
        )

    return True, "Runtime assets validated"
