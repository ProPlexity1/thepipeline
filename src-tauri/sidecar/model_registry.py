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
    _loaded_models = []
    for model_file in sorted(_MODELS_DIR.glob("*.json")):
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                model_data = json.load(f)
                identity = model_data.get("identity", {})
                model_id = identity.get("id")
                # Disabled entries stay on disk for reference but never reach the UI.
                if identity.get("status", "active") not in ("active", "beta"):
                    continue
                if model_id:
                    MODEL_CONFIG[model_id] = model_data
                    _loaded_models.append(model_id)
                else:
                    print(f"[REGISTRY] WARNING: {model_file.name} missing identity.id", flush=True)
        except Exception as e:
            print(f"[REGISTRY] ERROR loading {model_file.name}: {e}", flush=True)
    # Only print at DEBUG level for the sidecar; workers stay silent
    if os.environ.get("REGISTRY_VERBOSE") == "1":
        for mid in _loaded_models:
            print(f"[REGISTRY] Loaded model: {mid}", flush=True)
    else:
        print(f"[REGISTRY] Loaded {len(_loaded_models)} models", flush=True)

# Load all shared resources from shared_resources/ folder
SHARED_RESOURCES: Dict[str, Any] = {}
if _SHARED_DIR.exists():
    for shared_file in sorted(_SHARED_DIR.glob("*.json")):
        with open(shared_file, "r", encoding="utf-8") as f:
            shared_data = json.load(f)
            # Key by filename stem (e.g. ltx-shared-0.9.5.json -> ltx-shared-0.9.5)
            shared_key = shared_file.stem
            SHARED_RESOURCES[shared_key] = shared_data

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


def resolve_asset_path(directive: str) -> Path:
    """'shared:<key>/<sub>' or 'model:<id>/<sub>' -> absolute path under MODELS_DIR.
    Rejects anything that would escape MODELS_DIR."""
    if directive.startswith("shared:"):
        key, _, sub = directive[len("shared:"):].partition("/")
        if key not in SHARED_RESOURCES:
            raise ValueError(f"Unknown shared resource: {key}")
        base = MODELS_DIR / SHARED_RESOURCES[key]["local_dir"]
    elif directive.startswith("model:"):
        key, _, sub = directive[len("model:"):].partition("/")
        base = MODELS_DIR / key
    else:
        raise ValueError(f"Asset path must start with 'shared:' or 'model:': {directive}")
    path = (base / sub).resolve() if sub else base.resolve()
    if not path.is_relative_to(MODELS_DIR.resolve()):
        raise ValueError(f"Asset path escapes models dir: {directive}")
    return path


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

# ── Download manifest (handles shared_resources + github_archive extras) ─────

def shared_resource_keys(model: dict) -> list[str]:
    """distribution.shared_resources may be one key or a list of keys."""
    key = model.get("distribution", {}).get("shared_resources")
    if not key:
        return []
    return [key] if isinstance(key, str) else list(key)


def get_download_manifest(model_id: str) -> list[dict]:
    """Build download manifest. Each entry has a 'provider' key so the
    downloader can dispatch to the correct handler (huggingface / github_archive)."""
    model = get_model(model_id)
    dist = model["distribution"]
    manifest = []

    # 1. Main model files (always HuggingFace)
    for filename in dist["files"]["required"]:
        manifest.append({
            "provider": "huggingface",
            "repo_id": dist["repo"],
            "revision": dist.get("revision", "main"),
            "filename": filename,
            "local_path": f"{model_id}/{filename}",
            "shared": False,
        })

    # 2. Shared resources (HF files + optional github_archive extras)
    for shared_key in shared_resource_keys(model):
        if shared_key not in SHARED_RESOURCES:
            raise ValueError(
                f"Model {model_id} references unknown shared_resources '{shared_key}'"
            )
        shared = SHARED_RESOURCES[shared_key]
        local_dir = shared["local_dir"]

        # 2a. HuggingFace files listed in the shared resource
        hf_provider = shared.get("provider", "huggingface")
        if hf_provider == "huggingface":
            for filename in shared.get("files", []):
                manifest.append({
                    "provider": "huggingface",
                    "repo_id": shared["repo"],
                    "revision": shared.get("revision", "main"),
                    "filename": filename,
                    "local_path": f"{local_dir}/{filename}",
                    "shared": True,
                })

        # 2b. Extra sources (e.g. GitHub archive for pipeline source code)
        for extra in shared.get("extra_sources", []):
            provider = extra.get("provider")
            if provider == "github_archive":
                target_subdir = extra["target_subdir"]
                marker = extra.get("marker_file", "__init__.py")
                manifest.append({
                    "provider": "github_archive",
                    "repo": extra["repo"],
                    "revision": extra.get("revision", "main"),
                    "extract_path": extra["extract_path"],
                    "target_subdir": target_subdir,
                    # local_path points at the marker file, used for existence checks
                    "local_path": f"{local_dir}/{target_subdir}/{marker}",
                    "install_dir": f"{local_dir}/{target_subdir}",
                    "shared": True,
                    "description": extra.get("description", ""),
                })
            elif provider == "github_release":
                # Prebuilt binaries: a release asset zip, verified by GitHub's sha256
                # digest, extracted into target_subdir. A marker file records success.
                for asset in extra["assets"]:
                    manifest.append({
                        "provider": "github_release",
                        "repo": extra["repo"],
                        "tag": extra["tag"],
                        "asset": asset["name"],
                        "sha256": asset["sha256"],
                        "size": asset["size"],
                        "local_path": f"{local_dir}/{extra['target_subdir']}/.installed_{asset['name']}",
                        "install_dir": f"{local_dir}/{extra['target_subdir']}",
                        "shared": True,
                        "description": extra.get("description", ""),
                    })
            else:
                raise ValueError(f"Unknown extra_source provider: {provider}")

    return manifest


def check_downloaded(model_id: str, models_dir: Path) -> bool:
    manifest = get_download_manifest(model_id)
    for entry in manifest:
        if not (models_dir / entry["local_path"]).exists():
            return False
    ok, _ = validate_runtime_assets(model_id, models_dir)
    return ok


def get_runtime_model_dir(model_id: str, models_dir: Path = MODELS_DIR) -> Path:
    model = get_model(model_id)
    keys = shared_resource_keys(model)
    if keys:
        shared = SHARED_RESOURCES.get(keys[0])
        if shared:
            return models_dir / shared["local_dir"]
    return models_dir / model_id


def validate_runtime_assets(model_id: str, models_dir: Path = MODELS_DIR) -> tuple[bool, str]:
    model = get_model(model_id)
    runtime = model.get("runtime", {})

    # Honor JSON opt-out (used by custom pipelines whose class name won't match
    # the downloaded HF model_index.json — e.g. Lightricks' LTXVideoPipeline
    # against Lightricks/LTX-Video's model_index.json which says 'LTXPipeline').
    if runtime.get("skip_model_index_validation"):
        return True, "Runtime validation skipped by model JSON"

    # Prefer explicit expected class if the JSON declares one; fall back to pipeline_class.
    expected_pipeline = runtime.get("expected_model_index_class") or runtime.get("pipeline_class")

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

# ── Runtime profile persistence ──────────────────────────────────────────────
# Per-machine learned config: which strategy actually works for this GPU/OS combo.
# Never modifies the read-only model JSON — lives in %LOCALAPPDATA%.

RUNTIME_PROFILES_DIR = Path(LOCAL_APPDATA) / "NeuralCut" / "runtime_profiles"
RUNTIME_PROFILES_DIR.mkdir(parents=True, exist_ok=True)

ERROR_REPORTS_DIR = Path(LOCAL_APPDATA) / "NeuralCut" / "error_reports"
ERROR_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _empty_profile() -> dict:
    return {
        "hardware_fingerprint": None,
        "preferred_strategy": None,
        "verdicts": {},  # keyed by strategy_name
    }


def load_runtime_profile(model_id: str) -> dict:
    path = RUNTIME_PROFILES_DIR / f"{model_id}.json"
    if not path.exists():
        return _empty_profile()
    try:
        # utf-8-sig handles files with OR without BOM
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        for k, v in _empty_profile().items():
            data.setdefault(k, v)
        return data
    except Exception as e:
        print(f"[REGISTRY] Corrupt runtime_profile for {model_id}: {e}. Resetting.", flush=True)
        return _empty_profile()

def save_runtime_profile(model_id: str, profile: dict) -> None:
    path = RUNTIME_PROFILES_DIR / f"{model_id}.json"
    tmp = path.with_suffix(".json.tmp")
    try:
        # Explicit utf-8 (no BOM) — Python's default is fine but be explicit
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    except Exception as e:
        print(f"[REGISTRY] Failed to save runtime_profile for {model_id}: {e}", flush=True)

def save_error_report(model_id: str, strategy_name: str, report: dict) -> Path:
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{stamp}_{model_id}_{strategy_name}.json"
    path = ERROR_REPORTS_DIR / fname
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass
    return path


def get_strategy_ladder(model_id: str) -> list[dict]:
    """Return the strategy_ladder declared in the model JSON, or a single default."""
    model = get_model(model_id)
    ladder = model.get("strategy_ladder")
    if ladder:
        return ladder
    # Fallback: one implicit strategy that does nothing (uses model's _pipeline_config as-is)
    return [{
        "name": "default",
        "description": "Default execution — no overrides",
        "steps_override": {},
        "extra_steps_before_generate": [],
    }]