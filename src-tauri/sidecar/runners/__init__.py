"""Model runners.

A runner owns the whole load -> generate -> export cycle for one model family,
including how components are staged through limited RAM/VRAM. Model JSONs opt
in with `runtime.runner`; everything else keeps using the step-function worker.

Every runner exposes:
    run(params: dict, model_cfg: dict, report: Reporter) -> str  (output path)
"""
import importlib

RUNNERS = {
    "wan_dmd": "runners.wan_dmd",
    "ltx": "runners.ltx",
    "comfy_minimax_h3": "runners.comfy_h3",
    "comfy_seedvr2": "runners.comfy_seedvr2",
    "esrgan": "runners.esrgan",
    "comfy_hunyuan15": "runners.comfy_hunyuan15",
    "comfy_image": "runners.comfy_image",
}


def get_runner(name: str):
    if name not in RUNNERS:
        raise ValueError(f"Unknown runner '{name}'. Known: {sorted(RUNNERS)}")
    return importlib.import_module(RUNNERS[name])
