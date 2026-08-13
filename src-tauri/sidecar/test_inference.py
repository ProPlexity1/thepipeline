import torch
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
from diffusers.utils import export_to_video

from model_registry import MODELS_DIR

print("Loading transformer from local file...", flush=True)
transformer = LTXVideoTransformer3DModel.from_single_file(
    str(MODELS_DIR / "ltx-video-ultra" / "ltxv-13b-0.9.8-distilled-fp8.safetensors"),
    torch_dtype=torch.float8_e4m3fn,
)

print("Loading pipeline from local components...", flush=True)
pipe = LTXPipeline.from_pretrained(
    str(MODELS_DIR.parent / "pipeline-0.9.8-distilled"),
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    local_files_only=True,  # ← forces no downloads
)

pipe.enable_model_cpu_offload()
print("Pipeline ready, generating...", flush=True)

video = pipe(
    prompt="a cat walking across a sunny garden, cinematic lighting, photorealistic",
    negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
    num_frames=81,
    width=704,
    height=480,
    num_inference_steps=8,
    guidance_scale=1.0,
).frames[0]

export_to_video(video, "test_output_ultra.mp4", fps=24)
print("Done — check test_output_ultra.mp4", flush=True)