from huggingface_hub import snapshot_download
from model_registry import MODELS_DIR

# Download only the non-transformer parts (text encoder, VAE, tokenizer, scheduler)
# Exclude the transformer since we already have it
snapshot_download(
    repo_id="Lightricks/LTX-Video-0.9.8-13B-distilled",
    ignore_patterns=["*transformer*", "*.bin", "flax_model*"],
    local_dir=str(MODELS_DIR / "pipeline-0.9.8-distilled"),
)
print("Pipeline components downloaded")
