<p align="center"><img src="public/images/logo.png" width="96" alt="ThePipeline logo"></p>

# ThePipeline

ThePipeline is a local AI studio for Windows. It makes videos, images and voice-overs with open-source
models on your own NVIDIA GPU. A built-in agent plans the project, researches it, writes the script
and prompts, and then drives the generators for you.

- **Agent**: describe what you want (an ad, a short film, an explainer) and it makes it.
- **Video**: text-to-video and image-to-video (HunyuanVideo 1.5, MiniMax H3, Wan 2.2, LTX).
- **Images**: 2K stills (Z-Image Turbo) and cinematic camera moves through a photo.
- **Voices**: natural voice-overs (Qwen3-TTS VoiceDesign, Kokoro).
- **Editor**: trim, titles, logos, music and voice tracks, export.

## Quick start (developers)

You need Windows, Python 3.11, Node.js 20+ and an NVIDIA GPU with 8 GB or more of VRAM.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

Then see [CLAUDE.md](CLAUDE.md) for how to run it, how the code is organised, and how to add and test models.
