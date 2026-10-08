# NeuralCut work checkpoint (updated 2026-10-02)

Goal: a one-click local app that downloads and runs open models (video first;
later chat, image, audio and an agent that chains them).
Hardware target: RTX 5060 8GB VRAM, 16GB RAM.

All work below is uncommitted in the working tree.

## Working and verified on the GPU
- **LTX-Video 0.9.5 (`runners/ltx.py`)**
  - Fixed the noisy-video bug. Root cause: the 0.9.0 VAE plus a forced dynamic-shift scheduler.
    Weights now come from the 0.9.5 checkpoint, built with the official configs in `shared:ltx-095-configs`.
  - T5 text encoder converted to bf16 on disk (`optimize.py`): 18 GB → 8.9 GB.
  - VAE stays in RAM during denoising, then the transformer and VAE swap, and decode tiles in time.
    Before this, Balanced spilled VRAM and took 580 s.
  - Measured: Fast 512×320 ≈ 2 min, Balanced 704×480×121 ≈ 2.5 min (2.0 s/step).
- **Warm worker pool** (`main.py` `WarmPool` + `generate_worker.py --serve`): pre-imports torch, which saves about 40 s per job.
- **Engine**
  - Token auth, CORS and Host checks.
  - `/jobs` resync. The UI polls it while jobs are active, so a missed WebSocket event can't leave a job stuck.
  - `/outputs/{name}/file` streams video. The Tauri asset protocol is now disabled.
  - Cancel was tested: it kills the worker tree and frees VRAM.
- **UI** (tested in the browser using the dev-only mode in `store.ts`)
  - Setup, Generate, Models (storage panel, leftovers list, licence gate), Settings and toasts.
  - Run the browser dev mode with `VITE_DEV_PORT=47999 VITE_DEV_TOKEN=devtoken123 npx vite`, plus the engine with the same token and port.
- **MiniMax H3 (FastVideo FastH3 8-step) through a private headless ComfyUI** (`runners/comfy_h3.py`)
  - ComfyUI is pinned to 2d6b732 and installed in `models/comfyui-engine`.
  - Its dependencies are installed and listed in `requirements.txt`.
  - The API graph was validated by real ComfyUI (`node_errors: {}`) using placeholder weights.
  - The licence gate excludes the EU, UK, South Korea and the US.
  - **Verified on the GPU (2026-10-02):** Fast 608x352x124 (5.2 s with sound) took 270 s end to end:
    engine start ~55 s, prompt ~48 s, 8 steps x ~19 s, decode ~30 s. Excellent quality; real AAC soundtrack.
  - Story test (6 shots, seed 7): Balanced 864x480 5 s ≈ 6.5 min (33 s/step); Best 1024x576 5 s ≈ 9.7 min;
    Balanced 10 s (243 frames) ≈ 17.5 min, no OOM. Script: scratchpad `story_run.py`.
- **Editor** (`editor.py`, `/edit/render`, `components/EditorPanel.tsx`, nav "Editor")
  - Library, storyboard (multi-shot generation that fills the timeline as shots finish, incl. auto-enhance),
    preview with trim, drag-to-reorder timeline, export with cuts or crossfades, fade from/to black, size.
  - ffmpeg on the CPU in its own lane (never waits for GPU jobs). Mixed sizes are letterboxed; silent clips get silence.
  - `/outputs` now includes duration/size/has_audio (probed with PyAV, cached by mtime).
  - Verified: 3-clip crossfade export in 4 s; 6-shot 34 s story draft in seconds.
- Warm worker prewarm is skipped when free RAM < 6 GB.

- **Wan 2.2 Fast (`runners/wan_dmd.py`): verified on the GPU, excellent quality, now the recommended default**
  - 480p: ~78 s with a cached prompt, ~2 min the first time.
  - 720p: 154 s (was 250 s). The transformer adapts: weights stay resident in fp8 when they fit, otherwise blocks stream from RAM (3× faster than spilling).
  - VAE decodes in bf16 (1.7× faster, 42.7 dB PSNR versus fp32), with adaptive tile size (512 px tiles are 31% faster).
- **Prompt embedding cache** (`runners/common.py`, capped at 512 MB): re-rolling a prompt skips the text encoder.
  LTX Fast re-roll: 40 s, down from 268 s originally.
- **Engine watchdog in the UI**: restarts a dead engine automatically.
- **Real Tauri app tested**: random port and token, CSP, and the engine tree is stopped on exit.

- **UI telemetry**
  - Sidebar performance graphs: CPU, RAM, GPU and VRAM, collapsible.
  - Live job log in each job card.
  - On-this-PC time estimates (`telemetry.py`).
  - Sticky Length slider.
- **Downloads**
  - Resume automatically after an engine restart (`pending_downloads.json`).
  - Checkpoints are fsynced.
  - After a power cut, progress is recovered by scanning the partial file for unwritten regions; sha256 still verifies the result.
- **Enhancers (video upscaling)**
  - `realesrgan-sharpen` (runner `esrgan.py`, streamed frame by frame): verified working, 480p → 1080p in about 2 min.
  - `seedvr2-enhance` (runner `comfy_seedvr2.py`, ComfyUI native SeedVR2 3B int8): verified, excellent detail.
    480p → 1080p takes ~600 s per 5 s clip. The VAE decode is ~330 s of that, on ComfyUI's slow eager path because the build is cu128.
  - Auto-enhance chaining verified: one `/generate` with `enhancer_id` led to an enhanced copy in the gallery.
  - Engine: `/enhance`, plus auto-enhance after generation (`enhancer_id`/`enhance_target` on `/generate`).
  - UI: "Enhance after generating" selector, an Enhance button in the preview, and enhanced badges.
  - The ComfyUI engine class is shared in `runners/comfy_engine.py`.
- Note: ComfyUI's optimised CUDA kernels need torch cu130 (we ship cu128). The int8/nvfp4 models run on the slower eager path until torch is upgraded.

## In progress
- Downloads (resume automatically; scratchpad `dl.py`):
  - `wan22-fast` (24 GB): the transformer is done and verified, the shared text encoder and VAE are still downloading.
  - `minimax-h3-fast` (41 GB): done and sha256-verified.
  - The downloader caught a corrupt shard after a session crash (sha256). Fixed with flush-before-checkpoint and atomic checkpoint writes.
- Not yet GPU-tested: H3 Balanced/Best presets and longer clips.

## Pending decisions for the user
- Deleting the 65.7 GB of leftover model folders (shown in Models → "can be freed").
- Committing this work. Nothing is committed yet.
