# ThePipeline: guide for Claude

You're working on **ThePipeline**, a Windows desktop app: a local, all-in-one AI studio. A local
chat agent plans projects and drives open-source video, image and voice models running on the
user's own NVIDIA GPU. Nothing goes to the cloud except optional web research.

This file gets you up to speed: what the app does, how it's built, how to set it up and run it,
how to add and test models, and what we've already learned the hard way. Read all of it before
changing anything.

> Repo: https://github.com/ProPlexity1/thepipeline (formerly `neuralcut`). Old names still work
> on purpose inside the code (see "Renaming" below).

---

## 1. What the app does

Sidebar pages, in order:

| Page | What it does |
|---|---|
| **Agent** | Chat with a local LLM (llama.cpp, e.g. Qwen3.5-9B with vision). It researches the web, writes scripts and shot lists, and calls app tools: `generate_image`, `generate_video`, `animate_image`, `speak`, `start_production`, `list_models`, `list_jobs`, `web_search`, `read_webpage`. Generated images, videos and voice lines appear inline in the chat. |
| **Video** | Text-to-video and image-to-video with any installed video model, plus a job queue and gallery. |
| **Images** | Generate stills (Z-Image Turbo), upload images, and **photo motion**: a cinematic 2.5D camera move through a still. |
| **Voices** | Text to speech: Kokoro 82M (CPU, 54 preset voices) and Qwen3-TTS VoiceDesign (a natural human voice from a text description). |
| **Editor** | Timeline editor (ffmpeg): trim, split, speed, volume, crossfades, title and logo overlays, audio tracks, export. |
| **Models** | Download and manage models by kind (video, image, voice, chat, tool). Shows "Best for this PC" from detected hardware, but also lists models above the PC's limits. |
| **Settings** | Web research on/off, optional Brave Search API key, agent model. |

### Productions (the main feature)

`start_production` turns an approved plan into a finished video with no further input:

- **mode "image"** (default; fast and sharp):
  1. One 2K keyframe per shot (Z-Image Turbo, about 2 min each on an 8 GB card).
  2. One continuous voice-over.
  3. Shot lengths are fitted to the voice.
  4. Each still gets a photo-motion camera move.
  5. Everything is assembled with the voice, a logo watermark and an exact-text end card.
- **mode "video"**: keyframes are animated by an image-to-video model (real motion, much slower).

State is saved after every step in `%LOCALAPPDATA%\ThePipeline\productions\*.json`. Productions resume after a restart. When one finishes, a message with a playable video is posted in the chat that started it.

### GPU arbitration (important on 8 GB cards)

- The agent's LLM and the generators can't share VRAM.
- When a GPU job starts, the agent finishes its current reply, then the LLM unloads ("sleeps").
- A new message to the agent waits visibly ("Waiting for the graphics card…") and continues by itself when the GPU is free.
- On a 16 GB card this still applies; it just hurts less.

---

## 2. Architecture

```
src/                      React 19 + Vite + Tailwind 4 UI (TypeScript)
  App.tsx, store.ts       app state, WebSocket events, API calls
  api.ts                  fetch helpers; auth header x-pipeline-token
  components/*Panel.tsx   one file per page (AgentPanel, GeneratePanel, ImagesPanel, ...)
src-tauri/                Tauri 2 shell (Rust). Starts the Python engine with a random port + token.
  src/lib.rs              sidecar launch: env PIPELINE_TOKEN, PIPELINE_LOG_DIR, SIDECAR_PORT
  sidecar/                the Python engine (FastAPI, Python 3.11)
    main.py               HTTP/WebSocket API, job queues, app tools for the agent, productions glue
    agent.py              agent turns, tools, web search chain, system prompt
    llm.py                llama-server lifecycle (sleep/wake, GPU handshake)
    production.py         production state machine (image/video modes)
    editor.py             ffmpeg timeline renderer, end cards, overlays
    generate_worker.py    generic diffusers worker (older models)
    model_registry.py     loads model JSONs, paths, hardware fit
    downloader.py         resumable, verified Hugging Face / GitHub downloads
    storage.py            output/image/audio paths
    logo_track.py         tracks a logo patch in generated video and re-composites the real logo
    runners/              one module per model family: run(params, cfg, report) -> output path
      comfy_engine.py     private headless ComfyUI (pinned commit) on a loopback port
      comfy_hunyuan15.py  HunyuanVideo 1.5 (t2v, i2v, 720p super-resolution)
      comfy_image.py      Z-Image Turbo
      comfy_h3.py         MiniMax H3 (has audio; first/last frame)
      wan_dmd.py, ltx.py  Wan 2.2 / LTX via diffusers
      photo_motion.py     depth-based 2-layer parallax (Depth Anything V2 Small)
      kokoro_tts.py, qwen_tts_runner.py   voices
    video_models/models/*.json            one JSON per model (see section 5)
    video_models/shared_resources/*.json  files shared between models (ComfyUI, encoders, llama.cpp)
```

**Queues:**
- **Generation queue (GPU, one at a time):** video, image, enhance and Qwen voice jobs.
- **Light queue:** Kokoro voices and photo motion.

Every job reports progress over the WebSocket (`job_status`, `job_log`, `job_created`).

**Data locations:**
- Models: `%LOCALAPPDATA%\ThePipeline\models`. If an install from before the rename exists, `%LOCALAPPDATA%\NeuralCut` is used instead.
- Outputs: `<repo>/outputs/` if that folder exists, otherwise `%LOCALAPPDATA%\ThePipeline\outputs`. Override with `MODELS_DIR` and `OUTPUT_DIR`.

---

## 3. Setup on a new Windows PC

Prerequisites:
- Python **3.11** (`py -3.11`)
- Node.js 20+
- Rust (only for the desktop build)
- An NVIDIA driver with CUDA 12.8 support
- Git
- About 100 GB free disk for a useful set of models

```powershell
git clone <repo-url> thepipeline
cd thepipeline
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

`setup.ps1` does the following:
- `npm install`
- Creates `src-tauri/sidecar/venv` with torch 2.11 + cu128 (required for RTX 50-series / Blackwell), installs `requirements.txt`, and downloads the spaCy English model.
- Creates `src-tauri/sidecar/voice_venv` for Qwen3-TTS. Qwen pins `transformers==4.57.3`, so it can't share the main venv. It borrows the main venv's torch through a `.pth` file.

Neither venv is committed to git. If `setup.ps1` fails part-way, read the error and fix it; don't skip steps.

---

## 4. Running it

### Dev mode (what you'll use for testing)

Run the engine and the UI separately. That way you can restart the engine and read its log.

```powershell
# terminal 1: engine
cd src-tauri\sidecar
$env:PIPELINE_TOKEN="devtoken123"; $env:SIDECAR_PORT="47999"
venv\Scripts\python.exe -u main.py

# terminal 2: UI in a browser (http://localhost:5173)
$env:VITE_DEV_PORT="47999"; $env:VITE_DEV_TOKEN="devtoken123"
npx vite --port 5173 --strictPort
```

- `.claude/launch.json` already defines the UI as the `ui-dev` preview.
- Every API call needs the header `x-pipeline-token: devtoken123`, or `?token=devtoken123` for media URLs. `/health` is open.
- Restart the engine after any Python change. Restarting drops queued jobs, but productions resume.

### Desktop app

`npx tauri dev` runs it in a desktop window, and the Rust shell starts the engine itself. `npx tauri build` makes the installer. The installer doesn't bundle the venvs yet (open item).

### Driving it from the API (handy for tests)

```bash
H='x-pipeline-token: devtoken123'
curl -s -H "$H" localhost:47999/models | head                     # all models + install state
curl -s -X POST -H "$H" localhost:47999/models/download/z-image-turbo
curl -s -X POST -H "$H" -H 'content-type: application/json' localhost:47999/generate \
  -d '{"model_id":"z-image-turbo","prompt":"...","profile":"detailed","width":1920,"height":1088}'
curl -s -H "$H" localhost:47999/jobs                               # {job_id: status}
# outputs: outputs/video_<job>.mp4, outputs/images/image_<job>.png, outputs/audio/voice_<job>.wav
```

Other routes:
- `/images/{name}/animate` (photo motion)
- `/voices/speak`
- `/edit/render`
- `/productions`
- `/agent/conversations/{id}/message`
- `/estimate`
- `/system/stats`, `/gpu/stats`

---

## 5. Models

Each model is a JSON file in `src-tauri/sidecar/video_models/models/`. Key sections:

- `identity`: `id`, `display_name`, `kind` (video | image | voice | chat | tool | enhancer), `status`.
- `capabilities`: `text_to_video`, `image_to_video`, `voice_design`, and so on. The UI and agent read these.
- `distribution`: HF `repo` + pinned `revision`, `files.required`, `shared_resources`, `license`. Downloads are resumable and verified.
- `license_gate`: shown to the user before downloading.
- `runtime.runner`: which module in `runners/` runs it, plus runner-specific paths and settings. `model:` and `shared:` path prefixes resolve to the models folder.
- `hardware`: `minimum_vram_gb`, `recommended_vram_gb`, and RAM equivalents. Used for "Best for this PC". Models above the PC's limits are still listed.
- `generation.defaults/limits`, `profiles` (fast/balanced/detailed), `ui`.

### Adding a model

1. Copy the closest existing JSON. Pin the HF revision and list exactly the files needed.
2. If it runs on ComfyUI, write the API-format graph in a new runner, modelled on `runners/comfy_hunyuan15.py` or `comfy_image.py`. Use the official ComfyUI example workflow for that model as the source of truth for node names and settings. Register it in `runners/__init__.py`.
3. Download it through the Models page or `POST /models/download/{id}`.
4. Generate with it, open the output and look at frames (extract a few with ffmpeg and view them). Write down speed, peak VRAM and quality in `PIPELINE_PLAN.md`.

### Installed and tested on the original dev PC (RTX 5060 8 GB, 16 GB RAM)

| Model | Kind | Result on 8 GB |
|---|---|---|
| Z-Image Turbo | image | **Good.** 1920×1088 in ~2 min. The workhorse for keyframes. |
| HunyuanVideo 1.5 HD (t2v + i2v) | video | 480p, 8 steps, then 720p super-resolution. **81 frames ≈ 18 min/shot.** 121 frames spills out of 8 GB VRAM (13 min/step). Good real motion. |
| MiniMax H3 fast | video | Has audio, first/last-frame control. Slow on 8 GB; its audio includes invented speech (don't use it under a voice-over). |
| Wan 2.2 TI2V 5B (fast/DMD) | video | Works; image-to-video comes out over-saturated and smeared. Not recommended. |
| LTX-Video 0.9.x | video | Works; low quality. |
| Kokoro 82M | voice | Fast, CPU. Sounds synthetic. |
| Qwen3-TTS 1.7B VoiceDesign | voice | **Sounds human.** Use a description + fixed seed for a consistent voice. |
| Depth Anything V2 Small | tool | Photo motion depth. |
| Qwen3.5-9B (llama.cpp Q4) | chat | Agent brain with vision. |
| SeedVR2 / Real-ESRGAN | enhancer | Upscaling didn't help (artifacts in, artifacts out). Don't bother. |

### What to test on a 16 GB card (e.g. RTX 5080)

The point of testing on a bigger card is to find the setups an 8 GB card can't run:

- **HunyuanVideo 1.5 at 121 frames** (5 s). Profiles in `hunyuan15-i2v-hd.json` were capped at 81 frames for 8 GB. Try 121, and check speed and that VRAM doesn't spill (`/gpu/stats` or `nvidia-smi`).
- **Larger video models not yet added:**
  - Wan 2.2 14B t2v/i2v (fp8 in ComfyUI)
  - LTX-2.3 distilled
  - HunyuanVideo 1.5 at 720p native
- **Image editing / character consistency:** FLUX.2 Klein 4B, Qwen-Image-Edit.
- **Voice cloning:** Qwen3-TTS Base. **Music:** ACE-Step.
- The 27B chat models (`chat-qwen38-27b`, `chat-gemma4-26b-a4b`) as the agent brain.
- Mark each model JSON with what you found, e.g. `hardware.notes`, so the Models page tells users the truth.

---

## 6. Lessons already learned (don't repeat these)

- **Fix quality at generation time.** Upscalers and enhancers didn't rescue weak video.
- **Image-led beats video-led** for ads and explainers on consumer GPUs: sharp 2K stills plus camera moves, with video used only for a few action shots. A 27 s ad: ~16 min of stills plus ~17 min per video shot, then about 1 min to assemble.
- **Never ask a generator for text, logos, phone numbers or addresses.** They come out garbled. Composite the real logo afterwards:
  - Stills: paste it onto the patch.
  - Video: use `logo_track.py`.
  - Exact text: the end card (`editor.make_end_card`).
- **Negative words backfire.** "No holster" draws a holster. Describe only what should appear; frame above the belt or crop if gear appears.
- **Photo motion:** the two-layer renderer (subject cut out by depth, background inpainted, layers moved separately) is what removed ghosting and cropping. Wide shots with no clear subject fall back to a clean 2D move.
- **One voice per video.** MiniMax H3's generated audio contains speech; don't mix it under a voice-over.
- **ComfyUI graphs:** follow the official example workflows exactly (sampler, shift, CFG, steps). Small deviations wrecked quality.
- **Windows:** never let a worker hang on `from_pretrained` in a subprocess with a pipe that nobody drains. Runners stream logs line by line.

---

## 7. Renaming (NeuralCut → ThePipeline)

The app was called NeuralCut, then "The Pipeline", now **ThePipeline**. Compatibility shims, so don't remove them:

- The engine accepts `PIPELINE_TOKEN` or `NEURALCUT_TOKEN`, and `PIPELINE_LOG_DIR` or `NEURALCUT_LOG_DIR`.
- It accepts the headers `x-pipeline-token` and `x-neuralcut-token`.
- The data folder falls back to `%LOCALAPPDATA%\NeuralCut` when it exists.
- `optimize.py` reads the old `.neuralcut_converted.json` marker.
- `main.tsx` migrates `neuralcut.*` localStorage keys to `thepipeline.*`.

---

## 8. Working rules

- **Don't delete model files or outputs without asking the user.** Downloads are tens of GB.
- **Every change goes through a pull request; never push to `main`.** The maintainers review PRs and merge them. See section 9.
- After UI changes, run `npx tsc --noEmit -p .`. After Python changes, restart the engine and exercise the change through the API.
- **Look at results before calling them good:** extract frames, view images, measure audio. Generated media fails in ways logs don't show.
- **Long jobs:** queue them, then wait on the output file or `/jobs` instead of sleeping blindly.
- **Open items:**
  - The installer should bootstrap the venvs.
  - Add more models (see the 16 GB test list).
  - Manga PDF → animated motion-comic clip: split pages into panels (OpenCV, right-to-left reading order), have the vision LLM read each panel, then photo motion + voices + assembly.
  - Music.
  - Voice cloning.

---

## 9. Sending changes back (pull requests)

If you change anything (code, model JSONs, docs, test notes), send it back as a pull request to
`ProPlexity1/thepipeline`. The maintainers review it, merge it into `main`, and bring it into their
own install.

1. Start from the latest `main`: `git checkout main && git pull`.
2. Make a branch named for the work, e.g. `models/wan22-14b` or `fix/hunyuan-121-frames`.
3. Commit in small, clear steps. Commit messages are in plain English and say what changed and why.
4. Push and open the PR:
   - If you can push to the repo: `git push -u origin <branch>`, then `gh pr create --base main`.
   - If you can't (the usual case for a tester), fork it first: `gh repo fork --remote`, push the
     branch to the fork, then `gh pr create --repo ProPlexity1/thepipeline --base main`.
5. Write the PR description for a reviewer who wasn't there:
   - what changed and why
   - the test PC (GPU, VRAM, RAM, driver)
   - what you ran
   - results: speed per clip, peak VRAM, and a short quality verdict for each model tested
   - anything that failed or still needs work
6. Never push to `main`, force-push over someone else's work, or merge your own PR.

Don't put model weights, venvs, outputs, logs or API keys in a PR (`.gitignore` covers most of these;
check `git status` before committing). To share sample outputs, describe them in the PR or attach a
few frames as images.
