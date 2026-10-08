# The Pipeline: build plan

The project was formerly NeuralCut.

Goal: an all-in-one, fully local AI studio. You give the agent an idea, such as a series, an ad or a short. It plans the project, researches it, designs consistent characters, writes the script and dialogue, and prompts each shot. Then it unloads itself, and the image, voice and video models produce the assets. They are assembled in a real multi-track editor that has its own AI assistant.

## Principles
- **One heavy model at a time.** A resource manager owns the GPU. The LLM is stopped ("goes to sleep") before any generator runs, and vice versa. This is what makes it work on 8 GB.
- **Every step is resumable.** Projects are saved as JSON after each step, so a power cut never loses more than the step in progress.
- **Hardware-aware.** Every model has a tier. The Models page recommends the best model per role for this PC and never caps stronger PCs.
- **Local by default.** The only things that go online are model downloads and web research, which you can switch off in Settings.

## Runtimes
| Role | Runtime | Notes |
|---|---|---|
| Chat agent | `llama-server` from llama.cpp (pinned release, sha256 verified; CUDA 13 build, CPU fallback) | OpenAI-compatible API, tool calling (`--jinja`), vision through mmproj. Runs on a private loopback port, like ComfyUI. |
| Image | Private ComfyUI (already shipped) | Z-Image Turbo (text to image), FLUX.2 Klein 4B (multi-reference editing, for character consistency). |
| Voice | Python runner (`qwen-tts`, `kokoro`) | Qwen3-TTS VoiceDesign creates a voice from a description, and Base clones it for every line. Kokoro covers low-end PCs. |
| Video | Existing runners | H3 image-to-video and Wan TI2V take the character reference as the first frame. |
| Edit | ffmpeg (`editor.py`) | Extended to multiple tracks. |

## Model tiers (verified on HF, 2026-10-02)
- **Chat:**
  - Qwen3.5-4B Q4_K_M (2.7 GB plus 0.7 GB vision) for ≤6 GB VRAM.
  - Qwen3.5-9B Q4_K_M (5.7 + 0.9 GB) for 8 GB.
  - Gemma-4-26B-A4B or Qwen3.8-27B (~17 GB) for 16–24 GB.
  - Qwen3.6-35B-A3B (22 GB) for 24 GB+.
- **Image:**
  - Z-Image Turbo int8 (6.2 GB) plus Qwen3-4B fp4 text encoder (3.5 GB) plus VAE. Apache-2.0.
  - FLUX.2 Klein 4B (7.75 GB) plus Qwen3-4B fp4 (3.85 GB) plus VAE. Apache-2.0.
  - Anime-specialist checkpoint: later.
- **Voice:**
  - Kokoro-82M (0.3 GB, Apache-2.0).
  - Qwen3-TTS-12Hz-1.7B VoiceDesign and Base (4.5 GB each, Apache-2.0).

## Phases
1. **Platform.**
   - Model kinds: `chat`, `image`, `voice`, alongside the existing `video` and `enhancer`.
   - Hardware tiers and per-role recommendations.
   - Models page tabs.
   - A resource manager that loads one heavy model at a time.
   - App name: "The Pipeline". The data folder stays where it is, so nothing is re-downloaded.
2. **Agent chat.**
   - llama-server runtime.
   - Agent page: chat, attach images, streamed replies, tool calls shown inline.
   - Tools: web search (DuckDuckGo HTML, no key), read web page, and look at an image.
   - Sleep and wake.
3. **Images page.** Text to image, image editing with reference images, character sheets, and a gallery.
4. **Voices page.** Design a voice from a description, save voices, and turn script lines into audio.
5. **Production agent.**
   - The project lifecycle runs as follows:
     - **Brief:** you describe the idea.
     - **Plan:** the agent works out style (realistic, anime, …), picks models, researches, and writes characters, script, dialogue and the shot list.
     - **Review:** you approve the plan.
     - **Produce:** the agent sleeps while the queue makes character references, voices, then shots (image to video from the references).
     - **Assemble:** shots and voice lines go onto the editor timeline.
     - **Check:** the agent wakes and reviews frames with vision.
   - Ads: the agent builds the brief around an uploaded logo or product image, using it as a reference and as a logo overlay.
6. **Pro editor.**
   - Multi-track: video layers, voice, music and overlays.
   - Split at the playhead, cut, speed changes, text and logo layers with position and opacity.
   - An AI edit assistant that changes the timeline through tools.

## Status
- [x] Video generation (Wan 2.2 Fast, LTX, MiniMax H3), enhancers (SeedVR2, Real-ESRGAN), storyboard, single-track editor.
- [x] SeedVR2 upscales long clips in ~5 s segments (a 10 s clip at 1080p spilled 8 GB VRAM and ran 4x slower).
- [x] Engine work folders are cleaned up (log handle was left open) and swept on startup.
- [x] 2026-10-03 quality pivot: SeedVR2 judged useless; Wan 2.2 Fast at 720p (7 min, soft faces) and H3 (CG look)
      are not production-grade from text alone. Strategy is now image-first: photoreal keyframe (Z-Image Turbo), then
      HunyuanVideo 1.5 image-to-video (8-step distilled, 480p), then HY 720p super-resolution (6 steps, frame-consistent).
- [x] Models: `hunyuan15-i2v-hd` (production), `hunyuan15-hd` (T2V draft add-on, partial 8.3 GB .part kept),
      shared `hunyuan15-common` + `hunyuan15-upsampler-720p`; `z-image-turbo` (kind image). Runners `comfy_hunyuan15`,
      `comfy_image`. ComfyEngine supports extra model folders. Graph wiring mirrors the official ComfyUI templates.
- [x] Productions (`production.py`): agent hands a shot list to `start_production`; engine makes keyframes, animates
      them (start_image), assembles with crossfades; resumable; `/productions` endpoints.
- [x] Agent: llama.cpp runtime, conversations, tools (web_search, read_webpage, list_models, generate_video,
      generate_image, start_production, list_jobs), sleeps for GPU jobs. UI: Agent page, Models tabs with
      "Best for this PC", Images page (generate, upload, animate), app renamed "The Pipeline".
- [x] Pro editor: per-clip speed (0.25-4x, pitch-kept audio), volume/mute, split at playhead, overlay layers
      (titles drawn with Pillow, logos/images) with timing/position/size/opacity/fades, live layer preview.
- [ ] Test on GPU once downloads finish: agent chat, Z-Image keyframe, HY I2V + SR timing/quality, a full production.
- [ ] Voices (Qwen3-TTS VoiceDesign + Base, Kokoro), voice/music tracks in the editor, AI edit assistant.

## Paused 2026-10-07 (PC shut down by user)
- Canadian Shield Security ad: H3 version delivered (`outputs/video_canadian_shield_security_ad.mp4`).
- HD version with HunyuanVideo 1.5 I2V (81 frames + 720p SR, ~18-20 min/shot) in progress:
  done: s4_radio `video_79d2ed9f.mp4`, s1_site `video_82e8da99.mp4`. To redo (were queued, lost on shutdown):
  s2_patrol (start `image_777d808d_logo.png`), s3_gate (`image_d1891d1a_logo.png`), s6_final (`image_74ef74b8_logo.png`);
  prompts in scratchpad `ad_hy.json` script. Then assemble like the H3 cut, using H3 logo close-up `video_23c77f5c.mp4`
  and H3 shot audio muxed onto HY shots; end card `video_endcard_csecurity.mp4` (re-render at 1280x720).
- Lessons: HY 720p SR at 121 frames spills 8 GB (13 min/step); at 81 frames 82 s/step. HY decode ~7 min (to tune).
  Wan I2V works but over-saturates/smears. H3 I2V with first+last frame pins logos perfectly.

## 2026-10-08: image-led direction (user decision)
Local video generation on 8 GB can't match cloud free tiers (2+ h per 20 s, still worse), so the default is now
**image-led**: 2K photoreal stills (Z-Image Turbo, ~2 min each) -> photo motion (Depth Anything V2 Small parallax,
seconds per shot, full sharpness) -> real logo composited on clothing -> human voice-over -> editor. A 26 s 1080p ad
builds in ~1 min after the stills. Video models stay for shots that need real motion ("make an image move a certain way").
- Done: `runners/photo_motion.py` (+ `/images/{name}/animate`), voices: Kokoro 82M (CPU, presets) and Qwen3-TTS 1.7B
  VoiceDesign (GPU, describe a voice; separate `voice_venv` sharing torch, pins transformers 4.57.3),
  `/voices`, `/voices/speak`, `/audio` library, editor `audio` tracks, `logo_track.py` (tracked logo replacement
  for generated video), agent tools `speak` + `animate_image`. UI: Voices page, camera-move controls in the image viewer,
  editor audio track (synced preview, mixed in export).
- Lessons: "orbit" on people causes ghosting -> push/pull for people. Z-Image ignores negations ("no holster") -> don't name
  unwanted objects; frame shots to exclude them. Mixed H3 audio contains generated speech -> never use it as ambience.
- Next: logo placement tool in the UI (Images: "Put my logo on this"), agent productions in image-led mode (stills + motion +
  voice + assembly), music (ACE-Step / MiniMax Music), Qwen3-TTS Base for voice cloning/consistent cast, remote GPU /
  cloud connectors for full-motion shots, packaging (installer, voice_venv bootstrap).
