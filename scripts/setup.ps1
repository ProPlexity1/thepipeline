# ThePipeline: one-time developer setup on Windows.
# Creates the two Python environments the engine needs and installs the UI packages.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Requires: Python 3.11 (py launcher), Node.js 20+, an NVIDIA driver recent enough for CUDA 12.8.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sidecar = Join-Path $root "src-tauri\sidecar"
$cu = "https://download.pytorch.org/whl/cu128"

Write-Host "== UI packages" -ForegroundColor Cyan
Push-Location $root; npm install; Pop-Location

Write-Host "== Main engine venv" -ForegroundColor Cyan
$venv = Join-Path $sidecar "venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) { py -3.11 -m venv $venv }
$py = "$venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.11.0 torchvision==0.26.0 --index-url $cu
& $py -m pip install -r "$sidecar\requirements.txt"
& $py -m spacy download en_core_web_sm   # Kokoro's English phonemizer

Write-Host "== Voice venv (Qwen3-TTS)" -ForegroundColor Cyan
$vvenv = Join-Path $sidecar "voice_venv"
if (-not (Test-Path "$vvenv\Scripts\python.exe")) { py -3.11 -m venv $vvenv }
$vpy = "$vvenv\Scripts\python.exe"
# Share the main venv's torch instead of downloading it again.
"$venv\Lib\site-packages" | Out-File -Encoding ascii "$vvenv\Lib\site-packages\main_venv.pth"
& $vpy -m pip install --upgrade pip
& $vpy -m pip install qwen-tts==0.1.1 --no-deps
& $vpy -m pip install -r "$sidecar\requirements-voice.txt"
& $vpy -m pip install torchaudio==2.11.0 --index-url $cu --no-deps

Write-Host "== Check" -ForegroundColor Cyan
& $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $vpy -c "import qwen_tts, transformers; print('voice env ok, transformers', transformers.__version__)"
Write-Host "Done. See CLAUDE.md for how to run it." -ForegroundColor Green
