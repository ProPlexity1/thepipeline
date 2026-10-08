import { useState, useCallback, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import type {
  GPUInfo,
  ModelInfo,
  GenerationJob,
  AppView,
  SidecarStatus,
  LicenseInfo,
  GenerationStatus,
  OutputItem,
  ImageItem,
  AudioItem,
  StorageReport,
  SystemInfo,
} from './types';
import { api, post, del, configureApi, healthy, wsUrl, apiReady } from './api';
import type { RenderPayload } from './components/EditorPanel';

/** Agent events from the WebSocket, fanned out to whoever is listening (the Agent page). */
type AgentListener = (msg: any) => void;
const agentListeners = new Set<AgentListener>();
export function onAgentEvent(fn: AgentListener) {
  agentListeners.add(fn);
  return () => { agentListeners.delete(fn); };
}

export interface Notice {
  id: number;
  kind: 'error' | 'success' | 'info';
  text: string;
}

/** Turn a registry entry into the shape the UI works with. */
function toModelInfo(id: string, vmr: any): ModelInfo {
  const gen = vmr.generation || {};
  const defaults = gen.defaults || {};
  const limits = gen.limits || {};
  const ui = vmr.ui || {};
  const identity = vmr.identity || {};
  const hardware = vmr.hardware || {};
  const dist = vmr.distribution || {};
  const fps = defaults.fps || 24;
  const frames = limits.num_frames;

  return {
    id,
    display_name: identity.display_name || id,
    family: identity.family || 'unknown',
    variant: identity.variant || '',
    name: identity.display_name || id,
    minVram: hardware.minimum_vram_gb || 0,
    size: dist.estimated_download_size_gb || 0,
    resolution: `${limits.width?.max || defaults.width || 0}×${limits.height?.max || defaults.height || 0}`,
    fps,
    duration: frames ? `${(frames.min / fps).toFixed(1)}–${(frames.max / fps).toFixed(1)}s` : 'Variable',
    huggingFaceRepo: dist.repo || '',
    tier: ui.tier || 'standard',
    description: ui.description || '',
    pros: ui.pros || [],
    cons: ui.cons || [],
    tags: ui.tags || [],
    accent_color: ui.accent_color || '#06b6d4',
    size_gb: dist.estimated_download_size_gb || 0,
    repo: dist.repo || '',
    license: dist.license,
    license_gate: vmr.license_gate || null,
    kind: (['enhancer', 'chat', 'image', 'voice'].includes(identity.kind) ? identity.kind : 'video') as ModelInfo['kind'],
    quality_rank: ui.quality_rank || 0,
    enhance_targets: vmr.enhance?.targets || [],
    minimum_vram_gb: hardware.minimum_vram_gb || 0,
    recommended_vram_gb: hardware.recommended_vram_gb || 0,
    recommended_ram_gb: hardware.recommended_ram_gb || 0,
    minimum_ram_gb: hardware.minimum_ram_gb || 0,
    speed_label: ui.speed_label || '',
    recommended: !!ui.recommended,
    hidden_controls: gen.hidden_controls || [],
    capabilities: vmr.capabilities || {
      text_to_video: true, image_to_video: false, video_to_video: false, audio_generation: false,
      lora: false, controlnet: false, max_prompt_tokens: null,
    },
    generation_defaults: defaults,
    generation_limits: limits,
    profiles: vmr.profiles,
    downloaded: false,
    downloading: false,
    progress: 0,
    downloadProgress: 0,
    speed_mbps: 0,
    speedMbps: 0,
    eta_seconds: 0,
    etaSeconds: 0,
    downloadError: null,
  };
}

function mergeState(m: ModelInfo, s: any): ModelInfo {
  if (!s) return m;
  return {
    ...m,
    downloaded: s.downloaded ?? m.downloaded,
    downloading: s.downloading ?? m.downloading,
    progress: s.progress ?? m.progress,
    downloadProgress: s.progress ?? m.downloadProgress,
    speed_mbps: s.speed_mbps ?? 0,
    speedMbps: s.speed_mbps ?? 0,
    eta_seconds: s.eta_seconds ?? 0,
    etaSeconds: s.eta_seconds ?? 0,
    downloadedBytes: s.downloaded_bytes ?? m.downloadedBytes,
    totalBytes: s.total_bytes ?? m.totalBytes,
    downloadError: s.error !== undefined ? s.error : m.downloadError,
    downloadStage: s.stage ?? null,
  };
}

const ACTIVE: GenerationStatus[] = ['queued', 'loading_model', 'generating', 'post_processing'];
export const isActiveJob = (j: GenerationJob) => ACTIVE.includes(j.status);

// Dev only: `npm run dev` in a plain browser has no Tauri shell, so connect to an
// engine started by hand (VITE_DEV_PORT / VITE_DEV_TOKEN). Stripped from builds.
const browserDev = import.meta.env.DEV && !('__TAURI_INTERNALS__' in window);

async function tauriInvoke<T>(cmd: string): Promise<T> {
  if (browserDev) {
    if (cmd === 'start_sidecar') {
      return { running: true, port: Number(import.meta.env.VITE_DEV_PORT), token: import.meta.env.VITE_DEV_TOKEN, message: '' } as T;
    }
    if (cmd === 'detect_gpu') {
      const port = Number(import.meta.env.VITE_DEV_PORT);
      configureApi(port, import.meta.env.VITE_DEV_TOKEN);
      const sys = await api<SystemInfo>('/system');
      return { name: sys.gpu_name, vram: Math.round(sys.vram_gb), driver: sys.driver, cuda_version: '?', detected: !!sys.gpu_name } as T;
    }
    return undefined as T;
  }
  return invoke<T>(cmd);
}

export function useAppStore() {
  const [view, setView] = useState<AppView>('setup');
  const [setupStep, setSetupStep] = useState(0);
  const [gpu, setGpu] = useState<GPUInfo | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [outputs, setOutputs] = useState<OutputItem[]>([]);
  const [storage, setStorage] = useState<StorageReport | null>(null);
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [notices, setNotices] = useState<Notice[]>([]);
  const [sidecarStatus, setSidecarStatus] = useState<SidecarStatus>({
    running: false, port: 0, comfyui_ready: false, version: '', python_version: '',
  });
  const [sidecarError, setSidecarError] = useState<string | null>(null);
  const [license, setLicense] = useState<LicenseInfo>({
    key: '', valid: false, tier: 'free', features: ['Unlimited local generation'],
  });
  const wsRef = useRef<WebSocket | null>(null);
  const jobsRef = useRef<GenerationJob[]>([]);
  const noticeId = useRef(0);

  const notify = useCallback((kind: Notice['kind'], text: string) => {
    const id = ++noticeId.current;
    setNotices((n) => [...n.slice(-3), { id, kind, text }]);
    setTimeout(() => setNotices((n) => n.filter((x) => x.id !== id)), kind === 'error' ? 9000 : 4500);
  }, []);
  const dismissNotice = useCallback((id: number) => setNotices((n) => n.filter((x) => x.id !== id)), []);

  const detectGPU = useCallback(async () => {
    setSetupStep(1);
    try {
      setGpu(await tauriInvoke<GPUInfo>('detect_gpu'));
    } catch {
      setGpu({ name: 'No NVIDIA GPU detected', vram: 0, vram_mb: 0, vram_gb: 0, driver: 'N/A',
        cuda_version: 'N/A', temperature: 0, detected: false });
    }
    setSetupStep(2);
  }, []);

  const fetchModels = useCallback(async () => {
    try {
      const [config, states] = await Promise.all([api('/models/config'), api('/models')]);
      const merged = Object.entries(config.models as Record<string, any>)
        .map(([id, vmr]) => mergeState(toModelInfo(id, vmr), states[id]))
        .sort((a, b) => Number(b.downloaded) - Number(a.downloaded)
          || Number(b.recommended) - Number(a.recommended) || a.size - b.size);
      setModels(merged);
      const videoModels = merged.filter((m) => m.kind === 'video');
      setSelectedModel((prev) =>
        prev && videoModels.some((m) => m.id === prev && m.downloaded)
          ? prev
          : videoModels.find((m) => m.downloaded && m.capabilities.text_to_video !== false)?.id
            || videoModels.find((m) => m.downloaded)?.id || videoModels[0]?.id || '');
      return merged;
    } catch (err) {
      console.error('[NeuralCut] fetchModels failed:', err);
      return [];
    }
  }, []);

  const [images, setImages] = useState<ImageItem[]>([]);
  const [audio, setAudio] = useState<AudioItem[]>([]);
  const refreshAudio = useCallback(async () => {
    try {
      setAudio(await api<AudioItem[]>('/audio'));
    } catch (err) {
      console.error('[Pipeline] audio failed:', err);
    }
  }, []);
  const refreshImages = useCallback(async () => {
    try {
      setImages(await api<ImageItem[]>('/images'));
    } catch (err) {
      console.error('[Pipeline] images failed:', err);
    }
  }, []);

  const refreshOutputs = useCallback(async () => {
    try {
      setOutputs(await api<OutputItem[]>('/outputs'));
    } catch (err) {
      console.error('[NeuralCut] outputs failed:', err);
    }
  }, []);

  const refreshStorage = useCallback(async () => {
    try {
      setStorage(await api<StorageReport>('/storage'));
    } catch (err) {
      console.error('[NeuralCut] storage failed:', err);
    }
  }, []);

  // Applies one job_status update (from the WebSocket or a /jobs resync).
  // Toasts fire once per job, only for jobs started from this window.
  const myJobs = useRef(new Set<string>());
  const announced = useRef(new Set<string>());
  const applyJobStatus = useCallback((msg: any) => {
    setJobs((prev) =>
      prev.map((j) => {
        if (j.id !== msg.job_id) return j;
        if (['done', 'error', 'cancelled'].includes(j.status)) return j;  // final states never regress
        return {
          ...j,
          status: msg.status as GenerationStatus,
          progress: msg.progress ?? j.progress,
          eta: msg.eta ?? 0,
          message: msg.message || (msg.status === j.status ? j.message : undefined),
          outputPath: msg.outputPath || j.outputPath,
          error: msg.error || undefined,
          elapsedSeconds: msg.elapsed_seconds,
          enhanceJobId: msg.enhance_job_id || j.enhanceJobId,
          endTime: ['done', 'error', 'cancelled'].includes(msg.status) ? Date.now() : j.endTime,
        };
      }));
    const id = msg.job_id;
    if ((msg.status === 'done' || msg.status === 'error') && myJobs.current.has(id) && !announced.current.has(id)) {
      announced.current.add(id);
      if (msg.status === 'done') {
        refreshOutputs();
        const kind = jobsRef.current.find((j) => j.id === id)?.kind;
        if (kind === 'image' || String(msg.outputPath || '').endsWith('.png')) refreshImages();
        if (kind === 'voice' || String(msg.outputPath || '').endsWith('.wav')) refreshAudio();
        notify('success', kind === 'voice' ? 'Your voice-over is ready.' : kind === 'motion' ? 'Your photo motion is ready.' : kind === 'image' ? 'Your image is ready.' : kind === 'edit' ? 'Your edit is ready. It is in the gallery.'
          : msg.enhance_job_id ? 'Your video is ready. Enhancing it now…' : 'Your video is ready.');
      } else {
        notify('error', `Generation failed: ${msg.error || 'unknown error'}`);
      }
    }
  }, [notify, refreshOutputs]);

  const resyncJobs = useCallback(async () => {
    try {
      const states = await api<Record<string, any>>('/jobs');
      Object.values(states).forEach(applyJobStatus);
      const orphaned = jobsRef.current.filter((j) => isActiveJob(j) && myJobs.current.has(j.id) && !states[j.id]);
      if (orphaned.length) {
        const outs = await api<OutputItem[]>('/outputs');
        setOutputs(outs);
        for (const j of orphaned) {
          const made = outs.some((o) => o.name === `video_${j.id}.mp4`);
          applyJobStatus(made
            ? { job_id: j.id, status: 'done', progress: 100 }
            : { job_id: j.id, status: 'error', error: 'Interrupted because the AI engine restarted. Please generate again.' });
        }
      }
      for (const st of Object.values(states)) {
        if (!myJobs.current.has(st.job_id) || ['done', 'error', 'cancelled'].includes(st.status)) continue;
        const logs = await api<string[]>(`/jobs/${st.job_id}/logs`);
        setJobs((prev) => prev.map((j) => (j.id === st.job_id && (j.logs?.length ?? 0) < logs.length
          ? { ...j, logs: logs.slice(-300) } : j)));
      }
    } catch { /* engine restarting; the next tick retries */ }
  }, [applyJobStatus]);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;
    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;

    ws.onmessage = (event) => {
      let msg: any;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === 'download_progress') {
        setModels((prev) => prev.map((m) => (m.id === msg.model_id ? mergeState(m, msg) : m)));
        if (msg.downloaded && !msg.downloading) {
          notify('success', 'Model installed and verified. Ready to use.');
          refreshStorage();
          setModels((prev) => {
            if (prev.find((m) => m.id === msg.model_id)?.kind === 'video') setSelectedModel((s) => s || msg.model_id);
            return prev;
          });
        } else if (msg.error && !msg.downloading) {
          notify('error', msg.error);
        }
      } else if (msg.type === 'agent_event' || msg.type === 'agent_state' || msg.type === 'production') {
        if (msg.type === 'production') { refreshOutputs(); refreshImages(); }
        agentListeners.forEach((fn) => fn(msg));
      } else if (msg.type === 'job_status') {
        applyJobStatus(msg);
      } else if (msg.type === 'job_created') {
        if (msg.parent_job_id && !myJobs.current.has(msg.parent_job_id)) return;
        myJobs.current.add(msg.job_id);
        setJobs((prev) => prev.some((j) => j.id === msg.job_id) ? prev : [{
          id: msg.job_id, prompt: msg.prompt || prev.find((j) => j.id === msg.parent_job_id)?.prompt || 'Enhancing video',
          negative_prompt: '', model_id: msg.model_id, status: 'queued', progress: 0, eta: 0,
          startTime: Date.now(), kind: ['generate', 'image', 'voice', 'motion'].includes(msg.kind) ? msg.kind : 'enhance', summary: msg.summary, parentId: msg.parent_job_id || undefined,
          estimateSeconds: msg.estimate_seconds, logs: [],
        }, ...prev]);
      } else if (msg.type === 'job_log') {
        setJobs((prev) => prev.map((j) => {
          if (j.id !== msg.job_id) return j;
          const logs = j.logs ? [...j.logs] : [];
          if (msg.replace && logs.length) logs[logs.length - 1] = msg.line;
          else logs.push(msg.line);
          return { ...j, logs: logs.slice(-300) };
        }));
      } else if (msg.type === 'strategy_attempt' && msg.status === 'failed') {
        setJobs((prev) => prev.map((j) => (j.id === msg.job_id
          ? { ...j, message: 'Retrying with a lower-memory method…' } : j)));
      }
    };
    ws.onopen = () => resyncJobs();
    ws.onclose = () => {
      if (wsRef.current === ws) wsRef.current = null;
      setTimeout(() => { if (apiReady()) connectWebSocket(); }, 2000);
    };
    ws.onerror = () => ws.close();
  }, [notify, refreshStorage, applyJobStatus, resyncJobs]);

  useEffect(() => {
    if (!sidecarStatus.running) return;
    fetchModels();
    refreshOutputs();
    refreshImages();
    refreshAudio();
    refreshStorage();
    api<SystemInfo>('/system').then(setSystem).catch(() => {});
    connectWebSocket();
  }, [sidecarStatus.running, fetchModels, refreshOutputs, refreshStorage, connectWebSocket]);

  // Safety net while anything is running: resync every few seconds.
  const hasActiveJobs = jobs.some(isActiveJob);
  jobsRef.current = jobs;
  useEffect(() => {
    if (!hasActiveJobs || !sidecarStatus.running) return;
    const t = setInterval(resyncJobs, 4000);
    return () => clearInterval(t);
  }, [hasActiveJobs, sidecarStatus.running, resyncJobs]);

  const startSidecar = useCallback(async () => {
    setSidecarError(null);
    try {
      const st = await tauriInvoke<{ running: boolean; port: number; token: string; message: string }>('start_sidecar');
      if (!st.running) {
        setSidecarError(st.message || 'The AI engine did not start.');
        return;
      }
      configureApi(st.port, st.token);
      // The engine imports torch on start, which can take a while on a cold disk.
      const deadline = Date.now() + 90_000;
      while (Date.now() < deadline) {
        const h = await healthy();
        if (h) {
          setSidecarStatus({ running: true, port: st.port, token: st.token, comfyui_ready: true,
            version: h.version, python_version: h.python_version });
          return;
        }
        await new Promise((r) => setTimeout(r, 500));
      }
      setSidecarError('The AI engine started but did not respond. Check the log in Settings.');
    } catch (err) {
      setSidecarError(String(err));
    }
  }, []);

  const restartEngine = useCallback(async () => {
    wsRef.current?.close();
    wsRef.current = null;
    try { await tauriInvoke('stop_sidecar'); } catch { /* already stopped */ }
    setSidecarStatus((s) => ({ ...s, running: false }));
    await startSidecar();
  }, [startSidecar]);

  // Watchdog: if the engine stops answering (crash, killed by antivirus, …),
  // show it as offline and restart it so the user never has to.
  const restarting = useRef(false);
  useEffect(() => {
    if (!sidecarStatus.running) return;
    let failures = 0;
    const t = setInterval(async () => {
      if (restarting.current) return;
      if (await healthy()) {
        failures = 0;
        return;
      }
      failures += 1;
      if (failures >= 3) {
        restarting.current = true;
        notify('error', 'The AI engine stopped responding. Restarting it…');
        await restartEngine();
        restarting.current = false;
        failures = 0;
      }
    }, 10_000);
    return () => clearInterval(t);
  }, [sidecarStatus.running, restartEngine, notify]);

  const downloadModel = useCallback(async (modelId: string, acceptLicense = false) => {
    setModels((prev) => prev.map((m) => (m.id === modelId ? { ...m, downloading: true, downloadError: null } : m)));
    try {
      await post(`/models/download/${modelId}`, { accept_license: acceptLicense });
    } catch (err: any) {
      setModels((prev) => prev.map((m) => (m.id === modelId ? { ...m, downloading: false } : m)));
      notify('error', err.message);
    }
  }, [notify]);

  const cancelDownload = useCallback(async (modelId: string) => {
    try {
      await post(`/models/download/${modelId}/cancel`);
      notify('info', 'Download paused. Progress is kept, so it resumes where it stopped.');
    } catch (err: any) {
      notify('error', err.message);
    }
  }, [notify]);

  const deleteModel = useCallback(async (modelId: string) => {
    try {
      const res = await del<{ removed: string[] }>(`/models/${modelId}`);
      notify('success', `Removed ${res.removed.length} folder${res.removed.length === 1 ? '' : 's'}.`);
    } catch (err: any) {
      notify('error', err.message);
    }
    await fetchModels();
    refreshStorage();
  }, [fetchModels, notify, refreshStorage]);

  const deleteOrphan = useCallback(async (name: string) => {
    try {
      await del(`/storage/orphans/${encodeURIComponent(name)}`);
      notify('success', `Deleted ${name}.`);
    } catch (err: any) {
      notify('error', err.message);
    }
    refreshStorage();
  }, [notify, refreshStorage]);

  const startGeneration = useCallback(
    async (prompt: string, negPrompt: string, modelId: string, settings: Record<string, any>) => {
      try {
        const data = await post<{ job_id: string; estimate_seconds: number | null }>('/generate', {
          prompt, negative_prompt: negPrompt, model_id: modelId, ...settings,
        });
        myJobs.current.add(data.job_id);
        setJobs((prev) => [{
          id: data.job_id, prompt, negative_prompt: negPrompt, model_id: modelId,
          status: 'queued', progress: 0, eta: 0, startTime: Date.now(),
          estimateSeconds: data.estimate_seconds, logs: [],
          kind: settings.kind === 'image' ? 'image' : 'generate',
          summary: settings.kind === 'image' ? `Image ${settings.width}×${settings.height}`
            : `${settings.start_image ? 'From image · ' : ''}${settings.width}×${settings.height} · ${(settings.num_frames / (settings.fps || 24)).toFixed(1)}s`,
        }, ...prev]);
        return data.job_id;
      } catch (err: any) {
        notify('error', err.message);
        return null;
      }
    }, [notify]);

  /** Upscale/restore an existing video with an installed enhancer. */
  const enhanceVideo = useCallback(async (source: string, enhancerId: string, target: string, prompt: string) => {
    try {
      const data = await post<{ job_id: string }>('/enhance', { source, enhancer_id: enhancerId, target });
      myJobs.current.add(data.job_id);
      const name = models.find((m) => m.id === enhancerId)?.name || 'Enhance';
      setJobs((prev) => prev.some((j) => j.id === data.job_id) ? prev : [{
        id: data.job_id, prompt: prompt || source, negative_prompt: '', model_id: enhancerId,
        status: 'queued', progress: 0, eta: 0, startTime: Date.now(), kind: 'enhance',
        summary: `${name} to ${target}`, logs: [],
      }, ...prev]);
      notify('info', 'Enhancement queued. You can follow it in the Jobs list.');
    } catch (err: any) {
      notify('error', err.message);
    }
  }, [models, notify]);

  /** Expected seconds for these settings on this PC, or null without history. */
  const estimateFor = useCallback(async (modelId: string, settings: Record<string, any>) => {
    try {
      const res = await post<{ seconds: number } | null>('/estimate', { prompt: '-', model_id: modelId, ...settings });
      return res?.seconds ?? null;
    } catch {
      return null;
    }
  }, []);

  /** Stitch clips into one video (runs on the CPU, alongside any GPU job). */
  const renderEdit = useCallback(async (payload: RenderPayload) => {
    try {
      const data = await post<{ job_id: string; duration: number }>('/edit/render', payload);
      myJobs.current.add(data.job_id);
      setJobs((prev) => [{
        id: data.job_id, prompt: payload.title || `Edit of ${payload.clips.length} clips`, negative_prompt: '',
        model_id: 'editor', status: 'queued', progress: 0, eta: 0, startTime: Date.now(), kind: 'edit',
        summary: `${payload.clips.length} clips · ${data.duration.toFixed(1)}s · ${payload.transition === 'fade' ? 'crossfades' : 'cuts'}`,
        logs: [],
      }, ...prev]);
      return data.job_id;
    } catch (err: any) {
      notify('error', err.message);
      return null;
    }
  }, [notify]);

  /** Text to speech: a preset voice (Kokoro) or a voice described in words (Qwen3-TTS). */
  const speak = useCallback(async (body: { model_id: string; text: string; voice?: string; instruct?: string; language?: string; speed?: number; title?: string }) => {
    try {
      const data = await post<{ job_id: string }>('/voices/speak', body);
      myJobs.current.add(data.job_id);
      setJobs((prev) => [{
        id: data.job_id, prompt: body.text, negative_prompt: '', model_id: body.model_id, status: 'queued', progress: 0,
        eta: 0, startTime: Date.now(), kind: 'voice', summary: body.instruct ? 'Designed voice' : `Voice: ${body.voice}`, logs: [],
      }, ...prev]);
      return data.job_id;
    } catch (err: any) {
      notify('error', err.message);
      return null;
    }
  }, [notify]);

  /** Photo motion: a cinematic camera move through a still image. */
  const animatePhoto = useCallback(async (image: string, move: string, seconds: number, strength = 1) => {
    try {
      const data = await post<{ job_id: string }>(`/images/${encodeURIComponent(image)}/animate`, { move, seconds, strength });
      myJobs.current.add(data.job_id);
      setJobs((prev) => [{
        id: data.job_id, prompt: `Photo motion of ${image}`, negative_prompt: '', model_id: 'photo-motion', status: 'queued',
        progress: 0, eta: 0, startTime: Date.now(), kind: 'motion', summary: `${move.replace('_', ' ')} · ${seconds}s`, logs: [],
      }, ...prev]);
      return data.job_id;
    } catch (err: any) {
      notify('error', err.message);
      return null;
    }
  }, [notify]);

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      await post(`/generate/${jobId}/cancel`);
    } catch (err: any) {
      notify('error', err.message);
    }
  }, [notify]);

  const dismissJob = useCallback((jobId: string) => setJobs((prev) => prev.filter((j) => j.id !== jobId)), []);

  const deleteOutput = useCallback(async (name: string) => {
    try {
      await del(`/outputs/${encodeURIComponent(name)}`);
      setOutputs((prev) => prev.filter((o) => o.name !== name));
    } catch (err: any) {
      notify('error', err.message);
    }
  }, [notify]);

  const revealOutput = useCallback(async (name: string) => {
    try {
      await post(`/outputs/${encodeURIComponent(name)}/reveal`);
    } catch (err: any) {
      notify('error', err.message);
    }
  }, [notify]);

  const validateLicense = useCallback(async (key: string) => {
    try {
      const data = await post('/license/validate', { key });
      setLicense({ key, valid: data.valid, tier: data.tier, expires_at: data.expires_at, features: data.features });
    } catch {
      setLicense({ key, valid: false, tier: 'free', features: ['Unlimited local generation'] });
    }
  }, []);

  return {
    view, setView, setupStep, setSetupStep, gpu, setGpu, system,
    models, setModels, jobs, setJobs, outputs, storage,
    currentPrompt, setCurrentPrompt, negativePrompt, setNegativePrompt,
    selectedModel, setSelectedModel, sidecarStatus, setSidecarStatus, sidecarError,
    license, setLicense, notices, dismissNotice, notify,
    detectGPU, startSidecar, restartEngine, fetchModels, refreshOutputs, refreshStorage,
    downloadModel, cancelDownload, deleteModel, deleteOrphan,
    images, refreshImages, audio, refreshAudio, speak, animatePhoto,
    startGeneration, estimateFor, enhanceVideo, renderEdit, cancelJob, dismissJob, deleteOutput, revealOutput, validateLicense,
  };
}
