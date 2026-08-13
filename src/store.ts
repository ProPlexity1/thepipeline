import { useState, useCallback, useEffect } from 'react';
import type {
  GPUInfo,
  ModelInfo,
  GenerationJob,
  AppView,
  SidecarStatus,
  LicenseInfo,
  GenerationStatus,
  VMRRegistry,
  VMRModelEntry,
} from './types';
import { invoke } from '@tauri-apps/api/core';

let activeWs: WebSocket | null = null;
const activeEventSources: Record<string, EventSource> = {};

/**
 * Transform a VMRModelEntry into a ModelInfo (the shape the UI works with).
 * This extracts the relevant fields and derives UI-friendly values.
 */
function transformVMRToModelInfo(id: string, vmr: VMRModelEntry): ModelInfo {
  const gen = vmr.generation;
  const widthDefault = gen.defaults['width'] || 512;
  const heightDefault = gen.defaults['height'] || 512;
  const fpsDefault = gen.defaults['fps'] || 24;
  const numFramesDefault = gen.defaults['num_frames'] || 49;

  // Derive resolution string from defaults
  const resolution = `${widthDefault}×${heightDefault}`;

  // Derive duration range from frame limits and fps
  const frameLimits = gen.limits['num_frames'];
  const fpsLimits = gen.limits['fps'];
  let duration = 'Variable';
  if (frameLimits && fpsLimits) {
    const minSecs = (frameLimits.min / fpsLimits.default).toFixed(1);
    const maxSecs = (frameLimits.max / fpsLimits.default).toFixed(1);
    duration = `${minSecs}-${maxSecs}s`;
  }

  return {
    // Identity
    id,
    display_name: vmr.identity.display_name,
    family: vmr.identity.family,
    variant: vmr.identity.variant,

    // Legacy/alias fields for UI compatibility
    name: vmr.identity.display_name,
    minVram: vmr.hardware.minimum_vram_gb,
    size: vmr.distribution.estimated_download_size_gb,
    resolution,
    fps: fpsDefault,
    duration,
    huggingFaceRepo: vmr.distribution.repo,

    // UI presentation
    tier: vmr.ui.tier,
    description: vmr.ui.description,
    pros: vmr.ui.pros,
    cons: vmr.ui.cons,
    tags: vmr.ui.tags,
    accent_color: vmr.ui.accent_color,

    // Distribution
    size_gb: vmr.distribution.estimated_download_size_gb,
    repo: vmr.distribution.repo,

    // Hardware requirements
    minimum_vram_gb: vmr.hardware.minimum_vram_gb,
    recommended_vram_gb: vmr.hardware.recommended_vram_gb,
    recommended_ram_gb: vmr.hardware.recommended_ram_gb,

    // Capabilities
    capabilities: vmr.capabilities,

    // Generation settings for sliders & profiles
    generation_defaults: gen.defaults,
    generation_limits: gen.limits,
    profiles: vmr.profiles,
    scheduler: vmr.scheduler,
    quantization: vmr.quantization,

    // Download state (populated later from /models endpoint)
    downloaded: false,
    downloading: false,
    progress: 0,
    downloadProgress: 0,
    speed_mbps: 0,
    speedMbps: 0,
    eta_seconds: 0,
    etaSeconds: 0,
  };
}

/**
 * Fetch VMR from backend and transform into ModelInfo array.
 */
async function fetchVMRModels(): Promise<ModelInfo[]> {
  const res = await fetch('http://127.0.0.1:8188/models/config');
  const data: { metadata: any; models: Record<string, VMRModelEntry> } = await res.json();

  return Object.entries(data.models).map(([id, vmr]) => transformVMRToModelInfo(id, vmr));
}

const connectWebSocket = (
  setJobs: React.Dispatch<React.SetStateAction<GenerationJob[]>>,
  setGalleryItems: React.Dispatch<React.SetStateAction<GenerationJob[]>>,
  setModels: React.Dispatch<React.SetStateAction<ModelInfo[]>>
) => {
  if (activeWs) {
    if (activeWs.readyState === WebSocket.OPEN || activeWs.readyState === WebSocket.CONNECTING) {
      return;
    }
    activeWs.close();
  }

  const ws = new WebSocket('ws://127.0.0.1:8188/ws');
  activeWs = ws;

  ws.onopen = () => {
    console.log('[NeuralCut] WebSocket connected');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);

      if (msg.type === 'download_progress') {
        const { model_id, progress, speed_mbps, eta_seconds, downloading, downloaded } = msg;
        setModels((prev) =>
          prev.map((m) =>
            m.id === model_id
              ? {
                  ...m,
                  progress,
                  downloadProgress: progress,
                  speed_mbps,
                  speedMbps: speed_mbps,
                  eta_seconds,
                  etaSeconds: eta_seconds,
                  downloading,
                  downloaded,
                }
              : m
          )
        );
      }

      if (msg.type === 'job_status') {
        const { job_id, status, progress, eta, outputPath, error } = msg;
        setJobs((prev) =>
          prev.map((j) => {
            if (j.id === job_id) {
              const updatedJob = {
                ...j,
                status: status as GenerationStatus,
                progress,
                eta,
                outputPath: outputPath || undefined,
                error: error || undefined,
                endTime:
                  status === 'done' || status === 'error'
                    ? Date.now()
                    : j.endTime,
              };
              if (status === 'done' && j.status !== 'done') {
                setGalleryItems((gal) => {
                  if (gal.some((g) => g.id === job_id)) return gal;
                  return [updatedJob, ...gal];
                });
              }
              return updatedJob;
            }
            return j;
          })
        );
      }
    } catch (e) {
      console.error('[NeuralCut] Error parsing WebSocket message:', e);
    }
  };

  ws.onclose = () => {
    console.log('[NeuralCut] WebSocket closed, retrying in 3s...');
    activeWs = null;
    setTimeout(() => connectWebSocket(setJobs, setGalleryItems, setModels), 3000);
  };

  ws.onerror = (err) => {
    console.error('[NeuralCut] WebSocket error:', err);
    ws.close();
  };
};

export function useAppStore() {
  const [view, setView] = useState<AppView>('setup');
  const [setupStep, setSetupStep] = useState(0);
  const [gpu, setGpu] = useState<GPUInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [sidecarStatus, setSidecarStatus] = useState<SidecarStatus>({
    running: false,
    port: 8188,
    comfyui_ready: false,
    version: '1.0.0',
    python_version: '',
  });
  const [sidecarError, setSidecarError] = useState<string | null>(null);
  const [license, setLicense] = useState<LicenseInfo>({
    key: '',
    valid: false,
    tier: 'free',
    features: ['Basic generation', '512px max', 'Watermark'],
  });
  const [galleryItems, setGalleryItems] = useState<GenerationJob[]>([]);

  const detectGPU = useCallback(async () => {
    setSetupStep(1);
    try {
      const gpuInfo = await invoke<GPUInfo>('detect_gpu');
      setGpu(gpuInfo);
      setSetupStep(2);
    } catch (err) {
      console.error('GPU detection failed:', err);
      setGpu({
        name: 'No NVIDIA GPU detected',
        vram: 0,
        vram_mb: 0,
        vram_gb: 0,
        driver: 'N/A',
        cuda_version: 'N/A',
        temperature: 0,
        detected: false,
      });
      setSetupStep(2);
    }
  }, []);

  /**
   * Fetch the VMR from the backend and build the model list from it,
   * then layer live download status on top.
   */
  const fetchModels = useCallback(async () => {
    try {
      // Fetch VMR and transform to ModelInfo
      const baseModels = await fetchVMRModels();

      // Fetch live download status from backend
      const statusRes = await fetch('http://127.0.0.1:8188/models');
      const backendModels = await statusRes.json();

      // Merge: VMR data + download state
      const merged = baseModels.map((m) => {
        const backendModel = backendModels[m.id];
        if (backendModel) {
          return {
            ...m,
            downloaded: backendModel.downloaded,
            downloading: backendModel.downloading,
            progress: backendModel.progress,
            downloadProgress: backendModel.progress,
            speed_mbps: backendModel.speed_mbps ?? 0,
            speedMbps: backendModel.speed_mbps ?? 0,
            eta_seconds: backendModel.eta_seconds ?? 0,
            etaSeconds: backendModel.eta_seconds ?? 0,
          };
        }
        return m;
      });

      setModels(merged);

      // Default the selected model to the first downloaded one, or first one if none are downloaded
      setSelectedModel((prev) => {
        if (prev) return prev;
        const firstDownloaded = merged.find((m) => m.downloaded);
        return firstDownloaded?.id || merged[0]?.id || '';
      });
      return merged;
    } catch (err) {
      console.error('Failed to fetch VMR from sidecar:', err);
      return [];
    }
  }, []);

  useEffect(() => {
    if (sidecarStatus.running && sidecarStatus.comfyui_ready) {
      fetchModels();
      connectWebSocket(setJobs, setGalleryItems, setModels);
    }
  }, [sidecarStatus.running, sidecarStatus.comfyui_ready, fetchModels]);

  const startSidecar = useCallback(async () => {
    setSidecarError(null);
    try {
      const status = await invoke<{
        running: boolean;
        port: number;
        pid: number | null;
        message: string;
      }>('start_sidecar');

      console.log('Sidecar invoke result:', JSON.stringify(status));

      if (status.running) {
        const deadline = Date.now() + 15000;
        const pollHealth = async () => {
          try {
            const res = await fetch('http://127.0.0.1:8188/health');
            if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
            const data = await res.json();
            console.log('Health check result:', JSON.stringify(data));
            setSidecarStatus({
              running: true,
              port: 8188,
              comfyui_ready: data.comfyui_ready,
              version: data.version,
              python_version: data.python_version,
            });
            return;
          } catch (err) {
            if (Date.now() < deadline) {
              setTimeout(pollHealth, 500);
              return;
            }
            console.error('Health check failed:', err);
            setSidecarError('Backend started, but did not become ready in time.');
            setSidecarStatus((prev) => ({ ...prev, running: false }));
          }
        };
        pollHealth();
      } else {
        console.error('Sidecar reported not running:', status.message);
        setSidecarError(status.message || 'Backend did not start.');
      }
    } catch (err) {
      console.error('start_sidecar invoke failed:', err);
      setSidecarError(String(err));
    }
  }, []);

  const downloadModel = useCallback(async (modelId: string) => {
    if (activeEventSources[modelId]) return;

    setModels((prev) =>
      prev.map((m) =>
        m.id === modelId
          ? { ...m, downloading: true, progress: 0 }
          : m
      )
    );

    try {
      const startRes = await fetch(
        `http://127.0.0.1:8188/models/download/${modelId}`,
        { method: 'POST' }
      );
      const startData = await startRes.json();
      if (startData.status === 'error') {
        console.error('Failed to start model download:', startData.message);
        setModels((prev) =>
          prev.map((m) =>
            m.id === modelId
              ? { ...m, downloading: false, progress: 0 }
              : m
          )
        );
        return;
      }

      const es = new EventSource(
        `http://127.0.0.1:8188/models/download/${modelId}/progress`
      );
      activeEventSources[modelId] = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (!data.downloading || data.downloaded) {
            es.close();
            delete activeEventSources[modelId];
            setModels((prev) =>
              prev.map((m) =>
                m.id === modelId
                  ? {
                      ...m,
                      downloading: data.downloading,
                      downloaded: data.downloaded,
                      progress: data.progress,
                      downloadProgress: data.progress,
                    }
                  : m
              )
            );
          }
        } catch (e) {
          console.error('Error parsing SSE message:', e);
        }
      };

      es.onerror = () => {
        es.close();
        delete activeEventSources[modelId];
      };
    } catch (err) {
      console.error('Error initiating download:', err);
      setModels((prev) =>
        prev.map((m) =>
          m.id === modelId
            ? { ...m, downloading: false, progress: 0 }
            : m
        )
      );
    }
  }, []);

  const cancelDownload = useCallback(async (modelId: string) => {
    if (activeEventSources[modelId]) {
      activeEventSources[modelId].close();
      delete activeEventSources[modelId];
    }
    setModels((prev) =>
      prev.map((m) =>
        m.id === modelId ? { ...m, downloading: false, progress: 0 } : m
      )
    );
    try {
      await fetch(`http://127.0.0.1:8188/models/${modelId}`, {
        method: 'DELETE',
      });
    } catch (err) {
      console.error('Failed to cancel download on backend:', err);
    }
  }, []);

  const deleteModel = useCallback(async (modelId: string) => {
    setModels((prev) =>
      prev.map((m) =>
        m.id === modelId ? { ...m, downloaded: false, progress: 0 } : m
      )
    );
    try {
      await fetch(`http://127.0.0.1:8188/models/${modelId}`, {
        method: 'DELETE',
      });
    } catch (err) {
      console.error('Failed to delete model on backend:', err);
    }
  }, []);

  /**
   * Start a generation job.
   * Takes slider values from the UI, sends them to the backend.
   * Backend re-validates every setting against VMR limits via clamp_settings().
   */
  const startGeneration = useCallback(
    async (
      prompt: string,
      negPrompt: string,
      modelId: string,
      settings: Record<string, any>
    ) => {
      try {
        const res = await fetch('http://127.0.0.1:8188/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            negative_prompt: negPrompt,
            model_id: modelId,
            ...settings,
          }),
        });
        const data = await res.json();

        if (data.status === 'error') {
          console.error('Generation error:', data.message);
          return;
        }

        const newJob: GenerationJob = {
          id: data.job_id,
          prompt,
          negative_prompt: negPrompt,
          model_id: modelId,
          status: 'queued',
          progress: 0,
          eta: 0,
          startTime: Date.now(),
        };
        setJobs((prev) => [newJob, ...prev]);
      } catch (err) {
        console.error('Failed to start generation:', err);
      }
    },
    []
  );

  const validateLicense = useCallback(async (key: string) => {
    try {
      const res = await fetch('http://127.0.0.1:8188/license/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key }),
      });
      const data = await res.json();
      setLicense({
        key,
        valid: data.valid,
        tier: data.tier,
        expires_at: data.expires_at,
        features: data.features,
      });
    } catch (err) {
      console.error('License validation failed:', err);
      setLicense({
        key,
        valid: false,
        tier: 'free',
        features: ['Basic generation', '512px max', 'Watermark'],
      });
    }
  }, []);

  return {
    view,
    setView,
    setupStep,
    setSetupStep,
    gpu,
    setGpu,
    models,
    setModels,
    jobs,
    setJobs,
    currentPrompt,
    setCurrentPrompt,
    negativePrompt,
    setNegativePrompt,
    selectedModel,
    setSelectedModel,
    sidecarStatus,
    setSidecarStatus,
    sidecarError,
    license,
    setLicense,
    galleryItems,
    setGalleryItems,
    detectGPU,
    startSidecar,
    fetchModels,
    downloadModel,
    cancelDownload,
    deleteModel,
    startGeneration,
    validateLicense,
  };
}
