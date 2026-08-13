// ── VMR (Video Model Registry) Structure ──────────────────────────────────────

export interface VMRMetadata {
  schema_version: string;
  registry_version: string;
  minimum_app_version: string;
  last_updated: string;
}

export interface ModelSettingSpec {
  default: number;
  min: number;
  max: number;
  step: number;
}

export interface ModelCapabilities {
  text_to_video: boolean;
  image_to_video: boolean;
  video_to_video: boolean;
  audio_generation: boolean;
  lora: boolean;
  controlnet: boolean;
  max_prompt_tokens: number | null;
}

export interface ModelDistribution {
  provider: string;
  repo: string;
  revision: string;
  download_method: string;
  estimated_download_size_gb: number;
  required_disk_space_gb: number;
  shared_resources: string | null;
  files: {
    required: string[];
    optional: string[];
    generated: string[];
  };
}

export interface ModelRuntime {
  backend: string;
  pipeline_class: string;
  load_mode: string;
  transformer_class: string | null;
  transformer_file: string | null;
  dtype: string;
}

export interface ModelOptimization {
  offload_strategy: string;
  vae_tiling: boolean;
  vae_slicing: boolean;
  attention_slicing: boolean;
  torch_compile: boolean;
  quantization: string | null;
  notes: string | null;
}

export interface ModelHardware {
  minimum_vram_gb: number;
  recommended_vram_gb: number;
  recommended_ram_gb: number;
  recommended_gpu: string;
}

export interface ModelGeneration {
  defaults: Record<string, number>;
  limits: Record<string, ModelSettingSpec>;
}

export interface ModelUI {
  tier: string;
  featured: boolean;
  recommended: boolean;
  description: string;
  pros: string[];
  cons: string[];
  tags: string[];
  thumbnail: string | null;
  accent_color: string;
}

export interface ModelVerification {
  status: "verified" | "partially_verified" | "unverified" | "known_issue";
  verified_on: string | null;
  verified_sources: string[];
  notes: string;
}

export interface ModelIdentity {
  id: string;
  display_name: string;
  family: string;
  variant: string;
  status: string;
}

export interface GenerationProfile {
  label: string;
  description: string;
  steps: number;
  cfg_scale: number;
  width: number;
  height: number;
  num_frames: number;
  fps: number;
  scheduler?: string;
  quantization?: string;
}

export interface ModelSchedulerConfig {
  default: string;
  class_name: string;
  params: Record<string, any>;
}

export interface ModelQuantizationConfig {
  default: string;
  supported: string[];
}

export interface ModelValidationConfig {
  min_frames: number;
  min_width: number;
  min_height: number;
  check_nan: boolean;
  check_black_frames: boolean;
  min_std_dev: number;
  max_std_dev: number;
}

export interface VMRModelEntry {
  identity: ModelIdentity;
  capabilities: ModelCapabilities;
  distribution: ModelDistribution;
  runtime: ModelRuntime;
  optimization: ModelOptimization;
  hardware: ModelHardware;
  generation: ModelGeneration;
  ui: ModelUI;
  documentation: Record<string, string | null>;
  verification: ModelVerification;
  profiles?: Record<string, GenerationProfile>;
  scheduler?: ModelSchedulerConfig;
  quantization?: ModelQuantizationConfig;
  validation?: ModelValidationConfig;
}

export interface VMRRegistry {
  vmr_metadata: VMRMetadata;
  shared_resources: Record<string, any>;
  families: Record<string, any>;
  models: Record<string, VMRModelEntry>;
  changelog: any[];
}

// ── App State & UI Types ──────────────────────────────────────────────────────

export interface GPUInfo {
  name: string;
  vram: number;
  vram_mb: number;
  vram_gb: number;
  driver: string;
  cuda_version: string;
  temperature: number;
  detected: boolean;
}

export interface ModelDownloadState {
  downloaded: boolean;
  downloading: boolean;
  progress: number; // 0-100
  speed_mbps: number;
  eta_seconds: number;
}

/**
 * ModelInfo is the runtime view of a model: combines VMR data with local download state.
 * This is what the UI actually works with.
 */
export interface ModelInfo {
  // From VMR identity
  id: string;
  display_name: string;
  family: string;
  variant: string;

  // Legacy/alias fields for UI compatibility
  name: string;
  minVram: number;
  size: number;
  resolution: string;
  fps: number;
  duration: string;
  huggingFaceRepo: string;

  // From VMR ui
  tier: string;
  description: string;
  pros: string[];
  cons: string[];
  tags: string[];
  accent_color: string;

  // From VMR distribution
  size_gb: number;
  repo: string;

  // From VMR hardware
  minimum_vram_gb: number;
  recommended_vram_gb: number;
  recommended_ram_gb: number;

  // From VMR capabilities
  capabilities: ModelCapabilities;

  // From VMR generation (for UI sliders)
  generation_defaults: Record<string, number>;
  generation_limits: Record<string, ModelSettingSpec>;
  profiles?: Record<string, GenerationProfile>;
  scheduler?: ModelSchedulerConfig;
  quantization?: ModelQuantizationConfig;

  // From local download state
  downloaded: boolean;
  downloading: boolean;
  progress: number;
  downloadProgress: number;
  speed_mbps: number;
  speedMbps?: number;
  eta_seconds: number;
  etaSeconds?: number;
}

export interface GenerationParams {
  steps?: number;
  cfg_scale?: number;
  width?: number;
  height?: number;
  num_frames?: number;
  fps?: number;
  profile?: 'fast' | 'balanced' | 'detailed';
  seed?: number;
  scheduler?: string;
  quantization?: string;
}

export type GenerationStatus =
  | "idle"
  | "queued"
  | "loading_model"
  | "generating"
  | "post_processing"
  | "done"
  | "error";

export interface GenerationJob {
  id: string;
  prompt: string;
  negative_prompt: string;
  model_id: string;
  status: GenerationStatus;
  progress: number; // 0-100
  eta: number; // seconds remaining
  startTime: number;
  endTime?: number;
  outputPath?: string;
  thumbnailUrl?: string;
  error?: string;
}

export interface AppSettings {
  outputDir: string;
  autoStart: boolean;
  theme: "dark";
  maxConcurrentJobs: number;
  defaultModel: string;
  watermark: boolean;
}

export type AppView = "setup" | "main" | "settings" | "models" | "license";

export interface SidecarStatus {
  running: boolean;
  port: number;
  comfyui_ready: boolean;
  version: string;
  python_version: string;
}

export interface LicenseInfo {
  key: string;
  valid: boolean;
  tier: "free" | "pro" | "enterprise";
  expires_at?: string;
  features: string[];
}

// ── WebSocket & HTTP Types ───────────────────────────────────────────────────

export interface WebSocketMessage {
  type:
    | "connection_established"
    | "download_progress"
    | "job_status"
    | "gpu_stats";
  [key: string]: any;
}

export interface DownloadProgressMessage {
  type: "download_progress";
  model_id: string;
  progress: number;
  speed_mbps: number;
  eta_seconds: number;
  downloading: boolean;
  downloaded: boolean;
}

export interface JobStatusMessage {
  type: "job_status";
  job_id: string;
  status: GenerationStatus;
  progress: number;
  eta: number;
  outputPath?: string;
  error?: string;
}
