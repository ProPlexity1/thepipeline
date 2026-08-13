import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Wand2, ChevronDown, Clock, Loader2, CheckCircle2,
  AlertCircle, Play, ImageIcon, Trash2, Copy, Download,
  Maximize2, X, Film, Zap, Scale, Sparkles, Sliders, Settings2
} from 'lucide-react';
import type { GenerationJob, ModelInfo, GenerationStatus, GenerationProfile } from '../types';
import { cn } from '../utils/cn';
import { convertFileSrc } from '@tauri-apps/api/core';

interface GeneratePanelProps {
  prompt: string;
  negativePrompt: string;
  selectedModel: string;
  models: ModelInfo[];
  jobs: GenerationJob[];
  galleryItems: GenerationJob[];
  onPromptChange: (p: string) => void;
  onNegativePromptChange: (p: string) => void;
  onSelectModel: (m: string) => void;
  onGenerate: (prompt: string, negPrompt: string, model: string, settings: Record<string, any>) => void;
}

const STATUS_CONFIG: Record<GenerationStatus, { label: string; color: string; icon: React.ReactNode }> = {
  idle: { label: 'Idle', color: 'text-text-muted', icon: null },
  queued: { label: 'Queued', color: 'text-accent-amber', icon: <Clock className="h-3.5 w-3.5" /> },
  loading_model: { label: 'Loading Model', color: 'text-accent-cyan', icon: <Loader2 className="h-3.5 w-3.5 animate-spin" /> },
  generating: { label: 'Generating', color: 'text-accent-purple', icon: <Loader2 className="h-3.5 w-3.5 animate-spin" /> },
  post_processing: { label: 'Post-Processing', color: 'text-accent-blue', icon: <Loader2 className="h-3.5 w-3.5 animate-spin" /> },
  done: { label: 'Complete', color: 'text-accent-green', icon: <CheckCircle2 className="h-3.5 w-3.5" /> },
  error: { label: 'Error', color: 'text-accent-red', icon: <AlertCircle className="h-3.5 w-3.5" /> },
};

const PROMPT_SUGGESTIONS = [
  'A majestic dragon flying over snow-capped mountains at dawn',
  'A realistic macro timelapse of a closed flower bud in a sunlit meadow. The flower actively blooms throughout the entire shot: the petals slowly unfold one by one, the stem gently sways in a light breeze, nearby leaves move naturally, and tiny insects fly through the foreground. The camera performs a very subtle forward push-in while maintaining focus on the flower. Continuous visible motion from beginning to end, realistic organic movement, natural physics, cinematic photography, detailed textures, photorealistic.',
  'Underwater scene with colorful coral reef and tropical fish',
];

type ProfileKey = 'fast' | 'balanced' | 'detailed';

const PRESET_ICONS: Record<ProfileKey, React.ReactNode> = {
  fast: <Zap className="h-4 w-4 text-amber-400" />,
  balanced: <Scale className="h-4 w-4 text-cyan-400" />,
  detailed: <Sparkles className="h-4 w-4 text-purple-400" />,
};

export default function GeneratePanel({
  prompt,
  negativePrompt,
  selectedModel,
  models,
  jobs,
  galleryItems,
  onPromptChange,
  onNegativePromptChange,
  onSelectModel,
  onGenerate,
}: GeneratePanelProps) {
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [selectedGalleryItem, setSelectedGalleryItem] = useState<GenerationJob | null>(null);
  const [durations, setDurations] = useState<Record<string, number>>({});

  // Engine v2 State
  const [selectedProfile, setSelectedProfile] = useState<ProfileKey>('balanced');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced mode overrides state
  const [steps, setSteps] = useState<number>(20);
  const [cfgScale, setCfgScale] = useState<number>(4.0);
  const [width, setWidth] = useState<number>(640);
  const [height, setHeight] = useState<number>(352);
  const [numFrames, setNumFrames] = useState<number>(49);
  const [fps, setFps] = useState<number>(16);
  const [seed, setSeed] = useState<string>('');
  const [selectedScheduler, setSelectedScheduler] = useState<string>('Default');
  const [selectedQuantization, setSelectedQuantization] = useState<string>('auto');

  const activeModel = models.find(m => m.id === selectedModel);
  const activeJobs = jobs.filter(j => j.status !== 'done' && j.status !== 'error' && j.status !== 'idle');
  const canGenerate = prompt.trim().length > 0 && activeModel?.downloaded;

  // Sync profile defaults whenever active model or selected profile changes
  useEffect(() => {
    if (activeModel?.profiles && activeModel.profiles[selectedProfile]) {
      const prof = activeModel.profiles[selectedProfile];
      setSteps(prof.steps);
      setCfgScale(prof.cfg_scale);
      setWidth(prof.width);
      setHeight(prof.height);
      setNumFrames(prof.num_frames);
      setFps(prof.fps);
      if (prof.scheduler) setSelectedScheduler(prof.scheduler);
      if (prof.quantization) setSelectedQuantization(prof.quantization);
    } else if (activeModel?.generation_defaults) {
      setSteps(activeModel.generation_defaults.steps || 20);
      setCfgScale(activeModel.generation_defaults.cfg_scale || 4.0);
      setWidth(activeModel.generation_defaults.width || 640);
      setHeight(activeModel.generation_defaults.height || 352);
      setNumFrames(activeModel.generation_defaults.num_frames || 49);
      setFps(activeModel.generation_defaults.fps || 16);
    }
  }, [selectedModel, selectedProfile, activeModel]);

  const handleLoadedMetadata = (jobId: string, e: React.SyntheticEvent<HTMLVideoElement>) => {
    const dur = e.currentTarget.duration;
    if (dur && isFinite(dur)) {
      setDurations(prev => (prev[jobId] !== undefined ? prev : { ...prev, [jobId]: dur }));
    }
  };

  const handleGenerate = () => {
    if (canGenerate) {
      const payload: Record<string, any> = {
        profile: selectedProfile,
        steps,
        cfg_scale: cfgScale,
        width,
        height,
        num_frames: numFrames,
        fps,
      };

      if (seed.trim() !== '') payload.seed = parseInt(seed, 10);
      if (selectedScheduler !== 'Default') payload.scheduler = selectedScheduler;
      if (selectedQuantization !== 'auto') payload.quantization = selectedQuantization;

      onGenerate(prompt, negativePrompt, selectedModel, payload);
      onPromptChange('');
    }
  };

  return (
    <div className="flex h-full flex-1 overflow-hidden">
      {/* Left: Prompt & Controls */}
      <div className="flex w-[420px] flex-col border-r border-border-dim flex-shrink-0">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Prompt input */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-text-muted uppercase tracking-wider">
              Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              placeholder="Describe the video scene in detail..."
              rows={4}
              className="w-full rounded-xl bg-bg-tertiary border border-border-dim px-4 py-3 text-sm text-text-primary placeholder-text-muted focus:border-accent-purple focus:outline-none focus:ring-1 focus:ring-accent-purple/50 resize-none transition-colors"
            />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {PROMPT_SUGGESTIONS.slice(0, 3).map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => onPromptChange(suggestion)}
                  className="rounded-md bg-bg-tertiary/50 border border-border-dim px-2 py-1 text-xs text-text-muted hover:text-text-secondary hover:border-border-active transition-colors truncate max-w-[180px]"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>

          {/* Model Selector */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-text-muted uppercase tracking-wider">
              Model
            </label>
            <div className="relative">
              <button
                onClick={() => setShowModelDropdown(!showModelDropdown)}
                className="flex w-full items-center justify-between rounded-xl bg-bg-tertiary border border-border-dim px-4 py-2.5 text-sm text-text-primary hover:border-border-active transition-colors"
              >
                <div className="flex items-center gap-2">
                  <TierBadge tier={activeModel?.tier || 'standard'} />
                  <span>{activeModel?.name || 'Select model'}</span>
                  {activeModel && !activeModel.downloaded && (
                    <span className="text-xs text-accent-amber">(not downloaded)</span>
                  )}
                </div>
                <ChevronDown className={cn('h-4 w-4 text-text-muted transition-transform', showModelDropdown && 'rotate-180')} />
              </button>
              <AnimatePresence>
                {showModelDropdown && (
                  <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    className="absolute top-full left-0 right-0 z-20 mt-1 rounded-xl bg-bg-secondary border border-border-dim shadow-xl overflow-hidden"
                  >
                    {models.filter(model => model.downloaded).length === 0 ? (
                      <div className="px-4 py-3 text-xs text-text-muted text-center">
                        No downloaded models. Go to Model Manager to download a model.
                      </div>
                    ) : (
                      models.filter(model => model.downloaded).map(model => (
                        <button
                          key={model.id}
                          onClick={() => { onSelectModel(model.id); setShowModelDropdown(false); }}
                          className={cn(
                            'flex w-full items-center justify-between px-4 py-3 text-sm hover:bg-bg-hover transition-colors',
                            model.id === selectedModel ? 'bg-accent-purple/10' : ''
                          )}
                        >
                          <div className="flex items-center gap-2">
                            <TierBadge tier={model.tier} />
                            <div className="text-left">
                              <span className="text-text-primary">{model.name}</span>
                              <div className="text-xs text-text-muted">{model.resolution} · {model.fps}fps · {model.duration}</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-accent-green">Ready</span>
                          </div>
                        </button>
                      ))
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Engine v2 Quality Presets */}
          <div>
            <label className="mb-2 block text-xs font-medium text-text-muted uppercase tracking-wider">
              Quality Preset
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['fast', 'balanced', 'detailed'] as ProfileKey[]).map((key) => {
                const isSelected = selectedProfile === key;
                const prof = activeModel?.profiles?.[key];
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setSelectedProfile(key)}
                    className={cn(
                      'flex flex-col items-center justify-center p-3 rounded-xl border transition-all text-center',
                      isSelected
                        ? 'bg-accent-purple/10 border-accent-purple text-text-primary shadow-sm shadow-accent-purple/20'
                        : 'bg-bg-tertiary/60 border-border-dim text-text-muted hover:border-border-active hover:text-text-secondary'
                    )}
                  >
                    <div className="mb-1">{PRESET_ICONS[key]}</div>
                    <span className="text-xs font-semibold capitalize">{key}</span>
                    <span className="text-[10px] text-text-muted mt-0.5">
                      {prof ? `${prof.steps}st · ${prof.width}x${prof.height}` : key}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Collapsible Advanced Mode */}
          <div className="rounded-xl border border-border-dim bg-bg-tertiary/40 overflow-hidden">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex w-full items-center justify-between px-4 py-3 text-xs font-medium text-text-secondary hover:bg-bg-hover transition-colors"
            >
              <div className="flex items-center gap-2">
                <Sliders className="h-3.5 w-3.5 text-accent-purple" />
                <span>Advanced Mode Parameters</span>
              </div>
              <ChevronDown className={cn('h-4 w-4 text-text-muted transition-transform', showAdvanced && 'rotate-180')} />
            </button>

            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="px-4 pb-4 pt-1 space-y-3 border-t border-border-dim/50"
                >
                  {/* Steps */}
                  <div>
                    <div className="flex justify-between text-xs text-text-muted mb-1">
                      <span>Inference Steps</span>
                      <span className="text-text-secondary font-mono">{steps}</span>
                    </div>
                    <input
                      type="range"
                      min={10}
                      max={100}
                      step={5}
                      value={steps}
                      onChange={(e) => setSteps(Number(e.target.value))}
                      className="w-full accent-accent-purple"
                    />
                  </div>

                  {/* CFG Scale */}
                  <div>
                    <div className="flex justify-between text-xs text-text-muted mb-1">
                      <span>CFG Guidance Scale</span>
                      <span className="text-text-secondary font-mono">{cfgScale.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min={1.0}
                      max={10.0}
                      step={0.5}
                      value={cfgScale}
                      onChange={(e) => setCfgScale(Number(e.target.value))}
                      className="w-full accent-accent-purple"
                    />
                  </div>

                  {/* Resolution */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-[11px] text-text-muted block mb-1">Width</label>
                      <input
                        type="number"
                        step={16}
                        value={width}
                        onChange={(e) => setWidth(Number(e.target.value))}
                        className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] text-text-muted block mb-1">Height</label>
                      <input
                        type="number"
                        step={16}
                        value={height}
                        onChange={(e) => setHeight(Number(e.target.value))}
                        className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none"
                      />
                    </div>
                  </div>

                  {/* Frames & FPS */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-[11px] text-text-muted block mb-1">Num Frames</label>
                      <input
                        type="number"
                        step={8}
                        value={numFrames}
                        onChange={(e) => setNumFrames(Number(e.target.value))}
                        className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="text-[11px] text-text-muted block mb-1">FPS</label>
                      <input
                        type="number"
                        min={8}
                        max={60}
                        value={fps}
                        onChange={(e) => setFps(Number(e.target.value))}
                        className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none"
                      />
                    </div>
                  </div>

                  {/* Seed */}
                  <div>
                    <label className="text-[11px] text-text-muted block mb-1">Seed (optional)</label>
                    <input
                      type="text"
                      placeholder="Random (-1 or blank)"
                      value={seed}
                      onChange={(e) => setSeed(e.target.value)}
                      className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none placeholder:text-text-muted/50"
                    />
                  </div>

                  {/* Scheduler & Quantization Selectors */}
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <label className="text-[11px] text-text-muted block mb-1">Scheduler</label>
                      <select
                        value={selectedScheduler}
                        onChange={(e) => setSelectedScheduler(e.target.value)}
                        className="w-full rounded-lg bg-bg-secondary border border-border-dim px-2 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none"
                      >
                        <option value="Default">Default</option>
                        <option value="FlowMatch">FlowMatch</option>
                        <option value="DDIM">DDIM</option>
                        <option value="Euler">Euler</option>
                        <option value="DPMSolver">DPMSolver</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-[11px] text-text-muted block mb-1">Quantization</label>
                      <select
                        value={selectedQuantization}
                        onChange={(e) => setSelectedQuantization(e.target.value)}
                        className="w-full rounded-lg bg-bg-secondary border border-border-dim px-2 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none"
                      >
                        <option value="auto">Auto</option>
                        <option value="fp16">FP16</option>
                        <option value="bf16">BF16</option>
                        <option value="fp8">FP8</option>
                        <option value="quanto_int8">Quanto INT8</option>
                        <option value="none">None (FP32)</option>
                      </select>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Active Jobs */}
          {activeJobs.length > 0 && (
            <div>
              <label className="mb-2 block text-xs font-medium text-text-muted uppercase tracking-wider">
                Active Jobs ({activeJobs.length})
              </label>
              <div className="space-y-2">
                {activeJobs.map(job => {
                  const statusConf = STATUS_CONFIG[job.status];
                  return (
                    <motion.div
                      key={job.id}
                      layout
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="rounded-xl bg-bg-tertiary/50 border border-border-dim p-3 space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <div className={cn('flex items-center gap-1.5 text-xs font-medium', statusConf.color)}>
                          {statusConf.icon}
                          {statusConf.label}
                        </div>
                        {job.eta > 0 && (
                          <span className="text-xs text-text-muted">~{job.eta}s remaining</span>
                        )}
                      </div>
                      <p className="text-xs text-text-secondary line-clamp-1">{job.prompt}</p>
                      {job.status === 'generating' && (
                        <div className="h-1.5 rounded-full bg-bg-primary overflow-hidden">
                          <motion.div
                            className="h-full rounded-full bg-gradient-to-r from-accent-purple to-accent-blue progress-striped"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                      )}
                    </motion.div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Generate Button */}
        <div className="border-t border-border-dim p-4">
          <motion.button
            whileHover={canGenerate ? { scale: 1.01 } : {}}
            whileTap={canGenerate ? { scale: 0.99 } : {}}
            onClick={handleGenerate}
            disabled={!canGenerate}
            className={cn(
              'flex w-full items-center justify-center gap-2 rounded-xl py-3 font-medium text-sm transition-all',
              canGenerate
                ? 'bg-gradient-to-r from-accent-purple to-accent-blue text-white shadow-lg shadow-accent-purple/25 hover:shadow-accent-purple/40'
                : 'bg-bg-tertiary text-text-muted cursor-not-allowed'
            )}
          >
            <Wand2 className="h-4 w-4" />
            Generate Video
          </motion.button>
          {!activeModel?.downloaded && activeModel && (
            <p className="mt-2 text-center text-xs text-accent-amber">
              Download the {activeModel.name} model first
            </p>
          )}
        </div>
      </div>

      {/* Right: Gallery */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-border-dim px-4 py-3">
          <div className="flex items-center gap-2">
            <ImageIcon className="h-4 w-4 text-text-muted" />
            <h3 className="text-sm font-medium text-text-primary">Output Gallery</h3>
            <span className="text-xs text-text-muted">({galleryItems.length} videos)</span>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {galleryItems.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <Film className="h-12 w-12 text-text-muted/30 mb-3" />
              <p className="text-sm text-text-muted">No videos generated yet</p>
              <p className="text-xs text-text-muted/60 mt-1">Enter a prompt and click Generate to create your first video</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              {galleryItems.map((item, i) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.05 }}
                  className="group relative rounded-xl overflow-hidden border border-border-dim bg-bg-tertiary cursor-pointer hover:border-accent-purple/50 transition-colors"
                  onClick={() => setSelectedGalleryItem(item)}
                >
                  <div className="aspect-video relative">
                    {item.outputPath && (
                      <video
                        src={convertFileSrc(item.outputPath)}
                        className="h-full w-full object-cover"
                        muted
                        preload="metadata"
                        onLoadedMetadata={(e) => handleLoadedMetadata(item.id, e)}
                      />
                    )}
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 flex items-center justify-center transition-colors">
                      <Play className="h-8 w-8 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                    {durations[item.id] !== undefined && (
                      <div className="absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 text-xs text-white">
                        {Math.floor(durations[item.id] / 60)}:{String(Math.floor(durations[item.id] % 60)).padStart(2, '0')}
                      </div>
                    )}
                  </div>
                  <div className="p-2.5">
                    <p className="text-xs text-text-secondary line-clamp-2 leading-relaxed">{item.prompt}</p>
                    <div className="mt-1.5 flex items-center justify-between">
                      <TierBadge tier={models.find(m => m.id === item.model_id)?.tier || 'standard'} />
                      <span className="text-xs text-text-muted">
                        {new Date(item.endTime || item.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Lightbox */}
      <AnimatePresence>
        {selectedGalleryItem && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
            onClick={() => setSelectedGalleryItem(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative max-w-4xl w-full mx-4 rounded-2xl bg-bg-secondary border border-border-dim overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-border-dim px-4 py-3">
                <span className="text-sm font-medium text-text-primary">Video Preview</span>
                <div className="flex items-center gap-1">
                  <button className="p-1.5 rounded-md text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors">
                    <Copy className="h-4 w-4" />
                  </button>
                  <button className="p-1.5 rounded-md text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors">
                    <Download className="h-4 w-4" />
                  </button>
                  <button className="p-1.5 rounded-md text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors">
                    <Maximize2 className="h-4 w-4" />
                  </button>
                  <button className="p-1.5 rounded-md text-text-muted hover:text-text-primary hover:bg-bg-hover transition-colors">
                    <Trash2 className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => setSelectedGalleryItem(null)}
                    className="ml-1 p-1.5 rounded-md text-text-muted hover:text-accent-red hover:bg-accent-red/10 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
              {selectedGalleryItem.outputPath && (
                <video
                  src={convertFileSrc(selectedGalleryItem.outputPath)}
                  className="h-full w-full object-contain"
                  controls
                  autoPlay
                />
              )}
              <div className="border-t border-border-dim p-4">
                <p className="text-sm text-text-primary mb-2">{selectedGalleryItem.prompt}</p>
                <div className="flex items-center gap-3 text-xs text-text-muted">
                  <span>Model: {models.find(m => m.id === selectedGalleryItem.model_id)?.name}</span>
                  <span>•</span>
                  <span>{selectedGalleryItem.outputPath}</span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function TierBadge({ tier }: { tier: string }) {
  const colors: Record<string, string> = {
    lite: 'bg-green-500/15 text-green-400 border-green-500/30',
    standard: 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30',
    pro: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
    ultra: 'bg-purple-500/15 text-purple-400 border-purple-500/30',
  };
  return (
    <span className={cn('rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase', colors[tier] || colors.standard)}>
      {tier}
    </span>
  );
}
