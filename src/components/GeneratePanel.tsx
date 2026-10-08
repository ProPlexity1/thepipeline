import { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Wand2, ChevronDown, Loader2, CheckCircle2, Play, Trash2, Copy,
  FolderOpen, X, Film, Zap, Scale, Sparkles, Sliders, RotateCcw, Download, ImagePlus, Upload,
} from 'lucide-react';
import type { GenerationJob, ModelInfo, OutputItem, ImageItem } from '../types';
import { cn } from '../utils/cn';
import { formatDuration, videoUrl, imageUrl, post } from '../api';
import JobCard, { clock } from './JobCard';

interface GeneratePanelProps {
  prompt: string;
  negativePrompt: string;
  selectedModel: string;
  models: ModelInfo[];
  jobs: GenerationJob[];
  outputs: OutputItem[];
  onPromptChange: (p: string) => void;
  onNegativePromptChange: (p: string) => void;
  onSelectModel: (m: string) => void;
  onGenerate: (prompt: string, negPrompt: string, model: string, settings: Record<string, any>) => Promise<string | null>;
  onCancelJob: (id: string) => void;
  onDismissJob: (id: string) => void;
  onDeleteOutput: (name: string) => void;
  onRevealOutput: (name: string) => void;
  onGoToModels: () => void;
  onNotify: (kind: 'error' | 'success' | 'info', text: string) => void;
  onEstimate: (modelId: string, settings: Record<string, any>) => Promise<number | null>;
  onEnhance: (source: string, enhancerId: string, target: string, prompt: string) => void;
  images: ImageItem[];
  onRefreshImages: () => void;
}

const PROMPT_SUGGESTIONS = [
  'A majestic dragon flying over snow-capped mountains at dawn, cinematic, volumetric light',
  'Macro shot of a flower bud slowly blooming in a sunlit meadow, petals unfolding, gentle breeze',
  'Underwater coral reef with schools of colorful tropical fish, sun rays through the water',
];

type ProfileKey = 'fast' | 'balanced' | 'detailed';

const PRESETS: { key: ProfileKey; label: string; icon: React.ReactNode }[] = [
  { key: 'fast', label: 'Fast', icon: <Zap className="h-4 w-4 text-accent-amber" /> },
  { key: 'balanced', label: 'Balanced', icon: <Scale className="h-4 w-4 text-accent-cyan" /> },
  { key: 'detailed', label: 'Best', icon: <Sparkles className="h-4 w-4 text-accent-purple" /> },
];

const snap = (v: number, spec?: { min: number; max: number; step: number }) => {
  if (!spec) return v;
  const step = spec.step || 1;
  const s = Math.round((v - spec.min) / step) * step + spec.min;
  return Math.min(spec.max, Math.max(spec.min, s));
};

export default function GeneratePanel(props: GeneratePanelProps) {
  const {
    prompt, negativePrompt, selectedModel, models, jobs, outputs,
    onPromptChange, onNegativePromptChange, onSelectModel, onGenerate,
    onCancelJob, onDismissJob, onDeleteOutput, onRevealOutput, onGoToModels, onNotify, onEstimate, onEnhance,
    images, onRefreshImages,
  } = props;

  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [preview, setPreview] = useState<OutputItem | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [durations, setDurations] = useState<Record<string, number>>({});
  const [profile, setProfile] = useState<ProfileKey>('balanced');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [steps, setSteps] = useState(20);
  const [cfgScale, setCfgScale] = useState(4);
  const [width, setWidth] = useState(640);
  const [height, setHeight] = useState(352);
  const [numFrames, setNumFrames] = useState(49);
  const [fps, setFps] = useState(16);
  const [seed, setSeed] = useState('');

  const installed = models.filter((m) => m.downloaded && m.kind === 'video');
  const [startImage, setStartImage] = useState<string | null>(null);
  const [pickingImage, setPickingImage] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);
  const enhancers = models.filter((m) => m.kind === 'enhancer');
  const enhanceOptions = enhancers.filter((m) => m.downloaded).flatMap((m) =>
    m.enhance_targets.map((t) => ({ value: `${m.id}:${t.id}`, enhancerId: m.id, target: t.id, label: `${m.name} → ${t.label}` })));

  // "Enhance after generating": remembered between sessions.
  const [enhanceChoice, setEnhanceChoice] = useState<string>(() => {
    try { return localStorage.getItem('thepipeline.enhance') || ''; } catch { return ''; }
  });
  const chooseEnhance = (v: string) => {
    setEnhanceChoice(v);
    try { localStorage.setItem('thepipeline.enhance', v); } catch { /* storage unavailable */ }
  };
  const activeEnhance = enhanceOptions.find((o) => o.value === enhanceChoice);
  const [enhanceMenu, setEnhanceMenu] = useState(false);
  const activeModel = models.find((m) => m.id === selectedModel);
  const limits = activeModel?.generation_limits || {};
  const hidden = new Set(activeModel?.hidden_controls || []);
  const show = (k: string) => !hidden.has(k) && limits[k] && limits[k].min !== limits[k].max;

  const visibleJobs = jobs.filter((j) => j.status !== 'idle');
  const busy = jobs.some((j) => ['queued', 'loading_model', 'generating', 'post_processing'].includes(j.status));
  const takesImage = !!activeModel?.capabilities.image_to_video;
  const needsImage = takesImage && activeModel?.capabilities.text_to_video === false;
  const canGenerate = prompt.trim().length > 0 && !!activeModel?.downloaded && !submitting && (!needsImage || !!startImage);
  useEffect(() => { if (!takesImage) setStartImage(null); }, [takesImage]);

  const uploadStart = async (file?: File) => {
    if (!file || !file.type.startsWith('image/')) return;
    const data_url = await new Promise<string>((res, rej) => {
      const r = new FileReader(); r.onload = () => res(String(r.result)); r.onerror = () => rej(r.error); r.readAsDataURL(file);
    });
    try {
      const up = await post<{ name: string }>('/images/upload', { data_url });
      onRefreshImages();
      setStartImage(up.name);
      setPickingImage(false);
    } catch (e: any) {
      onNotify('error', e.message);
    }
  };

  // Quality presets set resolution/steps. Length is the user's own choice: once
  // they move the slider it survives preset changes, and resets only when the
  // model changes (each model has different length limits).
  const lengthChosen = useRef(false);
  useEffect(() => { lengthChosen.current = false; }, [selectedModel]);
  useEffect(() => {
    if (!activeModel) return;
    const p = activeModel.profiles?.[profile];
    const d = activeModel.generation_defaults || {};
    const src: Record<string, any> = p || d;
    setSteps(src.steps ?? d.steps ?? 20);
    setCfgScale(src.cfg_scale ?? d.cfg_scale ?? 4);
    setWidth(src.width ?? d.width ?? 640);
    setHeight(src.height ?? d.height ?? 352);
    if (!lengthChosen.current) setNumFrames(src.num_frames ?? d.num_frames ?? 49);
    setFps(src.fps ?? d.fps ?? 16);
  }, [selectedModel, profile, activeModel?.id]);

  useEffect(() => {
    if (!preview) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setPreview(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [preview]);

  // How long the current settings usually take on this PC (from past runs).
  const [estimate, setEstimate] = useState<number | null>(null);
  useEffect(() => {
    if (!activeModel?.downloaded) { setEstimate(null); return; }
    const t = setTimeout(async () => {
      setEstimate(await onEstimate(selectedModel, {
        profile, steps, width, height, num_frames: numFrames, fps,
      }));
    }, 300);
    return () => clearTimeout(t);
  }, [selectedModel, activeModel?.downloaded, profile, steps, width, height, numFrames, fps, jobs.length]);

  const clipSeconds = useMemo(() => (fps > 0 ? numFrames / fps : 0), [numFrames, fps]);

  const handleGenerate = async () => {
    if (!canGenerate || !activeModel) return;
    const payload: Record<string, any> = {
      profile,
      steps: snap(steps, limits.steps),
      cfg_scale: snap(cfgScale, limits.cfg_scale),
      width: snap(width, limits.width),
      height: snap(height, limits.height),
      num_frames: snap(numFrames, limits.num_frames),
      fps: snap(fps, limits.fps),
    };
    const s = parseInt(seed, 10);
    if (!isNaN(s) && s >= 0) payload.seed = s;
    if (takesImage && startImage) payload.start_image = startImage;
    if (activeEnhance) {
      payload.enhancer_id = activeEnhance.enhancerId;
      payload.enhance_target = activeEnhance.target;
    }
    setSubmitting(true);
    await onGenerate(prompt.trim(), negativePrompt, selectedModel, payload);
    setSubmitting(false);
  };

  const modelName = (id: string) => models.find((m) => m.id === id)?.name || id || 'Unknown model';

  return (
    <div className="flex h-full flex-1 overflow-hidden">
      {/* Left: prompt & controls */}
      <div className="flex w-[340px] xl:w-[400px] flex-col border-r border-border-dim bg-bg-secondary flex-shrink-0">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div>
            <label htmlFor="prompt" className="mb-1.5 block text-xs font-medium text-text-muted uppercase tracking-wider">
              Describe your video
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => onPromptChange(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleGenerate(); }}
              placeholder="A red fox trotting through a snowy pine forest at sunrise, soft light, slow camera pan…"
              rows={5}
              maxLength={4000}
              className="w-full rounded-xl bg-bg-tertiary border border-border-dim px-4 py-3 text-sm text-text-primary placeholder-text-muted focus:border-accent-purple/70 focus:outline-none focus:ring-2 focus:ring-accent-purple/20 resize-none"
            />
            <div className="mt-1 flex items-center justify-between text-[11px] text-text-muted">
              <span>Describe the subject, motion, setting and camera. Ctrl+Enter to generate.</span>
            </div>
            {!prompt && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {PROMPT_SUGGESTIONS.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => onPromptChange(s)}
                    title={s}
                    className="rounded-md bg-bg-tertiary/50 border border-border-dim px-2 py-1 text-xs text-text-muted hover:text-text-secondary hover:border-border-active transition-colors truncate max-w-[170px]"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Model selector */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-text-muted uppercase tracking-wider">Model</label>
            {installed.length === 0 ? (
              <button
                onClick={onGoToModels}
                className="flex w-full items-center justify-between rounded-xl border border-dashed border-accent-purple/40 bg-accent-purple/5 px-4 py-3 text-sm text-accent-purple hover:bg-accent-purple/10 transition-colors"
              >
                <span>No models installed yet. Get one</span>
                <Download className="h-4 w-4" />
              </button>
            ) : (
              <div className="relative">
                <button
                  onClick={() => setShowModelDropdown(!showModelDropdown)}
                  aria-expanded={showModelDropdown}
                  className="flex w-full items-center justify-between rounded-xl bg-bg-tertiary border border-border-dim px-4 py-2.5 text-sm text-text-primary hover:border-border-active transition-colors"
                >
                  <div className="text-left">
                    <div>{activeModel?.name || 'Select model'}</div>
                    {activeModel?.speed_label && <div className="text-[11px] text-text-muted">{activeModel.speed_label}</div>}
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
                      {installed.map((m) => (
                        <button
                          key={m.id}
                          onClick={() => { onSelectModel(m.id); setShowModelDropdown(false); }}
                          className={cn('flex w-full items-center justify-between px-4 py-3 text-sm hover:bg-bg-hover transition-colors text-left',
                            m.id === selectedModel && 'bg-accent-purple/10')}
                        >
                          <div>
                            <div className="text-text-primary">{m.name}</div>
                            <div className="text-xs text-text-muted">
                              {m.capabilities.text_to_video === false ? 'From an image · ' : m.capabilities.image_to_video ? 'Text or image · ' : ''}
                              {m.resolution} · {m.speed_label || m.duration}
                            </div>
                          </div>
                          {m.id === selectedModel && <CheckCircle2 className="h-4 w-4 text-accent-purple" />}
                        </button>
                      ))}
                      <button
                        onClick={() => { setShowModelDropdown(false); onGoToModels(); }}
                        className="w-full border-t border-border-dim px-4 py-2.5 text-left text-xs text-accent-purple hover:bg-bg-hover"
                      >
                        Get more models…
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>

          {/* Starting image (image-to-video models) */}
          {activeModel && takesImage && (
            <div>
              <label className="mb-1.5 block text-xs font-medium text-text-muted uppercase tracking-wider">
                Starting image {needsImage ? <span className="normal-case tracking-normal text-accent-amber">· required</span> : <span className="normal-case tracking-normal">· optional</span>}
              </label>
              {startImage ? (
                <div className="group relative overflow-hidden rounded-xl border border-border-dim bg-black">
                  <img src={imageUrl(startImage)} alt="Starting image" className="h-32 w-full object-cover" />
                  <div className="absolute inset-x-0 bottom-0 flex justify-end gap-1 bg-gradient-to-t from-black/70 to-transparent p-1.5">
                    <button onClick={() => setPickingImage(true)} className="rounded bg-black/70 px-2 py-1 text-[11px] text-white hover:bg-accent-purple">Change</button>
                    <button onClick={() => setStartImage(null)} className="rounded bg-black/70 px-2 py-1 text-[11px] text-white hover:bg-accent-red">Remove</button>
                  </div>
                </div>
              ) : (
                <button onClick={() => setPickingImage(true)}
                  className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-border-active bg-bg-tertiary/40 px-4 py-4 text-sm text-text-secondary hover:border-accent-purple/50 hover:text-text-primary">
                  <ImagePlus className="h-4 w-4" /> Choose a starting image
                </button>
              )}
              <p className="mt-1 text-[11px] text-text-muted">The video starts from this picture: same people, products and look. Describe the motion above.</p>
              <input ref={uploadRef} type="file" accept="image/png,image/jpeg,image/webp" hidden
                onChange={(e) => { uploadStart(e.target.files?.[0]); e.target.value = ''; }} />
              <AnimatePresence>
                {pickingImage && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6" onClick={() => setPickingImage(false)}>
                    <motion.div initial={{ scale: 0.97, y: 8 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.97 }}
                      onClick={(e) => e.stopPropagation()}
                      className="flex max-h-[80vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-border-dim bg-bg-secondary">
                      <div className="flex items-center justify-between border-b border-border-dim px-4 py-3">
                        <h3 className="text-sm font-semibold text-text-primary">Choose a starting image</h3>
                        <div className="flex items-center gap-2">
                          <button onClick={() => uploadRef.current?.click()}
                            className="flex items-center gap-1.5 rounded-md border border-border-dim px-2.5 py-1 text-xs text-text-secondary hover:text-text-primary">
                            <Upload className="h-3.5 w-3.5" /> Upload
                          </button>
                          <button onClick={() => setPickingImage(false)} aria-label="Close" className="text-text-muted hover:text-text-primary"><X className="h-4 w-4" /></button>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-2 overflow-y-auto p-3">
                        {images.length === 0 && <p className="col-span-3 p-6 text-center text-sm text-text-muted">No images yet. Make some on the Images page, or upload one.</p>}
                        {images.map((im) => (
                          <button key={im.name} onClick={() => { setStartImage(im.name); setPickingImage(false); }}
                            className="aspect-video overflow-hidden rounded-lg border border-border-dim bg-black hover:border-accent-purple">
                            <img src={imageUrl(im.name)} alt={im.prompt || im.name} loading="lazy" className="h-full w-full object-cover" />
                          </button>
                        ))}
                      </div>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Quality presets */}
          {activeModel && (
            <div>
              <label className="mb-2 block text-xs font-medium text-text-muted uppercase tracking-wider">Quality</label>
              <div className="grid grid-cols-3 gap-2">
                {PRESETS.map(({ key, label, icon }) => {
                  const p = activeModel.profiles?.[key];
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setProfile(key)}
                      aria-pressed={profile === key}
                      className={cn(
                        'relative flex flex-col items-center justify-center p-3 rounded-xl border text-center',
                        profile === key
                          ? 'border-accent-purple/60 text-text-primary'
                          : 'bg-bg-tertiary border-border-dim text-text-secondary hover:border-border-active hover:text-text-primary'
                      )}
                    >
                      {profile === key && (
                        <motion.span
                          layoutId="preset-active"
                          className="absolute inset-0 rounded-xl bg-accent-purple/12"
                          transition={{ type: 'spring', stiffness: 500, damping: 38 }}
                        />
                      )}
                      <div className="relative mb-1">{icon}</div>
                      <span className="relative text-xs font-semibold">{label}</span>
                      {p && (
                        <span className="relative text-[10px] text-text-muted mt-0.5">
                          {p.height}p · {(p.num_frames / (p.fps || 24)).toFixed(1)}s
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Clip length */}
          {activeModel && show('num_frames') && (
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="font-medium text-text-muted uppercase tracking-wider">Length</span>
                <span className="text-text-secondary font-mono">{clipSeconds.toFixed(1)}s</span>
              </div>
              <input
                type="range"
                aria-label="Clip length"
                min={limits.num_frames.min}
                max={limits.num_frames.max}
                step={limits.num_frames.step}
                value={numFrames}
                onChange={(e) => { lengthChosen.current = true; setNumFrames(Number(e.target.value)); }}
                className="w-full accent-accent-purple"
              />
              <p className="text-[11px] text-text-muted mt-0.5">Longer clips take proportionally longer to make.</p>
            </div>
          )}

          {/* Advanced */}
          {activeModel && (
            <div className="rounded-xl border border-border-dim bg-bg-tertiary/60 overflow-hidden">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                aria-expanded={showAdvanced}
                className="flex w-full items-center justify-between px-4 py-3 text-xs font-medium text-text-secondary hover:bg-bg-hover transition-colors"
              >
                <span className="flex items-center gap-2"><Sliders className="h-3.5 w-3.5 text-accent-purple" />Advanced</span>
                <ChevronDown className={cn('h-4 w-4 text-text-muted transition-transform', showAdvanced && 'rotate-180')} />
              </button>
              <AnimatePresence initial={false}>
                {showAdvanced && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1, transition: { duration: 0.25, ease: [0.22, 1, 0.36, 1] } }}
                    exit={{ height: 0, opacity: 0, transition: { duration: 0.18, ease: [0.4, 0, 1, 1] } }}
                    className="px-4 pb-4 pt-2 space-y-3 border-t border-border-dim/50 overflow-hidden"
                  >
                    {show('steps') && (
                      <RangeRow label="Detail steps" value={steps} spec={limits.steps} onChange={setSteps}
                        hint="More steps = finer detail, slower." />
                    )}
                    {show('cfg_scale') && (
                      <RangeRow label="Prompt strength" value={cfgScale} spec={limits.cfg_scale} onChange={setCfgScale}
                        format={(v) => v.toFixed(1)} hint="Higher follows the prompt more literally." />
                    )}
                    {(show('width') || show('height')) && (
                      <div className="grid grid-cols-2 gap-2">
                        {(['width', 'height'] as const).map((k) => (
                          <div key={k}>
                            <label className="text-[11px] text-text-muted block mb-1 capitalize">{k}</label>
                            <input
                              type="number"
                              min={limits[k]?.min}
                              max={limits[k]?.max}
                              step={limits[k]?.step}
                              value={k === 'width' ? width : height}
                              onChange={(e) => (k === 'width' ? setWidth : setHeight)(Number(e.target.value))}
                              onBlur={(e) => (k === 'width' ? setWidth : setHeight)(snap(Number(e.target.value), limits[k]))}
                              className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none"
                            />
                          </div>
                        ))}
                      </div>
                    )}
                    {show('fps') && (
                      <RangeRow label="Frames per second" value={fps} spec={limits.fps} onChange={setFps} />
                    )}
                    {!hidden.has('negative_prompt') && (
                      <div>
                        <label className="text-[11px] text-text-muted block mb-1">Avoid (negative prompt)</label>
                        <input
                          type="text"
                          value={negativePrompt}
                          onChange={(e) => onNegativePromptChange(e.target.value)}
                          placeholder="Uses the model's recommended default"
                          className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none placeholder:text-text-muted/50"
                        />
                      </div>
                    )}
                    <div>
                      <label className="text-[11px] text-text-muted block mb-1">Seed</label>
                      <input
                        type="text"
                        inputMode="numeric"
                        placeholder="Random"
                        value={seed}
                        onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ''))}
                        className="w-full rounded-lg bg-bg-secondary border border-border-dim px-3 py-1.5 text-xs text-text-primary focus:border-accent-purple focus:outline-none placeholder:text-text-muted/50"
                      />
                      <p className="text-[11px] text-text-muted mt-0.5">Same seed + same prompt gives the same video.</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Enhance after generating */}
          {activeModel && (
            <div>
              <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-text-muted uppercase tracking-wider">
                <Sparkles className="h-3.5 w-3.5 text-accent-cyan" /> Enhance after generating
              </label>
              {enhanceOptions.length > 0 ? (
                <>
                  <select
                    value={activeEnhance ? enhanceChoice : ''}
                    onChange={(e) => chooseEnhance(e.target.value)}
                    className="w-full rounded-xl bg-bg-tertiary border border-border-dim px-3 py-2 text-sm text-text-primary focus:border-accent-purple/70 focus:outline-none"
                  >
                    <option value="">Off: keep the original only</option>
                    {enhanceOptions.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                  </select>
                  <p className="text-[11px] text-text-muted mt-1">
                    {activeEnhance
                      ? 'An upscaled, more detailed copy is made right after each video. The original is kept too.'
                      : 'You can also enhance any finished video from its preview.'}
                  </p>
                </>
              ) : (
                <button
                  onClick={onGoToModels}
                  className="flex w-full items-center justify-between rounded-xl border border-dashed border-accent-cyan/40 bg-accent-cyan/5 px-3 py-2.5 text-left text-xs text-text-secondary hover:bg-accent-cyan/10"
                >
                  <span>Get sharper, higher-resolution videos: install an enhancer</span>
                  <Download className="h-3.5 w-3.5 text-accent-cyan" />
                </button>
              )}
            </div>
          )}

          {/* Jobs */}
          {visibleJobs.length > 0 && (
            <div>
              <label className="mb-2 block text-xs font-medium text-text-muted uppercase tracking-wider">Jobs</label>
              <div className="space-y-2">
                <AnimatePresence initial={false}>
                  {visibleJobs.map((job) => (
                    <JobCard key={job.id} job={job} onCancel={() => onCancelJob(job.id)} onDismiss={() => onDismissJob(job.id)} />
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}
        </div>

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
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
            {busy ? 'Add to queue' : 'Generate video'}
          </motion.button>
          <p className="mt-2 text-center text-xs text-text-muted">
            {!activeModel?.downloaded
              ? 'Install a model to start generating.'
              : !prompt.trim()
              ? 'Write a prompt to start.'
              : busy
              ? 'Videos are made one at a time; this one will start after the current job.'
              : estimate
              ? `Usually takes about ${clock(estimate)} on this PC`
              : activeModel.speed_label}
          </p>
        </div>
      </div>

      {/* Right: gallery */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between border-b border-border-dim bg-bg-secondary/60 px-4 py-3">
          <div className="flex items-center gap-2">
            <Film className="h-4 w-4 text-text-muted" />
            <h3 className="text-sm font-medium text-text-primary">Your videos</h3>
            <span className="text-xs text-text-muted">({outputs.length})</span>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {outputs.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <Film className="h-12 w-12 text-text-muted/30 mb-3" />
              <p className="text-sm text-text-muted">No videos yet</p>
              <p className="text-xs text-text-muted/60 mt-1">Describe a scene on the left and press Generate.</p>
            </div>
          ) : (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-3">
              {outputs.map((item, i) => (
                <motion.button
                  key={item.name}
                  layout
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0, transition: { delay: Math.min(i, 8) * 0.03, duration: 0.25, ease: [0.22, 1, 0.36, 1] } }}
                  whileHover={{ y: -2 }}
                  className="surface group relative text-left rounded-xl overflow-hidden border border-border-dim hover:border-border-active hover:shadow-lg hover:shadow-black/30"
                  onClick={() => { setPreview(item); setConfirmDelete(false); setEnhanceMenu(false); }}
                >
                  <div className="aspect-video relative bg-black">
                    <video
                      src={`${videoUrl(item.name)}#t=0.1`}
                      className="h-full w-full object-cover"
                      muted
                      preload="metadata"
                      onMouseEnter={(e) => e.currentTarget.play().catch(() => {})}
                      onMouseLeave={(e) => { e.currentTarget.pause(); e.currentTarget.currentTime = 0; }}
                      onLoadedMetadata={(e) => {
                        const d = e.currentTarget.duration;
                        if (isFinite(d)) setDurations((p) => ({ ...p, [item.name]: d }));
                      }}
                    />
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/0 group-hover:bg-black/10 transition-colors">
                      <Play className="h-8 w-8 text-white opacity-70 group-hover:opacity-0 transition-opacity drop-shadow" />
                    </div>
                    {item.enhance_target && (
                      <div className="absolute top-1.5 left-1.5 flex items-center gap-1 rounded-md bg-black/70 px-1.5 py-0.5 text-[10px] font-semibold text-accent-cyan">
                        <Sparkles className="h-3 w-3" /> {item.enhance_target}
                      </div>
                    )}
                    {durations[item.name] !== undefined && (
                      <div className="absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 text-xs text-white">
                        {durations[item.name].toFixed(1)}s
                      </div>
                    )}
                  </div>
                  <div className="p-2.5">
                    <p className="text-xs text-text-secondary line-clamp-2 leading-relaxed min-h-[2rem]">
                      {item.prompt || <span className="italic text-text-muted">Made before prompts were saved</span>}
                    </p>
                    <div className="mt-1.5 flex items-center justify-between text-[11px] text-text-muted">
                      <span className="truncate">{item.model_id ? modelName(item.model_id) : ''}</span>
                      <span>{new Date(item.created_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  </div>
                </motion.button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Preview */}
      <AnimatePresence>
        {preview && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, transition: { duration: 0.2 } }}
            exit={{ opacity: 0, transition: { duration: 0.15 } }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md"
            onClick={() => setPreview(null)}
          >
            <motion.div
              initial={{ scale: 0.96, opacity: 0, y: 8 }}
              animate={{ scale: 1, opacity: 1, y: 0, transition: { type: 'spring', stiffness: 380, damping: 32 } }}
              exit={{ scale: 0.97, opacity: 0, transition: { duration: 0.12 } }}
              role="dialog"
              aria-label="Video preview"
              className="relative max-w-5xl w-full mx-4 max-h-[92vh] flex flex-col rounded-2xl bg-bg-secondary border border-border-active/60 shadow-2xl shadow-black/60 overflow-hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b border-border-dim px-4 py-2.5">
                <span className="text-sm font-medium text-text-primary">Preview</span>
                <div className="flex items-center gap-1">
                  <div className="relative">
                    <button
                      onClick={() => setEnhanceMenu((v) => !v)}
                      aria-expanded={enhanceMenu}
                      className="flex items-center gap-1.5 rounded-md bg-accent-cyan/12 px-2.5 py-1 text-xs font-medium text-accent-cyan hover:bg-accent-cyan/20"
                    >
                      <Sparkles className="h-3.5 w-3.5" /> Enhance
                    </button>
                    <AnimatePresence>
                      {enhanceMenu && (
                        <motion.div
                          initial={{ opacity: 0, y: -4, scale: 0.98 }}
                          animate={{ opacity: 1, y: 0, scale: 1, transition: { duration: 0.15 } }}
                          exit={{ opacity: 0, y: -4, transition: { duration: 0.1 } }}
                          className="absolute right-0 top-full z-10 mt-1 w-64 overflow-hidden rounded-xl border border-border-active/60 bg-bg-secondary shadow-2xl shadow-black/50"
                        >
                          {enhanceOptions.length ? enhanceOptions.map((o) => (
                            <button
                              key={o.value}
                              onClick={() => { onEnhance(preview.name, o.enhancerId, o.target, preview.prompt); setEnhanceMenu(false); }}
                              className="block w-full px-3 py-2.5 text-left text-xs text-text-primary hover:bg-bg-hover"
                            >
                              {o.label}
                            </button>
                          )) : (
                            <button
                              onClick={() => { setEnhanceMenu(false); setPreview(null); onGoToModels(); }}
                              className="block w-full px-3 py-2.5 text-left text-xs text-accent-cyan hover:bg-bg-hover"
                            >
                              Install an enhancer first…
                            </button>
                          )}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                  <IconButton label="Reuse this prompt" onClick={() => { onPromptChange(preview.prompt); if (preview.model_id && models.some((m) => m.id === preview.model_id && m.downloaded)) onSelectModel(preview.model_id); setPreview(null); }}>
                    <RotateCcw className="h-4 w-4" />
                  </IconButton>
                  <IconButton label="Copy prompt" onClick={() => { navigator.clipboard.writeText(preview.prompt).then(() => onNotify('success', 'Prompt copied.')); }}>
                    <Copy className="h-4 w-4" />
                  </IconButton>
                  <IconButton label="Show in folder" onClick={() => onRevealOutput(preview.name)}>
                    <FolderOpen className="h-4 w-4" />
                  </IconButton>
                  {confirmDelete ? (
                    <button
                      onClick={() => { onDeleteOutput(preview.name); setPreview(null); }}
                      className="rounded-md bg-accent-red/15 px-2.5 py-1 text-xs font-medium text-accent-red hover:bg-accent-red/25"
                    >
                      Delete permanently?
                    </button>
                  ) : (
                    <IconButton label="Delete video" danger onClick={() => setConfirmDelete(true)}>
                      <Trash2 className="h-4 w-4" />
                    </IconButton>
                  )}
                  <IconButton label="Close" onClick={() => setPreview(null)}>
                    <X className="h-4 w-4" />
                  </IconButton>
                </div>
              </div>
              <div className="bg-black flex-1 min-h-0 flex items-center justify-center">
                <video src={videoUrl(preview.name)} className="max-h-[70vh] w-full object-contain" controls autoPlay loop />
              </div>
              <div className="border-t border-border-dim p-4">
                <p className="text-sm text-text-primary mb-2 select-text">{preview.prompt}</p>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-text-muted">
                  {preview.model_id && <span>{modelName(preview.model_id)}</span>}
                  {preview.enhance_target && (
                    <span className="flex items-center gap-1 text-accent-cyan"><Sparkles className="h-3 w-3" /> Enhanced to {preview.enhance_target}</span>
                  )}
                  {preview.settings?.width && !preview.enhance_target && <span>{preview.settings.width}×{preview.settings.height}</span>}
                  {preview.settings?.seed != null && <span>Seed {preview.settings.seed}</span>}
                  {preview.elapsed_seconds && <span>Made in {formatDuration(preview.elapsed_seconds)}</span>}
                  <span>{(preview.bytes / 1024 ** 2).toFixed(1)} MB</span>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function RangeRow({ label, value, spec, onChange, format, hint }: {
  label: string; value: number; spec: { min: number; max: number; step: number };
  onChange: (v: number) => void; format?: (v: number) => string; hint?: string;
}) {
  return (
    <div>
      <div className="flex justify-between text-xs text-text-muted mb-1">
        <span>{label}</span>
        <span className="text-text-secondary font-mono">{format ? format(value) : value}</span>
      </div>
      <input
        type="range" aria-label={label} min={spec.min} max={spec.max} step={spec.step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-accent-purple"
      />
      {hint && <p className="text-[11px] text-text-muted mt-0.5">{hint}</p>}
    </div>
  );
}

function IconButton({ label, onClick, children, danger }: {
  label: string; onClick: () => void; children: React.ReactNode; danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      aria-label={label}
      className={cn('p-1.5 rounded-md text-text-muted transition-colors',
        danger ? 'hover:text-accent-red hover:bg-accent-red/10' : 'hover:text-text-primary hover:bg-bg-hover')}
    >
      {children}
    </button>
  );
}
