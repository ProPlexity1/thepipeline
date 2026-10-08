import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, Reorder, motion } from 'framer-motion';
import {
  Clapperboard, ChevronLeft, ChevronRight, Copy, Film, Loader2, Pause, Play, Plus, Scissors,
  Sparkles, Trash2, Volume2, VolumeX, Wand2, AlertCircle, ListPlus, X, Type, ImageIcon, Layers, Gauge, Mic,
} from 'lucide-react';
import type { AudioItem, GenerationJob, ImageItem, ModelInfo, OutputItem } from '../types';
import { audioUrl, imageUrl, videoUrl } from '../api';
import { cn } from '../utils/cn';
import JobCard from './JobCard';

// ── Timeline model ───────────────────────────────────────────────────────────

interface Clip {
  key: string;
  /** Gallery file; unset while the shot is still being generated. */
  name?: string;
  start: number;
  end: number;
  /** Playback speed: 0.25 (slow motion) to 4 (fast forward). */
  speed?: number;
  /** 0 mutes the clip's own sound; up to 2 boosts it. */
  volume?: number;
  /** Job that will produce this clip (generation, then its enhancement). */
  pendingJob?: string;
  label?: string;
  failed?: string;
}

/** A title or picture drawn over the video for part of the timeline. */
interface Layer {
  key: string;
  kind: 'text' | 'image';
  text: string;
  image?: string;
  start: number;
  end: number;
  x: number;
  y: number;
  width: number;   // images: fraction of the frame width
  size: number;    // titles: fraction of the frame height
  opacity: number;
  color: string;
  background: boolean;
}

/** A voice-over or music file placed on the timeline. */
interface AudioClip {
  key: string;
  name: string;
  start: number;
  volume: number;
}

interface ExportSettings {
  transition: 'cut' | 'fade';
  fade_seconds: number;
  fade_edges: boolean;
  resolution: 'auto' | '720p' | '1080p';
  title: string;
}

interface Shot { key: string; text: string }

const TIMELINE_KEY = 'neuralcut.timeline';
const LAYERS_KEY = 'pipeline.layers';
const AUDIO_KEY = 'pipeline.audioTrack';
const EXPORT_KEY = 'neuralcut.exportSettings';
const STORY_KEY = 'neuralcut.storyboard';
const SPEEDS = [0.25, 0.5, 0.75, 1, 1.5, 2, 4];

function load<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v ? { ...fallback, ...JSON.parse(v) } : fallback;
  } catch {
    return fallback;
  }
}
function loadList<T>(key: string): T[] {
  try {
    const v = JSON.parse(localStorage.getItem(key) || '[]');
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}
function save(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

const uid = () => Math.random().toString(36).slice(2, 10);
const fmt = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
const baseName = (p: string) => p.split(/[\\/]/).pop() || p;
const clipLen = (c: Clip) => (c.end - c.start) / (c.speed || 1);

export interface RenderPayload {
  clips: { name: string; start: number; end: number; speed: number; volume: number }[];
  overlays: Omit<Layer, 'key'>[];
  audio: { name: string; start: number; volume: number }[];
  transition: 'cut' | 'fade'; fade_seconds: number; fade_edges: boolean;
  resolution: 'auto' | '720p' | '1080p'; title: string;
}

interface EditorPanelProps {
  outputs: OutputItem[];
  images: ImageItem[];
  audio: AudioItem[];
  models: ModelInfo[];
  jobs: GenerationJob[];
  onRender: (payload: RenderPayload) => Promise<string | null>;
  onGenerate: (prompt: string, negPrompt: string, modelId: string, settings: Record<string, any>) => Promise<string | null>;
  onCancelJob: (id: string) => void;
  onDismissJob: (id: string) => void;
  onNotify: (kind: 'error' | 'success' | 'info', text: string) => void;
  onRefreshOutputs: () => void;
}

export default function EditorPanel(props: EditorPanelProps) {
  const { outputs, images, audio, models, jobs, onRender, onGenerate, onCancelJob, onDismissJob, onNotify, onRefreshOutputs } = props;

  const [clips, setClips] = useState<Clip[]>(() => loadList<Clip>(TIMELINE_KEY));
  const [layers, setLayers] = useState<Layer[]>(() => loadList<Layer>(LAYERS_KEY));
  const [audioClips, setAudioClips] = useState<AudioClip[]>(() => loadList<AudioClip>(AUDIO_KEY));
  const [selectedAudio, setSelectedAudio] = useState<string | null>(null);
  const [pickingAudio, setPickingAudio] = useState(false);
  const [settings, setSettings] = useState<ExportSettings>(() => load(EXPORT_KEY, {
    transition: 'fade', fade_seconds: 0.4, fade_edges: true, resolution: 'auto', title: '',
  } as ExportSettings));
  const [selected, setSelected] = useState<string | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [tab, setTab] = useState<'library' | 'story'>('library');
  const [exportJob, setExportJob] = useState<string | null>(null);
  const [picking, setPicking] = useState(false);

  useEffect(() => save(TIMELINE_KEY, clips), [clips]);
  useEffect(() => save(LAYERS_KEY, layers), [layers]);
  useEffect(() => save(AUDIO_KEY, audioClips), [audioClips]);
  const audioByName = useMemo(() => new Map(audio.map((a) => [a.name, a])), [audio]);
  useEffect(() => save(EXPORT_KEY, settings), [settings]);

  const byName = useMemo(() => new Map(outputs.map((o) => [o.name, o])), [outputs]);

  // Clips whose file was deleted from the gallery can't be rendered.
  const ready = clips.filter((c) => c.name && !c.pendingJob && byName.has(c.name));
  const missing = clips.filter((c) => c.name && !c.pendingJob && !byName.has(c.name));
  const pending = clips.filter((c) => c.pendingJob);
  const fadeLen = settings.transition === 'fade' && ready.length > 1
    ? Math.max(0.1, Math.min(settings.fade_seconds, ...ready.map((c) => clipLen(c) / 2))) : 0;
  const total = Math.max(0, ready.reduce((t, c) => t + clipLen(c), 0) - fadeLen * Math.max(0, ready.length - 1));
  /** Where each ready clip starts on the output timeline. */
  const offsets = useMemo(() => {
    const m = new Map<string, number>();
    let t = 0;
    for (const c of ready) { m.set(c.key, t); t += clipLen(c) - fadeLen; }
    return m;
  }, [ready, fadeLen]);

  // Storyboard shots fill in as their generation (and enhancement) finishes.
  useEffect(() => {
    if (!pending.length) return;
    let changed = false;
    const next = clips.map((c) => {
      if (!c.pendingJob) return c;
      const job = jobs.find((j) => j.id === c.pendingJob);
      if (job?.status === 'done' && job.enhanceJobId) {
        changed = true;
        return { ...c, pendingJob: job.enhanceJobId };
      }
      const fileName = job?.status === 'done' && job.outputPath ? baseName(job.outputPath) : `video_${c.pendingJob}.mp4`;
      const out = byName.get(fileName);
      if (out && (job ? job.status === 'done' : true)) {
        changed = true;
        return { ...c, name: out.name, start: 0, end: out.duration || 5, pendingJob: undefined };
      }
      if (job && (job.status === 'error' || job.status === 'cancelled')) {
        changed = true;
        return { ...c, pendingJob: undefined, failed: job.error || 'Cancelled' };
      }
      return c;
    });
    if (changed) setClips(next);
  }, [jobs, byName]);

  useEffect(() => {
    if (pending.some((c) => jobs.find((j) => j.id === c.pendingJob)?.status === 'done')) onRefreshOutputs();
  }, [jobs]);

  const addClip = (o: OutputItem) => {
    const c: Clip = { key: uid(), name: o.name, start: 0, end: o.duration || 5, speed: 1, volume: 1 };
    setClips((prev) => [...prev, c]);
    setSelected(c.key);
    setSelectedLayer(null);
  };
  const update = (key: string, patch: Partial<Clip>) =>
    setClips((prev) => prev.map((c) => (c.key === key ? { ...c, ...patch } : c)));
  const remove = (key: string) => {
    setClips((prev) => prev.filter((c) => c.key !== key));
    if (selected === key) setSelected(null);
  };
  const move = (key: string, dir: -1 | 1) => setClips((prev) => {
    const i = prev.findIndex((c) => c.key === key);
    const j = i + dir;
    if (i < 0 || j < 0 || j >= prev.length) return prev;
    const next = [...prev];
    [next[i], next[j]] = [next[j], next[i]];
    return next;
  });
  /** Cut a clip in two at a source time. */
  const split = (key: string, at: number) => setClips((prev) => {
    const i = prev.findIndex((c) => c.key === key);
    const c = prev[i];
    if (!c || at <= c.start + 0.15 || at >= c.end - 0.15) return prev;
    const b = { ...c, key: uid(), start: at };
    return [...prev.slice(0, i), { ...c, end: at }, b, ...prev.slice(i + 1)];
  });

  const addLayer = (kind: Layer['kind'], image?: string) => {
    const l: Layer = {
      key: uid(), kind, text: kind === 'text' ? 'Your title' : '', image,
      start: 0, end: Math.min(Math.max(total, 1), 3), x: kind === 'text' ? 0.5 : 0.86, y: kind === 'text' ? 0.82 : 0.14,
      width: 0.18, size: 0.07, opacity: 1, color: '#ffffff', background: false,
    };
    setLayers((prev) => [...prev, l]);
    setSelectedLayer(l.key);
    setSelected(null);
    setPicking(false);
  };
  const updateLayer = (key: string, patch: Partial<Layer>) =>
    setLayers((prev) => prev.map((l) => (l.key === key ? { ...l, ...patch } : l)));

  const exportVideo = async () => {
    if (!ready.length) return;
    const id = await onRender({
      clips: ready.map((c) => ({ name: c.name!, start: c.start, end: c.end, speed: c.speed || 1, volume: c.volume ?? 1 })),
      overlays: layers
        .filter((l) => (l.kind === 'text' ? l.text.trim() : l.image && images.some((im) => im.name === l.image)))
        .map(({ key: _k, ...l }) => ({ ...l, start: Math.min(l.start, total), end: Math.min(l.end, total) })),
      audio: audioClips.filter((a) => audioByName.has(a.name)).map(({ name, start, volume }) => ({ name, start, volume })),
      ...settings,
    });
    if (id) setExportJob(id);
  };

  const editJobs = jobs.filter((j) => j.kind === 'edit');
  const exportRunning = editJobs.some((j) => ['queued', 'post_processing'].includes(j.status));
  const finishedExport = exportJob ? jobs.find((j) => j.id === exportJob && j.status === 'done') : undefined;

  const selectedClip = clips.find((c) => c.key === selected) || null;
  const layer = layers.find((l) => l.key === selectedLayer) || null;
  const aclip = audioClips.find((a) => a.key === selectedAudio) || null;

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Left: library / storyboard */}
      <aside className="flex w-[300px] flex-shrink-0 flex-col border-r border-border-dim bg-bg-secondary">
        <div className="flex gap-1 border-b border-border-dim p-2">
          {([['library', 'Clips', <Film key="f" className="h-3.5 w-3.5" />], ['story', 'Storyboard', <Clapperboard key="c" className="h-3.5 w-3.5" />]] as const).map(([id, label, icon]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={cn('relative flex flex-1 items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium',
                tab === id ? 'text-text-primary' : 'text-text-muted hover:text-text-secondary')}
            >
              {tab === id && <motion.span layoutId="editor-tab" className="absolute inset-0 rounded-md bg-bg-hover" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
              <span className="relative flex items-center gap-1.5">{icon}{label}</span>
            </button>
          ))}
        </div>
        <div className="flex-1 overflow-y-auto">
          {tab === 'library'
            ? <Library outputs={outputs} onAdd={addClip} />
            : <Storyboard models={models} onGenerate={onGenerate} onNotify={onNotify}
                onQueued={(items) => { setClips((prev) => [...prev, ...items]); }} />}
        </div>
      </aside>

      {/* Centre: preview, trim, timeline */}
      <section className="flex min-w-0 flex-1 flex-col">
        <Preview clips={ready} selected={selectedClip} offsets={offsets} layers={layers} audioClips={audioClips.filter((a) => audioByName.has(a.name))}
          onSetTrim={(k, p) => update(k, p)} onSplit={split} finished={finishedExport} />

        {/* Timeline */}
        <div className="border-t border-border-dim bg-bg-secondary px-4 pb-3 pt-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-text-muted">
              <Scissors className="h-3.5 w-3.5" /> Timeline
              <span className="normal-case tracking-normal text-text-secondary">
                {ready.length} clip{ready.length === 1 ? '' : 's'} · {fmt(total)}
                {pending.length > 0 && ` · ${pending.length} still generating`}
              </span>
            </div>
            {clips.length > 0 && (
              <button onClick={() => { setClips([]); setSelected(null); }} className="text-xs text-text-muted hover:text-accent-red">
                Clear timeline
              </button>
            )}
          </div>
          {clips.length === 0 ? (
            <div className="flex h-[96px] items-center justify-center rounded-lg border border-dashed border-border-active text-sm text-text-muted">
              Add clips from the library, or write a storyboard to generate a sequence of shots.
            </div>
          ) : (
            <Reorder.Group axis="x" values={clips} onReorder={setClips} className="flex gap-2 overflow-x-auto pb-1">
              {clips.map((c, i) => (
                <TimelineItem
                  key={c.key} clip={c} index={i} job={jobs.find((j) => j.id === c.pendingJob)}
                  missing={missing.includes(c)} selected={selected === c.key}
                  onSelect={() => { setSelected(c.key); setSelectedLayer(null); }} onRemove={() => remove(c.key)}
                  onMove={(d) => move(c.key, d)}
                  onDuplicate={() => setClips((prev) => {
                    const at = prev.findIndex((x) => x.key === c.key);
                    return [...prev.slice(0, at + 1), { ...c, key: uid() }, ...prev.slice(at + 1)];
                  })}
                />
              ))}
            </Reorder.Group>
          )}

          {/* Layers track */}
          <div className="mt-3 flex items-center gap-2">
            <span className="flex w-16 flex-shrink-0 items-center gap-1 text-[11px] font-medium uppercase tracking-wider text-text-muted">
              <Layers className="h-3.5 w-3.5" /> Layers
            </span>
            <div className="relative h-9 flex-1 overflow-hidden rounded-md border border-border-dim bg-bg-primary">
              {layers.map((l, i) => {
                const span = Math.max(total, 0.1);
                return (
                  <button key={l.key} onClick={() => { setSelectedLayer(l.key); setSelected(null); }}
                    title={l.kind === 'text' ? l.text : l.image}
                    className={cn('absolute flex items-center gap-1 overflow-hidden rounded px-1.5 text-[10px] font-medium',
                      l.kind === 'text' ? 'bg-accent-amber/25 text-accent-amber' : 'bg-accent-cyan/25 text-accent-cyan',
                      selectedLayer === l.key && 'ring-1 ring-white/70')}
                    style={{
                      left: `${(Math.min(l.start, span) / span) * 100}%`,
                      width: `${Math.max(3, ((Math.min(l.end, span) - Math.min(l.start, span)) / span) * 100)}%`,
                      top: `${2 + (i % 2) * 16}px`, height: 14,
                    }}>
                    {l.kind === 'text' ? <Type className="h-2.5 w-2.5 flex-shrink-0" /> : <ImageIcon className="h-2.5 w-2.5 flex-shrink-0" />}
                    <span className="truncate">{l.kind === 'text' ? l.text : 'Image'}</span>
                  </button>
                );
              })}
            </div>
            <button onClick={() => addLayer('text')} title="Add a title"
              className="flex items-center gap-1 rounded-md border border-border-dim px-2 py-1.5 text-xs text-text-secondary hover:border-border-active hover:text-text-primary">
              <Type className="h-3.5 w-3.5" /> Title
            </button>
            <div className="relative">
              <button onClick={() => setPicking((p) => !p)} title="Add a logo or image"
                className="flex items-center gap-1 rounded-md border border-border-dim px-2 py-1.5 text-xs text-text-secondary hover:border-border-active hover:text-text-primary">
                <ImageIcon className="h-3.5 w-3.5" /> Logo
              </button>
              <AnimatePresence>
                {picking && (
                  <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="absolute bottom-full right-0 z-20 mb-2 grid max-h-72 w-72 grid-cols-3 gap-1.5 overflow-y-auto rounded-lg border border-border-dim bg-bg-secondary p-2 shadow-2xl">
                    {images.length === 0 && <p className="col-span-3 p-2 text-xs text-text-muted">Add images on the Images page (upload your logo there).</p>}
                    {images.map((im) => (
                      <button key={im.name} onClick={() => addLayer('image', im.name)}
                        className="aspect-square overflow-hidden rounded border border-border-dim bg-black hover:border-accent-purple">
                        <img src={imageUrl(im.name)} alt="" className="h-full w-full object-contain" />
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Audio track: voice-overs and music */}
          <div className="mt-2 flex items-center gap-2">
            <span className="flex w-16 flex-shrink-0 items-center gap-1 text-[11px] font-medium uppercase tracking-wider text-text-muted">
              <Mic className="h-3.5 w-3.5" /> Audio
            </span>
            <div className="relative h-9 flex-1 overflow-hidden rounded-md border border-border-dim bg-bg-primary">
              {audioClips.filter((a) => audioByName.has(a.name)).map((a) => {
                const span = Math.max(total, 0.1);
                const len = audioByName.get(a.name)?.duration || 3;
                return (
                  <button key={a.key} onClick={() => { setSelectedAudio(a.key); setSelectedLayer(null); setSelected(null); }}
                    title={audioByName.get(a.name)?.title}
                    className={cn('absolute top-1.5 flex h-6 items-center gap-1 overflow-hidden rounded bg-accent-green/25 px-1.5 text-[10px] font-medium text-accent-green',
                      selectedAudio === a.key && 'ring-1 ring-white/70')}
                    style={{ left: (Math.min(a.start, span) / span) * 100 + '%', width: Math.max(4, (Math.min(len, span - a.start) / span) * 100) + '%' }}>
                    <Mic className="h-2.5 w-2.5 flex-shrink-0" /><span className="truncate">{audioByName.get(a.name)?.title}</span>
                  </button>
                );
              })}
            </div>
            <div className="relative">
              <button onClick={() => setPickingAudio((p) => !p)} title="Add a voice-over"
                className="flex items-center gap-1 rounded-md border border-border-dim px-2 py-1.5 text-xs text-text-secondary hover:border-border-active hover:text-text-primary">
                <Mic className="h-3.5 w-3.5" /> Voice
              </button>
              <AnimatePresence>
                {pickingAudio && (
                  <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    className="absolute bottom-full right-0 z-20 mb-2 max-h-72 w-80 space-y-1 overflow-y-auto rounded-lg border border-border-dim bg-bg-secondary p-2 shadow-2xl">
                    {audio.length === 0 && <p className="p-2 text-xs text-text-muted">Make voice-overs on the Voices page first.</p>}
                    {audio.map((a) => (
                      <button key={a.name} onClick={() => {
                        const c = { key: uid(), name: a.name, start: 0.5, volume: 1 };
                        setAudioClips((p) => [...p, c]); setSelectedAudio(c.key); setPickingAudio(false);
                      }} className="w-full rounded px-2 py-1.5 text-left text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary">
                        <span className="block truncate">{a.title}</span>
                        <span className="text-[10px] text-text-muted">{a.duration ? a.duration.toFixed(1) + 's' : ''}</span>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>

      {/* Right: layer properties + export */}
      <aside className="flex w-[280px] flex-shrink-0 flex-col gap-4 overflow-y-auto border-l border-border-dim bg-bg-secondary p-4">
        <AnimatePresence initial={false}>
          {aclip && (
            <motion.div key={aclip.key} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="space-y-3 rounded-lg border border-accent-green/30 bg-accent-green/5 p-3">
              <div className="flex items-center justify-between">
                <h3 className="flex items-center gap-1.5 text-sm font-semibold text-text-primary"><Mic className="h-4 w-4 text-accent-green" /> Voice-over</h3>
                <div className="flex gap-1">
                  <button onClick={() => { setAudioClips((p) => p.filter((a) => a.key !== aclip.key)); setSelectedAudio(null); }}
                    aria-label="Remove voice-over" className="rounded p-1 text-text-muted hover:text-accent-red"><Trash2 className="h-3.5 w-3.5" /></button>
                  <button onClick={() => setSelectedAudio(null)} aria-label="Close" className="rounded p-1 text-text-muted hover:text-text-primary"><X className="h-3.5 w-3.5" /></button>
                </div>
              </div>
              <p className="line-clamp-2 text-xs text-text-secondary">{audioByName.get(aclip.name)?.title}</p>
              <RangeRow label="Starts at" value={aclip.start} min={0} max={Math.max(total, 0.1)} step={0.1} fmt={(v) => v.toFixed(1) + 's'}
                onChange={(v) => setAudioClips((p) => p.map((a) => (a.key === aclip.key ? { ...a, start: v } : a)))} />
              <RangeRow label="Volume" value={aclip.volume} min={0} max={2} step={0.05} fmt={(v) => Math.round(v * 100) + '%'}
                onChange={(v) => setAudioClips((p) => p.map((a) => (a.key === aclip.key ? { ...a, volume: v } : a)))} />
            </motion.div>
          )}
        </AnimatePresence>

        <AnimatePresence initial={false}>
          {layer && (
            <motion.div key={layer.key} initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="space-y-3 rounded-lg border border-border-dim bg-bg-tertiary/40 p-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-text-primary">{layer.kind === 'text' ? 'Title' : 'Logo / image'}</h3>
                <div className="flex gap-1">
                  <button onClick={() => { setLayers((p) => p.filter((l) => l.key !== layer.key)); setSelectedLayer(null); }}
                    aria-label="Delete layer" className="rounded p-1 text-text-muted hover:text-accent-red"><Trash2 className="h-3.5 w-3.5" /></button>
                  <button onClick={() => setSelectedLayer(null)} aria-label="Close" className="rounded p-1 text-text-muted hover:text-text-primary"><X className="h-3.5 w-3.5" /></button>
                </div>
              </div>
              {layer.kind === 'text' ? (
                <textarea value={layer.text} rows={2} maxLength={300} onChange={(e) => updateLayer(layer.key, { text: e.target.value })}
                  className="w-full resize-none rounded-md border border-border-dim bg-bg-secondary px-2.5 py-1.5 text-sm text-text-primary outline-none focus:border-accent-purple/60" />
              ) : layer.image && (
                <img src={imageUrl(layer.image)} alt="" className="h-16 w-full rounded border border-border-dim bg-black object-contain" />
              )}
              <RangeRow label="Starts" value={layer.start} min={0} max={Math.max(total, 0.1)} step={0.1} fmt={(v) => `${v.toFixed(1)}s`}
                onChange={(v) => updateLayer(layer.key, { start: Math.min(v, layer.end - 0.2) })} />
              <RangeRow label="Ends" value={Math.min(layer.end, Math.max(total, 0.1))} min={0} max={Math.max(total, 0.1)} step={0.1} fmt={(v) => `${v.toFixed(1)}s`}
                onChange={(v) => updateLayer(layer.key, { end: Math.max(v, layer.start + 0.2) })} />
              <RangeRow label="Left ↔ right" value={layer.x} min={0} max={1} step={0.01} fmt={(v) => `${Math.round(v * 100)}%`}
                onChange={(v) => updateLayer(layer.key, { x: v })} />
              <RangeRow label="Top ↕ bottom" value={layer.y} min={0} max={1} step={0.01} fmt={(v) => `${Math.round(v * 100)}%`}
                onChange={(v) => updateLayer(layer.key, { y: v })} />
              {layer.kind === 'text' ? (
                <RangeRow label="Size" value={layer.size} min={0.02} max={0.3} step={0.005} fmt={(v) => `${Math.round(v * 100)}`}
                  onChange={(v) => updateLayer(layer.key, { size: v })} />
              ) : (
                <RangeRow label="Size" value={layer.width} min={0.03} max={1} step={0.01} fmt={(v) => `${Math.round(v * 100)}%`}
                  onChange={(v) => updateLayer(layer.key, { width: v })} />
              )}
              <RangeRow label="Opacity" value={layer.opacity} min={0} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`}
                onChange={(v) => updateLayer(layer.key, { opacity: v })} />
              {layer.kind === 'text' && (
                <div className="flex items-center justify-between gap-2 text-xs text-text-secondary">
                  <label className="flex items-center gap-2">Colour
                    <input type="color" value={layer.color} onChange={(e) => updateLayer(layer.key, { color: e.target.value })}
                      className="h-6 w-8 cursor-pointer rounded border border-border-dim bg-transparent" />
                  </label>
                  <label className="flex cursor-pointer items-center gap-1.5">
                    <input type="checkbox" checked={layer.background} onChange={(e) => updateLayer(layer.key, { background: e.target.checked })}
                      className="accent-[var(--color-accent-purple)]" /> Dark box
                  </label>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <div>
          <h3 className="text-sm font-semibold text-text-primary">Export</h3>
          <p className="mt-1 text-xs leading-relaxed text-text-muted">
            Joins the timeline into one video with sound. Runs on the processor, so you can keep generating meanwhile.
          </p>
        </div>

        <Field label="Title">
          <input
            value={settings.title} maxLength={200} placeholder="My video"
            onChange={(e) => setSettings((s) => ({ ...s, title: e.target.value }))}
            className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2.5 py-1.5 text-sm text-text-primary outline-none focus:border-accent-purple/60"
          />
        </Field>

        <Field label="Between clips">
          <Segmented
            value={settings.transition}
            options={[['cut', 'Hard cut'], ['fade', 'Crossfade']]}
            onChange={(v) => setSettings((s) => ({ ...s, transition: v as ExportSettings['transition'] }))}
          />
          <AnimatePresence initial={false}>
            {settings.transition === 'fade' && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                <div className="mt-2 flex items-center gap-2 text-xs text-text-secondary">
                  <input type="range" min={0.2} max={1.5} step={0.1} value={settings.fade_seconds}
                    onChange={(e) => setSettings((s) => ({ ...s, fade_seconds: Number(e.target.value) }))}
                    className="flex-1 accent-[var(--color-accent-purple)]" aria-label="Crossfade length" />
                  <span className="w-10 text-right tabular-nums">{settings.fade_seconds.toFixed(1)}s</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </Field>

        <label className="flex cursor-pointer items-center justify-between text-sm text-text-secondary">
          Fade in from black and out at the end
          <input type="checkbox" checked={settings.fade_edges}
            onChange={(e) => setSettings((s) => ({ ...s, fade_edges: e.target.checked }))}
            className="h-4 w-4 accent-[var(--color-accent-purple)]" />
        </label>

        <Field label="Size">
          <Segmented
            value={settings.resolution}
            options={[['auto', 'Largest clip'], ['1080p', '1080p'], ['720p', '720p']]}
            onChange={(v) => setSettings((s) => ({ ...s, resolution: v as ExportSettings['resolution'] }))}
          />
        </Field>

        {missing.length > 0 && (
          <p className="flex gap-1.5 rounded-md bg-accent-amber/10 px-2.5 py-2 text-xs text-accent-amber">
            <AlertCircle className="mt-px h-3.5 w-3.5 flex-shrink-0" />
            {missing.length} clip{missing.length === 1 ? ' was' : 's were'} deleted from the gallery and will be skipped.
          </p>
        )}

        <motion.button
          whileTap={{ scale: 0.98 }}
          disabled={!ready.length || exportRunning}
          onClick={exportVideo}
          className="flex items-center justify-center gap-2 rounded-lg bg-accent-purple px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-accent-purple/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
        >
          {exportRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          {exportRunning ? 'Exporting…' : `Export ${fmt(total)} video`}
        </motion.button>
        {pending.length > 0 && ready.length > 0 && (
          <p className="-mt-2 text-xs text-text-muted">Exporting now leaves out the {pending.length} shot{pending.length === 1 ? '' : 's'} still generating.</p>
        )}

        <div className="space-y-2">
          <AnimatePresence initial={false}>
            {editJobs.map((j) => (
              <JobCard key={j.id} job={j} onCancel={() => onCancelJob(j.id)} onDismiss={() => onDismissJob(j.id)} />
            ))}
          </AnimatePresence>
        </div>
      </aside>
    </div>
  );
}

// ── Library ──────────────────────────────────────────────────────────────────

function Library({ outputs, onAdd }: { outputs: OutputItem[]; onAdd: (o: OutputItem) => void }) {
  if (!outputs.length) {
    return <p className="p-4 text-sm text-text-muted">Videos you generate appear here. Add them to the timeline to edit them together.</p>;
  }
  return (
    <ul className="space-y-1 p-2">
      {outputs.map((o) => (
        <li key={o.name}>
          <button
            onClick={() => onAdd(o)}
            className="group flex w-full items-center gap-2.5 rounded-lg p-1.5 text-left transition hover:bg-bg-hover"
            title="Add to timeline"
          >
            <div className="relative h-[46px] w-[82px] flex-shrink-0 overflow-hidden rounded-md bg-black">
              <video src={`${videoUrl(o.name)}#t=0.5`} preload="metadata" muted className="h-full w-full object-cover" />
              <span className="absolute bottom-0.5 right-0.5 rounded bg-black/70 px-1 text-[10px] tabular-nums text-white">
                {o.duration ? `${o.duration.toFixed(1)}s` : ''}
              </span>
              <span className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 transition group-hover:opacity-100">
                <Plus className="h-5 w-5 text-white" />
              </span>
            </div>
            <div className="min-w-0 flex-1">
              <p className="line-clamp-2 text-xs leading-snug text-text-secondary">{o.prompt || o.name}</p>
              <div className="mt-1 flex items-center gap-1.5 text-[10px] text-text-muted">
                {o.width ? <span>{o.width}×{o.height}</span> : null}
                {o.has_audio ? <Volume2 className="h-3 w-3" aria-label="Has sound" /> : <VolumeX className="h-3 w-3 opacity-50" aria-label="No sound" />}
                {o.enhanced_from && <span className="rounded bg-accent-cyan/15 px-1 text-accent-cyan">HD</span>}
                {o.edit && <span className="rounded bg-accent-purple/15 px-1 text-accent-purple">Edit</span>}
              </div>
            </div>
          </button>
        </li>
      ))}
    </ul>
  );
}

// ── Storyboard ───────────────────────────────────────────────────────────────

function Storyboard({ models, onGenerate, onNotify, onQueued }: {
  models: ModelInfo[];
  onGenerate: EditorPanelProps['onGenerate'];
  onNotify: EditorPanelProps['onNotify'];
  onQueued: (clips: Clip[]) => void;
}) {
  const saved = load(STORY_KEY, { style: '', shots: [] as Shot[], model: '', preset: 'balanced', enhancer: '' });
  const [style, setStyle] = useState(saved.style);
  const [shots, setShots] = useState<Shot[]>(saved.shots.length ? saved.shots : [{ key: uid(), text: '' }]);
  const videoModels = models.filter((m) => m.kind === 'video' && m.downloaded && m.capabilities.text_to_video !== false);
  const enhancers = models.filter((m) => m.kind === 'enhancer' && m.downloaded);
  const [modelId, setModelId] = useState(saved.model);
  const [preset, setPreset] = useState(saved.preset);
  const [enhancer, setEnhancer] = useState(saved.enhancer);
  const [busy, setBusy] = useState(false);

  const model = videoModels.find((m) => m.id === modelId) || videoModels[0];
  const profiles: Record<string, any> = model?.profiles || {};
  const presetKey = profiles[preset] ? preset : Object.keys(profiles)[0];

  useEffect(() => save(STORY_KEY, { style, shots, model: model?.id || '', preset: presetKey, enhancer }),
    [style, shots, model?.id, presetKey, enhancer]);

  const filled = shots.filter((s) => s.text.trim());

  const generateAll = async () => {
    if (!model || !filled.length) return;
    setBusy(true);
    const p = profiles[presetKey] || {};
    const target = enhancers.find((e) => e.id === enhancer)?.enhance_targets?.[0]?.id;
    const queued: Clip[] = [];
    for (const [i, shot] of filled.entries()) {
      const prompt = style.trim() ? `${shot.text.trim()} ${style.trim()}` : shot.text.trim();
      const id = await onGenerate(prompt, '', model.id, {
        profile: presetKey, steps: p.steps, cfg_scale: p.cfg_scale, width: p.width, height: p.height,
        num_frames: p.num_frames, fps: p.fps,
        ...(enhancer && target ? { enhancer_id: enhancer, enhance_target: target } : {}),
      });
      if (!id) break;
      queued.push({ key: uid(), start: 0, end: 0, pendingJob: id, label: `Shot ${i + 1}`, speed: 1, volume: 1 });
    }
    setBusy(false);
    if (queued.length) {
      onQueued(queued);
      onNotify('info', `${queued.length} shot${queued.length === 1 ? '' : 's'} queued. They join the timeline as each one finishes.`);
    }
  };

  if (!videoModels.length) {
    return <p className="p-4 text-sm text-text-muted">Install a video model from the Models page to generate shots.</p>;
  }

  return (
    <div className="space-y-4 p-3">
      <p className="text-xs leading-relaxed text-text-muted">
        Describe each shot of your story. They're generated one after another and placed on the timeline in order.
        For the most realistic, consistent results, ask the Agent to produce it instead (it makes a keyframe image for every shot first).
      </p>
      <Field label="Characters & look (added to every shot)">
        <textarea
          value={style} onChange={(e) => setStyle(e.target.value)} rows={3} maxLength={1500}
          placeholder="e.g. The hero is a man in a black suit with long dark hair. Night, rain, neon light, cinematic."
          className="w-full resize-none rounded-md border border-border-dim bg-bg-tertiary px-2.5 py-2 text-xs leading-relaxed text-text-primary outline-none focus:border-accent-purple/60"
        />
      </Field>
      <div className="space-y-2">
        <AnimatePresence initial={false}>
          {shots.map((s, i) => (
            <motion.div key={s.key} layout initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, height: 0 }}
              className="rounded-lg border border-border-dim bg-bg-tertiary/60 p-2">
              <div className="mb-1 flex items-center justify-between text-[11px] font-medium text-text-muted">
                Shot {i + 1}
                {shots.length > 1 && (
                  <button onClick={() => setShots((prev) => prev.filter((x) => x.key !== s.key))} aria-label={`Remove shot ${i + 1}`}
                    className="text-text-muted hover:text-accent-red"><X className="h-3.5 w-3.5" /></button>
                )}
              </div>
              <textarea
                value={s.text} rows={3} maxLength={2000}
                onChange={(e) => setShots((prev) => prev.map((x) => (x.key === s.key ? { ...x, text: e.target.value } : x)))}
                placeholder="What happens in this shot, and what it sounds like"
                className="w-full resize-none bg-transparent text-xs leading-relaxed text-text-primary outline-none"
              />
            </motion.div>
          ))}
        </AnimatePresence>
        <button onClick={() => setShots((prev) => [...prev, { key: uid(), text: '' }])}
          className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-border-active py-2 text-xs text-text-secondary hover:border-accent-purple/50 hover:text-text-primary">
          <ListPlus className="h-3.5 w-3.5" /> Add shot
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Field label="Model">
          <select value={model?.id} onChange={(e) => setModelId(e.target.value)}
            className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary outline-none">
            {videoModels.map((m) => <option key={m.id} value={m.id}>{m.display_name}</option>)}
          </select>
        </Field>
        <Field label="Quality">
          <select value={presetKey} onChange={(e) => setPreset(e.target.value)}
            className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary outline-none">
            {Object.entries(profiles).map(([k, p]: [string, any]) => <option key={k} value={k}>{p.label || k}</option>)}
          </select>
        </Field>
      </div>
      <Field label="Enhance each shot">
        <select value={enhancer} onChange={(e) => setEnhancer(e.target.value)}
          className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary outline-none">
          <option value="">Off</option>
          {enhancers.map((m) => <option key={m.id} value={m.id}>{m.display_name} ({m.enhance_targets?.[0]?.label || '1080p'})</option>)}
        </select>
      </Field>

      <motion.button
        whileTap={{ scale: 0.98 }} disabled={busy || !filled.length} onClick={generateAll}
        className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent-purple px-3 py-2 text-sm font-semibold text-white disabled:opacity-40"
      >
        {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
        Generate {filled.length || ''} shot{filled.length === 1 ? '' : 's'}
      </motion.button>
      {model?.speed_label && <p className="text-[11px] text-text-muted">{model.display_name}: {model.speed_label.toLowerCase()} per shot.</p>}
    </div>
  );
}

// ── Timeline item ────────────────────────────────────────────────────────────

function TimelineItem({ clip, index, job, missing, selected, onSelect, onRemove, onMove, onDuplicate }: {
  clip: Clip; index: number; job?: GenerationJob; missing: boolean; selected: boolean;
  onSelect: () => void; onRemove: () => void; onMove: (d: -1 | 1) => void; onDuplicate: () => void;
}) {
  const dur = clipLen(clip);
  const width = clip.name && !clip.pendingJob ? Math.max(112, Math.min(260, dur * 34)) : 150;
  const speed = clip.speed || 1;
  return (
    <Reorder.Item
      value={clip} id={clip.key}
      initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
      whileDrag={{ scale: 1.04, zIndex: 10 }}
      style={{ width }}
      onClick={onSelect}
      className={cn('group relative h-[96px] flex-shrink-0 cursor-grab overflow-hidden rounded-lg border bg-black active:cursor-grabbing',
        selected ? 'border-accent-purple ring-2 ring-accent-purple/30' : 'border-border-dim hover:border-border-active',
        missing && 'opacity-50')}
    >
      {clip.name && !clip.pendingJob ? (
        <video src={`${videoUrl(clip.name)}#t=${(clip.start + 0.3).toFixed(1)}`} preload="metadata" muted
          className="pointer-events-none h-full w-full object-cover" />
      ) : clip.failed ? (
        <div className="flex h-full flex-col items-center justify-center gap-1 bg-accent-red/10 px-2 text-center text-[11px] text-accent-red">
          <AlertCircle className="h-4 w-4" /> Failed
          <span className="line-clamp-2 text-text-muted">{clip.failed}</span>
        </div>
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-1.5 bg-bg-tertiary px-2 text-center">
          <Loader2 className="h-4 w-4 animate-spin text-accent-purple" />
          <span className="text-[11px] text-text-secondary">{job?.kind === 'enhance' ? 'Enhancing' : job?.status === 'queued' ? 'Waiting' : 'Generating'}</span>
          {job && job.progress > 0 && (
            <div className="h-1 w-full overflow-hidden rounded-full bg-black/40">
              <div className="h-full bg-accent-purple transition-all" style={{ width: `${job.progress}%` }} />
            </div>
          )}
        </div>
      )}
      <span className="absolute left-1 top-1 rounded bg-black/70 px-1.5 text-[10px] font-medium text-white">
        {clip.label || index + 1}
      </span>
      {clip.name && !clip.pendingJob && (
        <span className="absolute bottom-1 left-1 flex items-center gap-1 rounded bg-black/70 px-1.5 text-[10px] tabular-nums text-white">
          {dur.toFixed(1)}s
          {speed !== 1 && <span className="text-accent-amber">{speed}×</span>}
          {clip.volume === 0 && <VolumeX className="h-2.5 w-2.5" />}
        </span>
      )}
      <div className="absolute right-1 top-1 flex gap-0.5 opacity-0 transition group-hover:opacity-100">
        {[
          { icon: <ChevronLeft className="h-3 w-3" />, label: 'Move left', fn: () => onMove(-1) },
          { icon: <ChevronRight className="h-3 w-3" />, label: 'Move right', fn: () => onMove(1) },
          ...(clip.name ? [{ icon: <Copy className="h-3 w-3" />, label: 'Duplicate', fn: onDuplicate }] : []),
          { icon: <Trash2 className="h-3 w-3" />, label: 'Remove', fn: onRemove },
        ].map((b) => (
          <button key={b.label} aria-label={b.label} title={b.label}
            onClick={(e) => { e.stopPropagation(); b.fn(); }}
            onPointerDown={(e) => e.stopPropagation()}
            className="rounded bg-black/75 p-1 text-white hover:bg-accent-purple">{b.icon}</button>
        ))}
      </div>
    </Reorder.Item>
  );
}

// ── Preview & trim ───────────────────────────────────────────────────────────

function Preview({ clips, selected, offsets, layers, audioClips, onSetTrim, onSplit, finished }: {
  clips: Clip[]; selected: Clip | null; offsets: Map<string, number>; layers: Layer[]; audioClips: AudioClip[];
  onSetTrim: (key: string, p: Partial<Clip>) => void; onSplit: (key: string, at: number) => void;
  finished?: GenerationJob;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  // 'clip' previews the selected clip's trimmed range; 'all' plays the whole timeline.
  const [mode, setMode] = useState<'clip' | 'all' | 'result'>('clip');
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [rect, setRect] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const sel = selected && selected.name && !selected.pendingJob ? selected : null;
  const current: Clip | null = mode === 'all' ? clips[idx] || null : sel;
  const resultName = finished?.outputPath ? baseName(finished.outputPath) : null;
  const src = mode === 'result' && resultName ? resultName : current?.name;

  useEffect(() => { if (finished && resultName) setMode('result'); }, [resultName]);
  useEffect(() => { if (selected) setMode((m) => (m === 'all' && playing ? m : 'clip')); }, [selected?.key]);

  /** Where the picture actually sits inside the player (it is letterboxed). */
  const measure = useCallback(() => {
    const v = ref.current, box = boxRef.current;
    if (!v || !box || !v.videoWidth) { setRect(null); return; }
    const bw = box.clientWidth, bh = box.clientHeight;
    const scale = Math.min(bw / v.videoWidth, bh / v.videoHeight);
    const w = v.videoWidth * scale, h = v.videoHeight * scale;
    setRect({ x: (bw - w) / 2, y: (bh - h) / 2, w, h });
  }, []);
  useEffect(() => {
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [measure]);

  const onLoaded = () => {
    const v = ref.current;
    if (!v) return;
    measure();
    if (mode !== 'result' && current) {
      v.currentTime = current.start;
      v.playbackRate = current.speed || 1;
    }
    if (mode === 'all' && playing) v.play().catch(() => {});
  };
  const onTime = () => {
    const v = ref.current;
    if (!v) return;
    setTime(v.currentTime);
    if (mode === 'result' || !current) return;
    if (v.currentTime >= current.end - 0.02) {
      if (mode === 'all' && idx < clips.length - 1) {
        setIdx((i) => i + 1);
      } else {
        v.pause();
        setPlaying(false);
        if (mode === 'all') setIdx(0);
      }
    }
  };
  const toggle = () => {
    const v = ref.current;
    if (!v) return;
    if (v.paused) {
      if (current && mode !== 'result' && (v.currentTime >= current.end - 0.05 || v.currentTime < current.start)) v.currentTime = current.start;
      if (current) v.playbackRate = current.speed || 1;
      v.play().catch(() => {});
    } else v.pause();
  };
  const playAll = () => {
    if (!clips.length) return;
    setMode('all');
    setIdx(0);
    setPlaying(true);
    const v = ref.current;
    if (v && clips[0]?.name === src) { v.currentTime = clips[0].start; v.playbackRate = clips[0].speed || 1; v.play().catch(() => {}); }
  };

  // Output-timeline time, so layers appear exactly when they will in the export.
  const tl = current && mode !== 'result' && offsets.has(current.key)
    ? offsets.get(current.key)! + Math.max(0, time - current.start) / (current.speed || 1) : null;
  const visibleLayers = tl === null ? [] : layers.filter((l) => tl >= l.start && tl <= l.end);

  // Keep voice-overs in step with the picture while previewing.
  const audioRefs = useRef(new Map<string, HTMLAudioElement>());
  useEffect(() => {
    for (const a of audioClips) {
      const el = audioRefs.current.get(a.key);
      if (!el) continue;
      el.volume = Math.min(1, a.volume);
      const local = tl === null ? -1 : tl - a.start;
      const inside = playing && local >= 0 && (!el.duration || local < el.duration);
      if (inside) {
        if (Math.abs(el.currentTime - local) > 0.25) el.currentTime = local;
        if (el.paused) el.play().catch(() => {});
      } else if (!el.paused) el.pause();
    }
  }, [tl, playing, audioClips]);
  const srcDur = sel ? Math.max(sel.end, ref.current?.duration && isFinite(ref.current.duration) ? ref.current.duration : 0) : 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
      <div ref={boxRef} className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-xl bg-black">
        {src ? (
          <video
            key={src} ref={ref} src={videoUrl(src)} className="absolute inset-0 h-full w-full object-contain"
            onLoadedMetadata={onLoaded} onTimeUpdate={onTime}
            onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
            controls={mode === 'result'} playsInline
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-sm text-text-muted">
            <Film className="h-8 w-8 opacity-40" />
            Select a clip on the timeline to preview and trim it.
          </div>
        )}
        {rect && visibleLayers.length > 0 && (
          <div className="pointer-events-none absolute" style={{ left: rect.x, top: rect.y, width: rect.w, height: rect.h }}>
            {visibleLayers.map((l) => (
              <div key={l.key} className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${l.x * 100}%`, top: `${l.y * 100}%`, opacity: l.opacity }}>
                {l.kind === 'text' ? (
                  <div className={cn('whitespace-pre text-center font-bold leading-tight', l.background ? 'rounded-md bg-black/60 px-[0.45em] py-[0.3em]' : '')}
                    style={{ fontSize: rect.h * l.size, color: l.color, textShadow: l.background ? undefined : '2px 2px 3px rgba(0,0,0,0.6)' }}>
                    {l.text}
                  </div>
                ) : l.image && (
                  <img src={imageUrl(l.image)} alt="" style={{ width: rect.w * l.width }} />
                )}
              </div>
            ))}
          </div>
        )}
        {audioClips.map((a) => (
          <audio key={a.key} ref={(el) => { if (el) audioRefs.current.set(a.key, el); else audioRefs.current.delete(a.key); }}
            src={audioUrl(a.name)} preload="auto" />
        ))}
        {mode === 'result' && (
          <span className="absolute left-3 top-3 rounded-md bg-accent-green/90 px-2 py-0.5 text-xs font-medium text-black">Exported video</span>
        )}
        {mode === 'all' && current && (
          <span className="absolute left-3 top-3 rounded-md bg-black/70 px-2 py-0.5 text-xs text-white">
            Clip {idx + 1} of {clips.length} · crossfades show as cuts in preview
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button onClick={toggle} disabled={!src || mode === 'result'}
          className="flex items-center gap-1.5 rounded-md bg-bg-tertiary px-3 py-1.5 text-xs font-medium text-text-primary hover:bg-bg-hover disabled:opacity-40">
          {playing && mode !== 'result' ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          {mode === 'all' ? 'Pause/resume' : 'Play clip'}
        </button>
        <button onClick={playAll} disabled={!clips.length}
          className="flex items-center gap-1.5 rounded-md bg-bg-tertiary px-3 py-1.5 text-xs font-medium text-text-primary hover:bg-bg-hover disabled:opacity-40">
          <Film className="h-3.5 w-3.5" /> Play timeline
        </button>
        {sel && mode === 'clip' && (
          <button onClick={() => ref.current && onSplit(sel.key, ref.current.currentTime)}
            title="Cut this clip in two at the playhead"
            className="flex items-center gap-1.5 rounded-md bg-bg-tertiary px-3 py-1.5 text-xs font-medium text-text-primary hover:bg-bg-hover">
            <Scissors className="h-3.5 w-3.5" /> Split here
          </button>
        )}
        {mode === 'result' && sel && (
          <button onClick={() => setMode('clip')} className="rounded-md px-2 py-1.5 text-xs text-text-secondary hover:text-text-primary">Back to editing</button>
        )}
        {resultName && mode !== 'result' && (
          <button onClick={() => setMode('result')} className="rounded-md px-2 py-1.5 text-xs text-accent-green hover:brightness-110">Watch export</button>
        )}
        <span className="ml-auto text-xs tabular-nums text-text-muted">{tl !== null ? fmt(tl) : src ? fmt(time) : ''}</span>
      </div>

      {sel && mode === 'clip' && (
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 rounded-lg border border-border-dim bg-bg-secondary p-3">
          <TrimSlider label="Start" value={sel.start} min={0} max={Math.max(0, sel.end - 0.2)} total={srcDur}
            onChange={(v) => onSetTrim(sel.key, { start: v })}
            onHere={() => ref.current && onSetTrim(sel.key, { start: Math.min(ref.current.currentTime, sel.end - 0.2) })} />
          <TrimSlider label="End" value={sel.end} min={sel.start + 0.2} max={srcDur} total={srcDur}
            onChange={(v) => onSetTrim(sel.key, { end: v })}
            onHere={() => ref.current && onSetTrim(sel.key, { end: Math.max(ref.current.currentTime, sel.start + 0.2) })} />
          <div className="flex items-center gap-2 text-xs">
            <Gauge className="h-3.5 w-3.5 text-text-muted" />
            <span className="text-text-secondary">Speed</span>
            <div className="flex flex-1 flex-wrap gap-0.5">
              {SPEEDS.map((sp) => (
                <button key={sp} onClick={() => { onSetTrim(sel.key, { speed: sp }); if (ref.current) ref.current.playbackRate = sp; }}
                  className={cn('rounded px-1.5 py-0.5 tabular-nums', (sel.speed || 1) === sp ? 'bg-accent-purple text-white' : 'text-text-muted hover:bg-bg-hover hover:text-text-primary')}>
                  {sp}×
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <button onClick={() => onSetTrim(sel.key, { volume: (sel.volume ?? 1) === 0 ? 1 : 0 })} aria-label="Mute"
              className="text-text-muted hover:text-text-primary">
              {(sel.volume ?? 1) === 0 ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
            </button>
            <span className="text-text-secondary">Volume</span>
            <input type="range" min={0} max={2} step={0.05} value={sel.volume ?? 1}
              onChange={(e) => { onSetTrim(sel.key, { volume: Number(e.target.value) }); if (ref.current) ref.current.volume = Math.min(1, Number(e.target.value)); }}
              className="flex-1 accent-[var(--color-accent-purple)]" aria-label="Clip volume" />
            <span className="w-9 text-right tabular-nums text-text-muted">{Math.round((sel.volume ?? 1) * 100)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

function TrimSlider({ label, value, min, max, total, onChange, onHere }: {
  label: string; value: number; min: number; max: number; total: number;
  onChange: (v: number) => void; onHere: () => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-text-secondary">{label} <span className="tabular-nums text-text-muted">{value.toFixed(2)}s</span></span>
        <button onClick={onHere} className="text-accent-purple hover:brightness-125">Set to playhead</button>
      </div>
      <input type="range" min={0} max={Math.max(total, 0.1)} step={0.04} value={value}
        onChange={(e) => onChange(Math.min(max, Math.max(min, Number(e.target.value))))}
        className="w-full accent-[var(--color-accent-purple)]" aria-label={`${label} time`} />
    </div>
  );
}

// ── Small controls ───────────────────────────────────────────────────────────

function RangeRow({ label, value, min, max, step, fmt: show, onChange }: {
  label: string; value: number; min: number; max: number; step: number; fmt: (v: number) => string; onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-[74px] flex-shrink-0 text-text-secondary">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-[var(--color-accent-purple)]" aria-label={label} />
      <span className="w-10 text-right tabular-nums text-text-muted">{show(value)}</span>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1.5 block text-[11px] font-medium uppercase tracking-wider text-text-muted">{label}</label>
      {children}
    </div>
  );
}

function Segmented({ value, options, onChange }: { value: string; options: [string, string][]; onChange: (v: string) => void }) {
  return (
    <div className="flex rounded-md bg-bg-tertiary p-0.5">
      {options.map(([v, label]) => (
        <button key={v} onClick={() => onChange(v)}
          className={cn('relative flex-1 rounded px-2 py-1 text-xs font-medium transition-colors',
            value === v ? 'text-text-primary' : 'text-text-muted hover:text-text-secondary')}>
          {value === v && <motion.span layoutId={`seg-${options.map((o) => o[0]).join('')}`} className="absolute inset-0 rounded bg-bg-hover" transition={{ type: 'spring', stiffness: 500, damping: 38 }} />}
          <span className="relative">{label}</span>
        </button>
      ))}
    </div>
  );
}
