import { useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Film, ImagePlus, Loader2, Sparkles, Trash2, Upload, X, Image as ImageIcon, Download, Clapperboard } from 'lucide-react';
import type { GenerationJob, ImageItem, ModelInfo } from '../types';
import { imageUrl, post, del } from '../api';
import { cn } from '../utils/cn';
import JobCard from './JobCard';

const ASPECTS = [
  { id: 'landscape', label: '16:9', w: 16, h: 9 },
  { id: 'portrait', label: '9:16', w: 9, h: 16 },
  { id: 'square', label: '1:1', w: 1, h: 1 },
] as const;

interface Props {
  models: ModelInfo[];
  images: ImageItem[];
  jobs: GenerationJob[];
  onGenerate: (prompt: string, negPrompt: string, modelId: string, settings: Record<string, any>) => Promise<string | null>;
  onRefresh: () => void;
  onCancelJob: (id: string) => void;
  onDismissJob: (id: string) => void;
  onGoToModels: () => void;
  onGoToVideo: () => void;
  onNotify: (kind: 'error' | 'success' | 'info', text: string) => void;
  onAnimatePhoto: (image: string, move: string, seconds: number) => Promise<string | null>;
}

/** Round to the model's 64 px grid. */
const snap = (v: number) => Math.max(512, Math.round(v / 64) * 64);

export default function ImagesPanel(props: Props) {
  const { models, images, jobs, onGenerate, onRefresh, onCancelJob, onDismissJob, onGoToModels, onGoToVideo, onNotify, onAnimatePhoto } = props;
  const imageModels = models.filter((m) => m.kind === 'image' && m.downloaded);
  const i2vModels = models.filter((m) => m.kind === 'video' && m.downloaded && m.capabilities.image_to_video);
  const [modelId, setModelId] = useState('');
  const model = imageModels.find((m) => m.id === modelId) || imageModels[0];
  const [prompt, setPrompt] = useState('');
  const [aspect, setAspect] = useState<(typeof ASPECTS)[number]['id']>('landscape');
  const [preset, setPreset] = useState('balanced');
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<ImageItem | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const imageJobs = jobs.filter((j) => j.kind === 'image' || j.kind === 'motion');
  const profiles: Record<string, any> = model?.profiles || {};
  const presetKey = profiles[preset] ? preset : 'balanced';

  const size = useMemo(() => {
    const p = profiles[presetKey] || { width: 1344, height: 768 };
    const long = Math.max(p.width, p.height);
    const a = ASPECTS.find((x) => x.id === aspect)!;
    return a.w >= a.h ? { width: long, height: snap(long * a.h / a.w) } : { width: snap(long * a.w / a.h), height: long };
  }, [profiles, presetKey, aspect]);

  const generate = async () => {
    if (!model || !prompt.trim()) return;
    setBusy(true);
    await onGenerate(prompt.trim(), '', model.id, { kind: 'image', profile: presetKey, steps: profiles[presetKey]?.steps, ...size });
    setBusy(false);
  };

  const upload = async (files: FileList | null) => {
    for (const f of Array.from(files || []).slice(0, 8)) {
      if (!f.type.startsWith('image/')) continue;
      const data_url = await new Promise<string>((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(String(r.result));
        r.onerror = () => rej(r.error);
        r.readAsDataURL(f);
      });
      try {
        await post('/images/upload', { data_url });
      } catch (e: any) {
        onNotify('error', e.message);
      }
    }
    onRefresh();
  };

  const remove = async (name: string) => {
    await del(`/images/${encodeURIComponent(name)}`).catch(() => {});
    if (open?.name === name) setOpen(null);
    onRefresh();
  };

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Create */}
      <aside className="flex w-[340px] flex-shrink-0 flex-col gap-4 overflow-y-auto border-r border-border-dim bg-bg-secondary p-4">
        <div>
          <h2 className="text-base font-semibold text-text-primary">Images</h2>
          <p className="mt-1 text-xs leading-relaxed text-text-muted">
            Make keyframes, character designs and product shots, then animate them into realistic video.
          </p>
        </div>
        {!model ? (
          <div className="rounded-lg border border-dashed border-border-active p-4 text-center text-sm text-text-secondary">
            No image model installed yet.
            <button onClick={onGoToModels} className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg bg-accent-purple px-3 py-2 text-sm font-semibold text-white">
              <Download className="h-4 w-4" /> Get an image model
            </button>
          </div>
        ) : (
          <>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={7} maxLength={4000}
              onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate(); }}
              placeholder="A photograph of… (subject, appearance, setting, lighting, lens, style)"
              className="w-full resize-none rounded-lg border border-border-dim bg-bg-tertiary px-3 py-2.5 text-sm leading-relaxed text-text-primary outline-none placeholder:text-text-muted focus:border-accent-purple/60" />
            <Field label="Shape">
              <div className="flex rounded-md bg-bg-tertiary p-0.5">
                {ASPECTS.map((a) => (
                  <button key={a.id} onClick={() => setAspect(a.id)}
                    className={cn('flex-1 rounded px-2 py-1.5 text-xs font-medium', aspect === a.id ? 'bg-bg-hover text-text-primary' : 'text-text-muted hover:text-text-secondary')}>
                    {a.label}
                  </button>
                ))}
              </div>
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <Field label="Quality">
                <select value={presetKey} onChange={(e) => setPreset(e.target.value)}
                  className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary outline-none">
                  {Object.entries(profiles).map(([k, p]: [string, any]) => <option key={k} value={k}>{p.label || k}</option>)}
                </select>
              </Field>
              <Field label="Model">
                <select value={model.id} onChange={(e) => setModelId(e.target.value)}
                  className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary outline-none">
                  {imageModels.map((m) => <option key={m.id} value={m.id}>{m.display_name}</option>)}
                </select>
              </Field>
            </div>
            <p className="-mt-1 text-[11px] text-text-muted">{size.width}×{size.height}px</p>
            <motion.button whileTap={{ scale: 0.98 }} disabled={busy || !prompt.trim()} onClick={generate}
              className="flex items-center justify-center gap-2 rounded-lg bg-accent-purple px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-accent-purple/20 hover:brightness-110 disabled:opacity-40 disabled:shadow-none">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />} Generate image
            </motion.button>
          </>
        )}
        <button onClick={() => fileRef.current?.click()}
          className="flex items-center justify-center gap-2 rounded-lg border border-border-dim px-3 py-2 text-sm text-text-secondary hover:border-border-active hover:text-text-primary">
          <Upload className="h-4 w-4" /> Upload a logo or photo
        </button>
        <input ref={fileRef} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden
          onChange={(e) => { upload(e.target.files); e.target.value = ''; }} />
        <div className="space-y-2">
          <AnimatePresence initial={false}>
            {imageJobs.map((j) => <JobCard key={j.id} job={j} onCancel={() => onCancelJob(j.id)} onDismiss={() => onDismissJob(j.id)} />)}
          </AnimatePresence>
        </div>
      </aside>

      {/* Gallery */}
      <section className="flex-1 overflow-y-auto p-5"
        onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); upload(e.dataTransfer.files); }}>
        {images.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-sm text-text-muted">
            <ImageIcon className="h-10 w-10 opacity-30" />
            Your images appear here. You can also drop pictures onto this area.
          </div>
        ) : (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3">
            {images.map((img, i) => (
              <motion.button key={img.name} layout initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1, transition: { delay: Math.min(i, 12) * 0.02 } }}
                onClick={() => setOpen(img)}
                className="group relative aspect-video overflow-hidden rounded-xl border border-border-dim bg-black text-left hover:border-border-active">
                <img src={imageUrl(img.name)} alt={img.prompt || img.name} loading="lazy" className="h-full w-full object-cover transition group-hover:scale-[1.02]" />
                {img.uploaded && <span className="absolute left-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] text-white">Uploaded</span>}
                <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2 opacity-0 transition group-hover:opacity-100">
                  <p className="line-clamp-2 text-[11px] text-white/90">{img.prompt || img.name}</p>
                </div>
              </motion.button>
            ))}
          </div>
        )}
      </section>

      <AnimatePresence>
        {open && (
          <ImageViewer image={open} i2vModels={i2vModels} onClose={() => setOpen(null)} onDelete={() => remove(open.name)}
            onAnimate={async (motionPrompt, mid) => {
              const m = models.find((x) => x.id === mid);
              const p = (m?.profiles || {}).balanced || {};
              const id = await onGenerate(motionPrompt, '', mid, { profile: 'balanced', ...p, start_image: open.name });
              if (id) {
                onNotify('info', 'Animating your image. Follow it on the Video page.');
                setOpen(null);
                onGoToVideo();
              }
            }}
            onGoToModels={onGoToModels}
            onCameraMove={async (move, seconds) => {
              const id = await onAnimatePhoto(open.name, move, seconds);
              if (id) { onNotify('info', 'Making your photo motion. It appears in the Video gallery in a few seconds.'); setOpen(null); }
            }} />
        )}
      </AnimatePresence>
    </div>
  );
}

function ImageViewer({ image, i2vModels, onClose, onDelete, onAnimate, onGoToModels, onCameraMove }: {
  image: ImageItem; i2vModels: ModelInfo[]; onClose: () => void; onDelete: () => void;
  onAnimate: (prompt: string, modelId: string) => void; onGoToModels: () => void;
  onCameraMove: (move: string, seconds: number) => void;
}) {
  const [move, setMove] = useState('push_in');
  const [secs, setSecs] = useState(4);
  const [motionPrompt, setMotionPrompt] = useState('');
  const [mid, setMid] = useState(i2vModels[0]?.id || '');
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6 backdrop-blur-sm" onClick={onClose}
      onKeyDown={(e) => e.key === 'Escape' && onClose()}>
      <motion.div initial={{ scale: 0.96, y: 8 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.96 }}
        onClick={(e) => e.stopPropagation()}
        className="flex max-h-full w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border-dim bg-bg-secondary lg:flex-row">
        <div className="flex min-h-0 flex-1 items-center justify-center bg-black">
          <img src={imageUrl(image.name)} alt="" className="max-h-[80vh] max-w-full object-contain" />
        </div>
        <div className="flex w-full flex-col gap-3 p-4 lg:w-[340px]">
          <div className="flex items-start justify-between">
            <p className="text-xs text-text-muted">{image.settings?.width}×{image.settings?.height}{image.uploaded ? ' · uploaded' : ''}</p>
            <button onClick={onClose} aria-label="Close" className="text-text-muted hover:text-text-primary"><X className="h-4 w-4" /></button>
          </div>
          {image.prompt && <p className="max-h-40 overflow-y-auto text-xs leading-relaxed text-text-secondary">{image.prompt}</p>}
          <div className="mt-auto space-y-2 rounded-lg border border-accent-green/30 bg-accent-green/5 p-3">
            <p className="flex items-center gap-1.5 text-sm font-medium text-text-primary"><Clapperboard className="h-4 w-4 text-accent-green" /> Camera move <span className="text-[11px] font-normal text-text-muted">· seconds, full sharpness</span></p>
            <div className="grid grid-cols-4 gap-1">
              {[['push_in', 'Push in'], ['pull_out', 'Pull out'], ['pan_left', 'Pan left'], ['pan_right', 'Pan right'], ['rise', 'Rise'], ['orbit_left', 'Orbit L'], ['orbit_right', 'Orbit R'], ['drift', 'Drift']].map(([id, label]) => (
                <button key={id} onClick={() => setMove(id)}
                  className={'rounded px-1.5 py-1 text-[11px] ' + (move === id ? 'bg-accent-green text-black' : 'bg-bg-secondary text-text-secondary hover:text-text-primary')}>{label}</button>
              ))}
            </div>
            <div className="flex items-center gap-2 text-xs text-text-secondary">
              <input type="range" min={2} max={10} step={0.5} value={secs} onChange={(e) => setSecs(Number(e.target.value))} className="flex-1 accent-[var(--color-accent-green)]" aria-label="Length" />
              <span className="w-8 tabular-nums">{secs}s</span>
            </div>
            <button onClick={() => onCameraMove(move, secs)}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-accent-green px-3 py-2 text-sm font-semibold text-black hover:brightness-110">
              <Film className="h-4 w-4" /> Make photo motion
            </button>
            <p className="text-[10px] text-text-muted">People: use push in or pull out. Orbits suit scenery.</p>
          </div>
          <div className="space-y-2 rounded-lg border border-border-dim bg-bg-tertiary/50 p-3">
            <p className="flex items-center gap-1.5 text-sm font-medium text-text-primary"><Film className="h-4 w-4 text-accent-cyan" /> Animate with AI <span className="text-[11px] font-normal text-text-muted">· people move, takes minutes</span></p>
            {i2vModels.length === 0 ? (
              <button onClick={onGoToModels} className="w-full rounded-md border border-border-dim px-3 py-2 text-xs text-text-secondary hover:text-text-primary">
                Install an image-to-video model first
              </button>
            ) : (
              <>
                <textarea value={motionPrompt} onChange={(e) => setMotionPrompt(e.target.value)} rows={3} maxLength={2000}
                  placeholder="What moves, and how the camera moves"
                  className="w-full resize-none rounded-md border border-border-dim bg-bg-secondary px-2.5 py-2 text-xs text-text-primary outline-none focus:border-accent-purple/60" />
                {i2vModels.length > 1 && (
                  <select value={mid} onChange={(e) => setMid(e.target.value)}
                    className="w-full rounded-md border border-border-dim bg-bg-secondary px-2 py-1.5 text-xs text-text-primary outline-none">
                    {i2vModels.map((m) => <option key={m.id} value={m.id}>{m.display_name}</option>)}
                  </select>
                )}
                <button disabled={!motionPrompt.trim()} onClick={() => onAnimate(motionPrompt.trim(), mid)}
                  className="flex w-full items-center justify-center gap-2 rounded-md bg-accent-purple px-3 py-2 text-sm font-semibold text-white disabled:opacity-40">
                  <ImagePlus className="h-4 w-4" /> Animate
                </button>
              </>
            )}
          </div>
          <button onClick={onDelete} className="flex items-center gap-1.5 self-start text-xs text-text-muted hover:text-accent-red">
            <Trash2 className="h-3.5 w-3.5" /> Delete image
          </button>
        </div>
      </motion.div>
    </motion.div>
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
