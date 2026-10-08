import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Download, Mic, Pause, Play, Sparkles, Trash2, Wand2, Loader2, AudioLines } from 'lucide-react';
import type { AudioItem, GenerationJob, VoiceModelInfo } from '../types';
import { api, audioUrl, del } from '../api';
import { cn } from '../utils/cn';
import JobCard from './JobCard';

interface Props {
  audio: AudioItem[];
  jobs: GenerationJob[];
  onSpeak: (body: { model_id: string; text: string; voice?: string; instruct?: string; language?: string; speed?: number; title?: string }) => Promise<string | null>;
  onRefresh: () => void;
  onCancelJob: (id: string) => void;
  onDismissJob: (id: string) => void;
  onGoToModels: () => void;
}

/** Ready-made descriptions for the voice designer; the user can edit them freely. */
const VOICE_STYLES = [
  { label: 'Warm narrator', text: 'A warm, confident male voice-over artist in his 40s with a natural accent. Calm, reassuring, trustworthy delivery, relaxed pace, sounds like a real person talking, not announcing.' },
  { label: 'Deep cinematic', text: 'A deep, rich male narrator in his 40s, cinematic but natural, steady and authoritative, measured pace with short natural pauses.' },
  { label: 'Friendly local', text: 'A friendly, grounded male voice in his 30s, conversational and genuine like a local business owner talking to a client, slight smile in the voice.' },
  { label: 'Professional woman', text: 'A clear, warm female voice in her 30s, professional and confident, natural conversational delivery, trustworthy and calm.' },
  { label: 'Energetic promo', text: 'An energetic, upbeat young female presenter, bright and enthusiastic but natural, quick lively pace, smiling.' },
  { label: 'Documentary', text: 'A thoughtful, gentle older male narrator in his 60s, slow and measured, warm documentary tone with natural breathing.' },
];

const LANGUAGES = ['English', 'Chinese', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian'];
const KEY = 'pipeline.voices';

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch { return {}; }
}

export default function VoicesPanel({ audio, jobs, onSpeak, onRefresh, onCancelJob, onDismissJob, onGoToModels }: Props) {
  const saved = loadPrefs();
  const [models, setModels] = useState<VoiceModelInfo[]>([]);
  const [modelId, setModelId] = useState<string>(saved.modelId || '');
  const [voice, setVoice] = useState<string>(saved.voice || 'am_michael');
  const [instruct, setInstruct] = useState<string>(saved.instruct || VOICE_STYLES[0].text);
  const [language, setLanguage] = useState<string>(saved.language || 'English');
  const [speed, setSpeed] = useState<number>(saved.speed || 1);
  const [text, setText] = useState<string>(saved.text || '');
  const [busy, setBusy] = useState(false);

  useEffect(() => { api<VoiceModelInfo[]>('/voices').then(setModels).catch(() => {}); }, []);
  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify({ modelId, voice, instruct, language, speed, text })); } catch { /* ignore */ }
  }, [modelId, voice, instruct, language, speed, text]);

  const installed = models.filter((m) => m.installed);
  // Prefer the voice designer (most natural); fall back to presets.
  const model = installed.find((m) => m.id === modelId) || installed.find((m) => m.voice_design) || installed[0];
  const voicesByLang = useMemo(() => {
    const g = new Map<string, VoiceModelInfo['voices']>();
    for (const v of model?.voices || []) g.set(v.language, [...(g.get(v.language) || []), v]);
    return [...g.entries()];
  }, [model]);
  const voiceJobs = jobs.filter((j) => j.kind === 'voice');

  const generate = async () => {
    if (!model || !text.trim()) return;
    setBusy(true);
    await onSpeak(model.voice_design
      ? { model_id: model.id, text: text.trim(), instruct, language }
      : { model_id: model.id, text: text.trim(), voice, speed });
    setBusy(false);
  };

  if (models.length && !installed.length) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center">
        <div className="max-w-md">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-green/15 text-accent-green"><Mic className="h-6 w-6" /></div>
          <h2 className="text-lg font-semibold text-text-primary">Add a voice model</h2>
          <p className="mt-2 text-sm text-text-secondary">Voices turn scripts into natural voice-overs on this PC.</p>
          <button onClick={onGoToModels} className="mt-5 inline-flex items-center gap-2 rounded-lg bg-accent-purple px-4 py-2 text-sm font-semibold text-white">
            <Download className="h-4 w-4" /> Choose a voice model
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-1 overflow-hidden">
      <aside className="flex w-[380px] flex-shrink-0 flex-col gap-4 overflow-y-auto border-r border-border-dim bg-bg-secondary p-4">
        <div>
          <h2 className="text-base font-semibold text-text-primary">Voices</h2>
          <p className="mt-1 text-xs leading-relaxed text-text-muted">Write a script, choose or describe a voice, and get a natural voice-over.</p>
        </div>

        {installed.length > 1 && (
          <div className="flex rounded-md bg-bg-tertiary p-0.5">
            {installed.map((m) => (
              <button key={m.id} onClick={() => setModelId(m.id)}
                className={cn('flex-1 rounded px-2 py-1.5 text-xs font-medium', model?.id === m.id ? 'bg-bg-hover text-text-primary' : 'text-text-muted hover:text-text-secondary')}>
                {m.voice_design ? 'Describe a voice' : 'Preset voices'}
              </button>
            ))}
          </div>
        )}

        {model?.voice_design ? (
          <>
            <Field label="Voice">
              <div className="mb-2 flex flex-wrap gap-1">
                {VOICE_STYLES.map((s) => (
                  <button key={s.label} onClick={() => setInstruct(s.text)}
                    className={cn('rounded-full border px-2.5 py-1 text-[11px]', instruct === s.text ? 'border-accent-green/60 bg-accent-green/10 text-accent-green' : 'border-border-dim text-text-secondary hover:border-border-active')}>
                    {s.label}
                  </button>
                ))}
              </div>
              <textarea value={instruct} onChange={(e) => setInstruct(e.target.value)} rows={4} maxLength={1000}
                placeholder="Describe the voice: age, gender, accent, tone, pace…"
                className="w-full resize-none rounded-md border border-border-dim bg-bg-tertiary px-2.5 py-2 text-xs leading-relaxed text-text-primary outline-none focus:border-accent-purple/60" />
            </Field>
            <Field label="Language">
              <select value={language} onChange={(e) => setLanguage(e.target.value)}
                className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary outline-none">
                {LANGUAGES.map((l) => <option key={l}>{l}</option>)}
              </select>
            </Field>
          </>
        ) : (
          <>
            <Field label="Voice">
              <select value={voice} onChange={(e) => setVoice(e.target.value)}
                className="w-full rounded-md border border-border-dim bg-bg-tertiary px-2 py-1.5 text-xs text-text-primary outline-none">
                {voicesByLang.map(([lang, vs]) => (
                  <optgroup key={lang} label={lang}>
                    {vs.map((v) => <option key={v.id} value={v.id}>{v.name} ({v.gender})</option>)}
                  </optgroup>
                ))}
              </select>
            </Field>
            <Field label={`Speed ${speed.toFixed(2)}×`}>
              <input type="range" min={0.7} max={1.4} step={0.05} value={speed} onChange={(e) => setSpeed(Number(e.target.value))}
                className="w-full accent-[var(--color-accent-purple)]" />
            </Field>
          </>
        )}

        <Field label="Script">
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={9} maxLength={5000}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate(); }}
            placeholder="Type what the voice should say. Write numbers the way they should be spoken."
            className="w-full resize-none rounded-md border border-border-dim bg-bg-tertiary px-2.5 py-2 text-sm leading-relaxed text-text-primary outline-none focus:border-accent-purple/60" />
          <p className="mt-1 text-right text-[11px] text-text-muted">~{Math.round(text.trim().split(/\s+/).filter(Boolean).length / 2.6)}s spoken</p>
        </Field>

        <motion.button whileTap={{ scale: 0.98 }} disabled={busy || !text.trim() || !model} onClick={generate}
          className="flex items-center justify-center gap-2 rounded-lg bg-accent-purple px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-accent-purple/20 hover:brightness-110 disabled:opacity-40 disabled:shadow-none">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : model?.voice_design ? <Wand2 className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
          Generate voice-over
        </motion.button>
        <p className="-mt-2 text-[11px] text-text-muted">
          {model?.voice_design ? 'Most natural. About a minute per script; uses the graphics card.' : 'Quick: a few seconds, runs on the processor.'}
        </p>

        <div className="space-y-2">
          <AnimatePresence initial={false}>
            {voiceJobs.map((j) => <JobCard key={j.id} job={j} onCancel={() => onCancelJob(j.id)} onDismiss={() => onDismissJob(j.id)} />)}
          </AnimatePresence>
        </div>
      </aside>

      <section className="flex-1 overflow-y-auto p-5">
        <h3 className="mb-3 text-xs font-medium uppercase tracking-wider text-text-muted">Your voice-overs</h3>
        {audio.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center gap-2 text-sm text-text-muted">
            <AudioLines className="h-10 w-10 opacity-30" /> Voice-overs you generate appear here. Add them to videos in the Editor.
          </div>
        ) : (
          <div className="space-y-2">
            {audio.map((a) => <AudioRow key={a.name} item={a} onDelete={async () => { await del(`/audio/${encodeURIComponent(a.name)}`).catch(() => {}); onRefresh(); }} />)}
          </div>
        )}
      </section>
    </div>
  );
}

function AudioRow({ item, onDelete }: { item: AudioItem; onDelete: () => void }) {
  const ref = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const dur = item.duration || 0;
  return (
    <motion.div layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
      className="group flex items-center gap-3 rounded-xl border border-border-dim bg-bg-secondary p-3 hover:border-border-active">
      <button onClick={() => { const a = ref.current; if (!a) return; if (a.paused) a.play(); else a.pause(); }}
        aria-label={playing ? 'Pause' : 'Play'}
        className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full bg-accent-purple text-white hover:brightness-110">
        {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
      </button>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm text-text-primary">{item.title}</p>
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-bg-tertiary">
          <div className="h-full bg-accent-purple transition-[width] duration-200" style={{ width: dur ? `${(t / dur) * 100}%` : '0%' }} />
        </div>
        <p className="mt-1 truncate text-[11px] text-text-muted">
          {dur ? `${dur.toFixed(1)}s · ` : ''}{item.instruct ? `“${item.instruct.slice(0, 70)}…”` : item.voice || ''}
        </p>
      </div>
      <button onClick={onDelete} aria-label="Delete" className="rounded p-1.5 text-text-muted opacity-0 hover:text-accent-red group-hover:opacity-100">
        <Trash2 className="h-4 w-4" />
      </button>
      <audio ref={ref} src={audioUrl(item.name)} preload="none" onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
        onEnded={() => { setPlaying(false); setT(0); }} onTimeUpdate={(e) => setT(e.currentTarget.currentTime)} />
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
