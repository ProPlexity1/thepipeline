import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  ArrowUp, Bot, Brain, ChevronDown, Globe, ImagePlus, Loader2, Moon, MessageSquarePlus, Square,
  Trash2, Wrench, X, Film, ListChecks, BookOpen, AlertCircle, Sparkles, Download, Clapperboard, Image as ImageIcon,
  Mic, Hourglass, CheckCircle2,
} from 'lucide-react';
import type { ModelInfo } from '../types';
import { api, post, del } from '../api';
import { onAgentEvent } from '../store';
import { cn } from '../utils/cn';
import Markdown from './Markdown';

// ── Types ────────────────────────────────────────────────────────────────────

type Part =
  | { type: 'reasoning'; text: string }
  | { type: 'text'; text: string }
  | { type: 'tool'; id: string; name: string; arguments: any; result?: any; seconds?: number }
  | { type: 'notice'; text: string };

interface Turn {
  user: { text: string; images: string[] };
  parts: Part[];
  streaming: boolean;
  error?: string;
  status?: string;  // e.g. waiting for the graphics card
}

interface ProductionSummary {
  id: string; title: string; status: string; mode: string; shots_total: number; shots_done: number;
  shots_failed: number; final: string | null; error: string | null;
}

const ATTACH_NOTE = /\n*\[Attached images saved in the gallery as: [^\]]*\]$/;

interface ConvSummary { id: string; title: string; updated_at: string; messages: number }

interface AgentStatus {
  state: 'sleeping' | 'loading' | 'ready';
  loaded_model: string | null;
  model_id: string | null;
  installed: string[];
  gpu_busy: boolean;
}

const CONV_KEY = 'pipeline.agent.conversation';

/** Stored conversation messages -> display turns. */
function toTurns(messages: any[]): Turn[] {
  const turns: Turn[] = [];
  for (const m of messages) {
    if (m.role === 'user') {
      turns.push({ user: { text: (m.content || '').replace(ATTACH_NOTE, ''), images: m.images || [] }, parts: [], streaming: false });
      continue;
    }
    const t = turns[turns.length - 1];
    if (!t) continue;
    if (m.role === 'assistant' && m.notice) {
      t.parts.push({ type: 'notice', text: m.content || '' });
    } else if (m.role === 'assistant') {
      if (m.reasoning) t.parts.push({ type: 'reasoning', text: m.reasoning });
      if (m.content) t.parts.push({ type: 'text', text: m.content });
      for (const c of m.tool_calls || []) {
        let args: any = {};
        try { args = JSON.parse(c.arguments || '{}'); } catch { /* shown raw below */ }
        t.parts.push({ type: 'tool', id: c.id, name: c.name, arguments: args });
      }
    } else if (m.role === 'tool') {
      const part = t.parts.find((p) => p.type === 'tool' && p.id === m.tool_call_id) as Extract<Part, { type: 'tool' }> | undefined;
      if (part) {
        try { part.result = JSON.parse(m.content); } catch { part.result = m.content; }
      }
    }
  }
  return turns;
}

/** Shrink an image so attachments stay small (the model sees ~1024px anyway). */
function readImage(file: File, max = 1024): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const scale = Math.min(1, max / Math.max(img.width, img.height));
      const c = document.createElement('canvas');
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      c.getContext('2d')!.drawImage(img, 0, 0, c.width, c.height);
      URL.revokeObjectURL(url);
      // PNG keeps logos' transparency and crisp edges; photos are smaller as JPEG.
      resolve(file.type === 'image/png' ? c.toDataURL('image/png') : c.toDataURL('image/jpeg', 0.88));
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Not an image')); };
    img.src = url;
  });
}

const SUGGESTIONS = [
  { icon: <Film className="h-4 w-4" />, text: 'Plan a 30-second action short about a lone swordsman, then make the shots.' },
  { icon: <Sparkles className="h-4 w-4" />, text: 'I want to start an anime series. Help me develop the story and main characters.' },
  { icon: <BookOpen className="h-4 w-4" />, text: 'Research what makes product ads go viral, then pitch me three ideas.' },
  { icon: <ListChecks className="h-4 w-4" />, text: 'Which models do I have installed, and what is each one best for?' },
];

// ── Panel ────────────────────────────────────────────────────────────────────

export default function AgentPanel({ models, onGoToModels, onNotify }: {
  models: ModelInfo[];
  onGoToModels: () => void;
  onNotify: (kind: 'error' | 'success' | 'info', text: string) => void;
}) {
  const [status, setStatus] = useState<AgentStatus | null>(null);
  const [convs, setConvs] = useState<ConvSummary[]>([]);
  const [convId, setConvId] = useState<string | null>(() => {
    try { return localStorage.getItem(CONV_KEY); } catch { return null; }
  });
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState('');
  const [images, setImages] = useState<string[]>([]);
  const [modelChoice, setModelChoice] = useState<string>('');
  const [productions, setProductions] = useState<Record<string, ProductionSummary>>({});
  const scrollRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const fileRef = useRef<HTMLInputElement>(null);
  // A chat created by send() already shows its first turn; don't reload it from disk (it isn't saved yet).
  const justCreated = useRef<string | null>(null);

  const chatModels = models.filter((m) => m.kind === 'chat');
  const installed = chatModels.filter((m) => m.downloaded);
  const recommended = chatModels.find((m) => m.recommended) || chatModels[0];
  const busy = turns.some((t) => t.streaming);

  const refreshStatus = useCallback(() => api<AgentStatus>('/agent/status').then(setStatus).catch(() => {}), []);
  const refreshConvs = useCallback(() => api<ConvSummary[]>('/agent/conversations').then(setConvs).catch(() => {}), []);

  useEffect(() => { refreshStatus(); refreshConvs(); }, [refreshStatus, refreshConvs, models]);
  useEffect(() => {
    api<ProductionSummary[]>('/productions')
      .then((list) => setProductions(Object.fromEntries(list.map((p) => [p.id, p]))))
      .catch(() => {});
  }, []);

  useEffect(() => {
    try { if (convId) localStorage.setItem(CONV_KEY, convId); else localStorage.removeItem(CONV_KEY); } catch { /* ignore */ }
    if (!convId) { setTurns([]); return; }
    if (justCreated.current === convId) { justCreated.current = null; return; }
    api(`/agent/conversations/${convId}`)
      .then((c) => setTurns(toTurns(c.messages)))
      .catch(() => setConvId(null));
  }, [convId]);

  // Live events for the open conversation.
  useEffect(() => onAgentEvent((msg) => {
    if (msg.type === 'production') {
      setProductions((p) => ({ ...p, [msg.production.id]: msg.production }));
      return;
    }
    if (msg.type === 'agent_state') {
      setStatus((s) => (s ? { ...s, state: msg.state, loaded_model: msg.state === 'sleeping' ? null : msg.model_id } : s));
      return;
    }
    if (msg.conv_id !== convId) {
      if (msg.event === 'done') refreshConvs();
      if (msg.event === 'notice') onNotify('info', msg.text.replace(/\*\*|`/g, '').slice(0, 160));
      return;
    }
    setTurns((prev) => {
      if (!prev.length) return prev;
      const next = [...prev];
      const t = { ...next[next.length - 1], parts: [...next[next.length - 1].parts] };
      next[next.length - 1] = t;
      const last = t.parts[t.parts.length - 1];
      switch (msg.event) {
        case 'status':
          t.status = msg.message;
          break;
        case 'notice':
          t.parts.push({ type: 'notice', text: msg.text });
          break;
        case 'delta': {
          t.status = undefined;
          const kind = msg.kind === 'reasoning' ? 'reasoning' : 'text';
          if (last && last.type === kind) t.parts[t.parts.length - 1] = { ...last, text: last.text + msg.text };
          else t.parts.push({ type: kind, text: msg.text });
          break;
        }
        case 'tool_call':
          t.status = undefined;
          t.parts.push({ type: 'tool', id: msg.id, name: msg.name, arguments: msg.arguments });
          break;
        case 'tool_result':
          t.parts = t.parts.map((p) => (p.type === 'tool' && p.id === msg.id ? { ...p, result: msg.result, seconds: msg.seconds } : p));
          break;
        case 'done':
          t.streaming = false;
          t.status = undefined;
          break;
        case 'error':
          t.streaming = false;
          t.status = undefined;
          t.error = msg.error;
          break;
      }
      return next;
    });
    if (msg.event === 'done' || msg.event === 'error') refreshConvs();
  }), [convId, refreshConvs, onNotify]);

  // Keep the newest text in view unless the user scrolled up to read.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const send = async (override?: string) => {
    const body = (override ?? text).trim();
    if ((!body && !images.length) || busy) return;
    let id = convId;
    try {
      if (!id) {
        const c = await post<{ id: string }>('/agent/conversations');
        id = c.id;
        justCreated.current = id;
        setConvId(id);
      }
      setTurns((prev) => [...prev, { user: { text: body, images }, parts: [], streaming: true }]);
      setText('');
      setImages([]);
      stick.current = true;
      await post(`/agent/conversations/${id}/message`, { text: body, images, model_id: modelChoice || undefined });
    } catch (err: any) {
      setTurns((prev) => prev.map((t, i) => (i === prev.length - 1 ? { ...t, streaming: false, error: err.message } : t)));
    }
  };

  const stop = () => { if (convId) post(`/agent/conversations/${convId}/cancel`).catch(() => {}); };

  const addFiles = async (files: FileList | File[]) => {
    const list = Array.from(files).filter((f) => f.type.startsWith('image/')).slice(0, 6 - images.length);
    try {
      const urls = await Promise.all(list.map((f) => readImage(f)));
      setImages((prev) => [...prev, ...urls].slice(0, 6));
    } catch {
      onNotify('error', "That file couldn't be read as an image.");
    }
  };

  const removeConv = async (id: string) => {
    await del(`/agent/conversations/${id}`).catch(() => {});
    if (id === convId) setConvId(null);
    refreshConvs();
  };

  const activeModel = chatModels.find((m) => m.id === (modelChoice || status?.model_id));

  if (!installed.length) {
    return <NoAgentModel model={recommended} onGoToModels={onGoToModels} />;
  }

  return (
    <div className="flex flex-1 overflow-hidden">
      {/* Conversations */}
      <aside className="flex w-[240px] flex-shrink-0 flex-col border-r border-border-dim bg-bg-secondary">
        <div className="p-3">
          <button onClick={() => setConvId(null)}
            className="flex w-full items-center gap-2 rounded-lg border border-border-dim bg-bg-tertiary px-3 py-2 text-sm font-medium text-text-primary transition hover:border-border-active hover:bg-bg-hover">
            <MessageSquarePlus className="h-4 w-4" /> New chat
          </button>
        </div>
        <div className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
          {convs.map((c) => (
            <div key={c.id} className={cn('group flex items-center rounded-md', c.id === convId ? 'bg-bg-hover' : 'hover:bg-bg-hover/60')}>
              <button onClick={() => setConvId(c.id)} className="min-w-0 flex-1 truncate px-2.5 py-2 text-left text-[13px] text-text-secondary group-hover:text-text-primary">
                {c.title}
              </button>
              <button onClick={() => removeConv(c.id)} aria-label="Delete chat"
                className="mr-1 rounded p-1 text-text-muted opacity-0 transition hover:text-accent-red group-hover:opacity-100">
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Chat */}
      <section className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-3 border-b border-border-dim px-5 py-2.5">
          <StatePill status={status} />
          <div className="relative">
            <select value={modelChoice || status?.model_id || ''} onChange={(e) => setModelChoice(e.target.value)}
              className="appearance-none rounded-md border border-border-dim bg-bg-tertiary py-1 pl-2.5 pr-7 text-xs text-text-primary outline-none hover:border-border-active">
              {installed.map((m) => <option key={m.id} value={m.id}>{m.display_name}</option>)}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3 w-3 -translate-y-1/2 text-text-muted" />
          </div>
          {status?.state !== 'sleeping' && (
            <button onClick={() => post('/agent/sleep').catch(() => {})} disabled={busy}
              className="ml-auto flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-40"
              title="Unload the agent to free graphics memory">
              <Moon className="h-3.5 w-3.5" /> Sleep now
            </button>
          )}
        </header>

        <div ref={scrollRef} onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }} className="flex-1 overflow-y-auto">
          {turns.length === 0 ? (
            <Welcome onPick={(t) => send(t)} model={activeModel} />
          ) : (
            <div className="mx-auto max-w-3xl space-y-6 px-5 py-6">
              {turns.map((t, i) => <TurnView key={i} turn={t} productions={productions} />)}
            </div>
          )}
        </div>

        {/* Composer */}
        <div className="border-t border-border-dim bg-bg-primary px-5 py-3"
          onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files); }}>
          <div className="mx-auto max-w-3xl">
            <AnimatePresence>
              {images.length > 0 && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
                  className="mb-2 flex gap-2 overflow-hidden">
                  {images.map((src, i) => (
                    <div key={i} className="group relative h-16 w-16 overflow-hidden rounded-lg border border-border-dim bg-black">
                      <img src={src} alt="" className="h-full w-full object-contain" />
                      <button onClick={() => setImages((p) => p.filter((_, j) => j !== i))} aria-label="Remove image"
                        className="absolute right-0.5 top-0.5 rounded-full bg-black/70 p-0.5 text-white opacity-0 group-hover:opacity-100">
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
            <div className="flex items-end gap-2 rounded-xl border border-border-dim bg-bg-secondary p-2 focus-within:border-accent-purple/50">
              <button onClick={() => fileRef.current?.click()} title="Attach images (logos, references…)"
                className="rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary">
                <ImagePlus className="h-4 w-4" />
              </button>
              <input ref={fileRef} type="file" accept="image/*" multiple hidden
                onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ''; }} />
              <textarea
                value={text} rows={1} maxLength={20000}
                onChange={(e) => setText(e.target.value)}
                onPaste={(e) => { if (e.clipboardData.files.length) { e.preventDefault(); addFiles(e.clipboardData.files); } }}
                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder="Describe what you want to make…"
                className="max-h-48 min-h-[36px] flex-1 resize-none bg-transparent px-1 py-2 text-sm text-text-primary outline-none placeholder:text-text-muted"
                style={{ fieldSizing: 'content' } as any}
              />
              {busy ? (
                <button onClick={stop} aria-label="Stop" className="rounded-lg bg-bg-hover p-2 text-text-primary hover:bg-accent-red/80">
                  <Square className="h-4 w-4" />
                </button>
              ) : (
                <motion.button whileTap={{ scale: 0.92 }} onClick={() => send()} aria-label="Send"
                  disabled={!text.trim() && !images.length}
                  className="rounded-lg bg-accent-purple p-2 text-white transition hover:brightness-110 disabled:opacity-30">
                  <ArrowUp className="h-4 w-4" />
                </motion.button>
              )}
            </div>
            <p className="mt-1.5 text-center text-[11px] text-text-muted">
              Runs on this PC. It sleeps while videos generate, and web research can be turned off in Settings.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

// ── Pieces ───────────────────────────────────────────────────────────────────

function StatePill({ status }: { status: AgentStatus | null }) {
  const s = status?.state || 'sleeping';
  const conf = {
    sleeping: { label: status?.gpu_busy ? 'Asleep · GPU busy' : 'Asleep', cls: 'bg-bg-tertiary text-text-muted', icon: <Moon className="h-3 w-3" /> },
    loading: { label: 'Waking up', cls: 'bg-accent-amber/15 text-accent-amber', icon: <Loader2 className="h-3 w-3 animate-spin" /> },
    ready: { label: 'Awake', cls: 'bg-accent-green/15 text-accent-green', icon: <span className="h-1.5 w-1.5 rounded-full bg-accent-green" /> },
  }[s];
  return (
    <span className={cn('flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium', conf.cls)}>
      {conf.icon}{conf.label}
    </span>
  );
}

function Welcome({ onPick, model }: { onPick: (t: string) => void; model?: ModelInfo }) {
  return (
    <div className="mx-auto flex h-full max-w-2xl flex-col items-center justify-center px-6 py-10 text-center">
      <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
        className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-purple/15 text-accent-purple">
        <Bot className="h-6 w-6" />
      </motion.div>
      <h2 className="text-xl font-semibold text-text-primary">What are we making?</h2>
      <p className="mt-2 max-w-md text-sm leading-relaxed text-text-secondary">
        Describe an idea: a short film, a series, an ad for your product. The agent plans it, researches it, writes the
        script and prompts, then hands the work to the video models.
      </p>
      <div className="mt-6 grid w-full grid-cols-2 gap-2">
        {SUGGESTIONS.map((s, i) => (
          <motion.button key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0, transition: { delay: 0.05 * i } }}
            onClick={() => onPick(s.text)}
            className="flex items-start gap-2.5 rounded-xl border border-border-dim bg-bg-secondary p-3 text-left text-[13px] leading-snug text-text-secondary transition hover:border-border-active hover:text-text-primary">
            <span className="mt-0.5 text-accent-purple">{s.icon}</span>{s.text}
          </motion.button>
        ))}
      </div>
      {model && <p className="mt-5 text-xs text-text-muted">Using {model.display_name}</p>}
    </div>
  );
}

function TurnView({ turn, productions }: { turn: Turn; productions: Record<string, ProductionSummary> }) {
  const hasText = turn.parts.some((p) => p.type === 'text' && p.text.trim());
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <div className="max-w-[85%] space-y-2">
          {turn.user.images.length > 0 && (
            <div className="flex flex-wrap justify-end gap-2">
              {turn.user.images.map((src, i) => (
                <img key={i} src={src} alt="" className="max-h-40 rounded-lg border border-border-dim bg-black object-contain" />
              ))}
            </div>
          )}
          {turn.user.text && (
            <div className="whitespace-pre-wrap rounded-2xl rounded-br-md bg-accent-purple/20 px-4 py-2.5 text-sm text-text-primary">
              {turn.user.text}
            </div>
          )}
        </div>
      </div>
      <div className="space-y-2.5">
        {turn.parts.map((p, i) =>
          p.type === 'reasoning' ? <Reasoning key={i} text={p.text} live={turn.streaming && i === turn.parts.length - 1} />
            : p.type === 'tool' ? <ToolCard key={i} part={p} production={p.result?.production_id ? productions[p.result.production_id] : undefined} />
            : p.type === 'notice' ? <Notice key={i} text={p.text} />
            : <div key={i} className="text-sm leading-relaxed text-text-primary"><Markdown text={p.text} /></div>)}
        {turn.streaming && (turn.status || (!hasText && !turn.parts.some((p) => p.type === 'reasoning'))) && (
          <div className="flex items-center gap-2 text-sm text-text-muted">
            {turn.status?.startsWith('Waiting') ? <Hourglass className="h-4 w-4 text-accent-amber" /> : <Loader2 className="h-4 w-4 animate-spin" />}
            {turn.status || 'Waking up the agent…'}
          </div>
        )}
        {turn.error && (
          <div className="flex items-start gap-2 rounded-lg border border-accent-red/30 bg-accent-red/10 px-3 py-2 text-sm text-text-primary">
            <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-accent-red" />{turn.error}
          </div>
        )}
      </div>
    </div>
  );
}

function Reasoning({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-border-dim bg-bg-secondary/60">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-text-muted hover:text-text-secondary">
        {live ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Brain className="h-3.5 w-3.5" />}
        {live ? 'Thinking…' : 'Thought it through'}
        <ChevronDown className={cn('ml-auto h-3.5 w-3.5 transition-transform', open && 'rotate-180')} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
            <div className="max-h-64 overflow-y-auto whitespace-pre-wrap border-t border-border-dim px-3 py-2 text-xs leading-relaxed text-text-muted">{text}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

const TOOL_LABEL: Record<string, (a: any) => string> = {
  web_search: (a) => `Searched the web for “${a.query}”`,
  read_webpage: (a) => `Read ${(() => { try { return new URL(a.url).hostname; } catch { return 'a web page'; } })()}`,
  list_models: () => 'Checked the installed models',
  generate_video: (a) => `Queued a video${a.seconds ? ` (${a.seconds}s)` : ''}`,
  list_jobs: () => 'Checked the job queue',
  generate_image: () => 'Queued an image',
  start_production: (a) => `Started the production “${a.title || 'Untitled'}”`,
  speak: () => 'Recorded a voice line',
  animate_image: (a) => `Animated ${a.image || 'an image'}`,
};

function Notice({ text }: { text: string }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-accent-green/30 bg-accent-green/10 px-3.5 py-2.5 text-sm text-text-primary">
      <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-accent-green" />
      <div className="min-w-0 flex-1"><Markdown text={text} /></div>
    </div>
  );
}

function ProductionProgress({ p }: { p: ProductionSummary }) {
  const pct = p.status === 'done' ? 100 : Math.round((p.shots_done / Math.max(1, p.shots_total)) * 90);
  const label = p.status === 'done' ? 'Finished' : p.status === 'failed' ? `Failed: ${p.error || ''}`
    : p.status === 'cancelled' ? 'Cancelled' : p.status === 'assembling' ? 'Putting it together…'
    : `${p.shots_done} of ${p.shots_total} shots ready${p.shots_failed ? ` · ${p.shots_failed} failed` : ''}`;
  return (
    <div className="border-t border-border-dim px-3 py-2">
      <div className="mb-1 flex justify-between text-[11px] text-text-muted"><span>{label}</span><span>{p.mode === 'video' ? 'Video mode' : 'Image mode'}</span></div>
      <div className="h-1.5 overflow-hidden rounded-full bg-bg-tertiary">
        <motion.div className={cn('h-full rounded-full', p.status === 'failed' ? 'bg-accent-red' : 'bg-accent-purple')} animate={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ToolCard({ part, production }: { part: Extract<Part, { type: 'tool' }>; production?: ProductionSummary }) {
  const [open, setOpen] = useState(false);
  const pending = part.result === undefined;
  const failed = !pending && part.result && part.result.error;
  const label = (TOOL_LABEL[part.name] || (() => part.name))(part.arguments || {});
  const icon = part.name.startsWith('web') || part.name === 'read_webpage' ? <Globe className="h-3.5 w-3.5" />
    : part.name === 'generate_video' ? <Film className="h-3.5 w-3.5" />
    : part.name === 'start_production' ? <Clapperboard className="h-3.5 w-3.5" />
    : part.name === 'generate_image' || part.name === 'animate_image' ? <ImageIcon className="h-3.5 w-3.5" />
    : part.name === 'speak' ? <Mic className="h-3.5 w-3.5" /> : <Wrench className="h-3.5 w-3.5" />;
  return (
    <div className={cn('rounded-lg border text-xs', failed ? 'border-accent-red/30 bg-accent-red/5' : 'border-border-dim bg-bg-secondary/60')}>
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-text-secondary hover:text-text-primary">
        {pending ? <Loader2 className="h-3.5 w-3.5 animate-spin text-accent-purple" /> : <span className={failed ? 'text-accent-red' : 'text-accent-cyan'}>{icon}</span>}
        <span className="truncate">{label}</span>
        {part.seconds !== undefined && <span className="ml-auto flex-shrink-0 text-text-muted">{part.seconds}s</span>}
        <ChevronDown className={cn('h-3.5 w-3.5 flex-shrink-0 transition-transform', part.seconds === undefined && 'ml-auto', open && 'rotate-180')} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
            <div className="border-t border-border-dim px-3 py-2">
              <ToolDetails part={part} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      {production && <ProductionProgress p={production} />}
    </div>
  );
}

function ToolDetails({ part }: { part: Extract<Part, { type: 'tool' }> }) {
  const r = part.result;
  if (r === undefined) return <p className="text-text-muted">Working…</p>;
  if (r?.error) return <p className="text-accent-red">{r.error}</p>;
  if (part.name === 'web_search') {
    return (
      <ul className="space-y-1.5">
        {(r.results || []).map((x: any, i: number) => (
          <li key={i}>
            <p className="font-medium text-text-primary">{x.title}</p>
            <p className="truncate text-accent-blue">{x.url}</p>
            {x.snippet && <p className="text-text-muted">{x.snippet}</p>}
          </li>
        ))}
        {r.source && <li className="text-text-muted">via {r.source}</li>}
      </ul>
    );
  }
  if (part.name === 'generate_video') {
    return <p className="text-text-secondary">Job {r.job_id} queued{r.position_in_queue ? `, position ${r.position_in_queue}` : ''}. Follow it on the Generate page.</p>;
  }
  return <pre className="max-h-56 overflow-auto whitespace-pre-wrap text-[11px] text-text-muted">{JSON.stringify(r, null, 2)}</pre>;
}

function NoAgentModel({ model, onGoToModels }: { model?: ModelInfo; onGoToModels: () => void }) {
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="max-w-md text-center">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-purple/15 text-accent-purple">
          <Bot className="h-6 w-6" />
        </div>
        <h2 className="text-lg font-semibold text-text-primary">Install the agent's brain</h2>
        <p className="mt-2 text-sm leading-relaxed text-text-secondary">
          The agent needs a chat model. It runs privately on this PC, plans your projects, writes scripts and prompts,
          and drives the video, image and voice models.
        </p>
        {model?.downloading ? (
          <div className="mt-5">
            <div className="h-2 overflow-hidden rounded-full bg-bg-tertiary">
              <motion.div className="h-full rounded-full bg-accent-purple" animate={{ width: `${model.progress || 0}%` }} />
            </div>
            <p className="mt-2 text-xs text-text-muted">Downloading {model.display_name}… {Math.round(model.progress || 0)}%</p>
          </div>
        ) : (
          <button onClick={onGoToModels}
            className="mt-5 inline-flex items-center gap-2 rounded-lg bg-accent-purple px-4 py-2 text-sm font-semibold text-white hover:brightness-110">
            <Download className="h-4 w-4" /> Choose a chat model
          </button>
        )}
        {model && !model.downloading && <p className="mt-3 text-xs text-text-muted">Recommended for this PC: {model.display_name} ({model.size_gb} GB)</p>}
      </div>
    </div>
  );
}

