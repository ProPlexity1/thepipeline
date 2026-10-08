import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, AlertCircle, Clock, Loader2, Square, X, TerminalSquare, ChevronDown } from 'lucide-react';
import type { GenerationJob, GenerationStatus } from '../types';
import { cn } from '../utils/cn';

const RUNNING: GenerationStatus[] = ['queued', 'loading_model', 'generating', 'post_processing'];

const STATUS: Record<GenerationStatus, { label: string; color: string; icon: React.ReactNode }> = {
  idle: { label: 'Idle', color: 'text-text-muted', icon: null },
  queued: { label: 'Waiting in line', color: 'text-accent-amber', icon: <Clock className="h-3.5 w-3.5" /> },
  loading_model: { label: 'Loading model', color: 'text-accent-cyan', icon: <Loader2 className="h-3.5 w-3.5 animate-spin" /> },
  generating: { label: 'Generating', color: 'text-accent-purple', icon: <Loader2 className="h-3.5 w-3.5 animate-spin" /> },
  post_processing: { label: 'Finishing', color: 'text-accent-blue', icon: <Loader2 className="h-3.5 w-3.5 animate-spin" /> },
  done: { label: 'Finished', color: 'text-accent-green', icon: <CheckCircle2 className="h-3.5 w-3.5" /> },
  error: { label: 'Failed', color: 'text-accent-red', icon: <AlertCircle className="h-3.5 w-3.5" /> },
  cancelled: { label: 'Cancelled', color: 'text-text-muted', icon: <Square className="h-3.5 w-3.5" /> },
};

export const clock = (s: number) => {
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
};

export default function JobCard({ job, onCancel, onDismiss }: {
  job: GenerationJob; onCancel: () => void; onDismiss: () => void;
}) {
  const running = RUNNING.includes(job.status);
  const conf = STATUS[job.status] || STATUS.queued;
  const [now, setNow] = useState(Date.now());
  const [showLog, setShowLog] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);

  // Follow new output like a terminal, unless the user scrolled up to read.
  useEffect(() => {
    const el = logRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [job.logs?.length, job.logs?.[job.logs.length - 1], showLog]);

  const elapsed = ((job.endTime ?? now) - job.startTime) / 1000;
  const finalSeconds = job.elapsedSeconds ?? elapsed;
  const est = job.estimateSeconds ?? null;
  let timeNote = '';
  if (running && job.status !== 'queued') {
    if (est) {
      const left = est - elapsed;
      timeNote = left > 5 ? `about ${clock(left)} left` : 'almost done (taking a little longer than usual)';
    } else if (job.eta > 0) {
      timeNote = `~${clock(job.eta)} left (first run, estimating)`;
    }
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1, transition: { type: 'spring', stiffness: 400, damping: 30 } }}
      exit={{ opacity: 0, scale: 0.98, transition: { duration: 0.15 } }}
      className={cn('rounded-xl border p-3 space-y-2',
        job.status === 'error' ? 'bg-accent-red/8 border-accent-red/30'
          : job.status === 'done' ? 'bg-accent-green/5 border-accent-green/25'
          : 'bg-bg-tertiary border-border-dim')}
    >
      <div className="flex items-center justify-between gap-2">
        <div className={cn('flex min-w-0 items-center gap-1.5 text-xs font-medium', conf.color)}>
          {conf.icon}
          <span className="truncate">{job.message && running ? job.message : conf.label}</span>
        </div>
        <div className="flex flex-shrink-0 items-center gap-1.5">
          <span className="font-mono text-xs tabular-nums text-text-primary" title="Time since you pressed Generate">
            {clock(job.status === 'done' ? finalSeconds : elapsed)}
          </span>
          {running ? (
            <button onClick={onCancel}
              className="rounded-md px-2 py-0.5 text-[11px] text-text-muted hover:text-accent-red hover:bg-accent-red/10">
              Cancel
            </button>
          ) : (
            <button onClick={onDismiss} aria-label="Dismiss"
              className="rounded-md p-1 text-text-muted hover:text-text-primary hover:bg-bg-hover">
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      <p className="text-xs text-text-secondary line-clamp-1">{job.prompt}</p>
      {job.summary && <p className="text-[11px] text-text-muted">{job.summary}</p>}
      {job.error && <p className="text-xs text-accent-red">{job.error}</p>}
      {job.status === 'done' && (
        <p className="text-xs text-accent-green">Finished in {clock(finalSeconds)} on this PC</p>
      )}

      {running && (
        <>
          <div className="h-2 rounded-full bg-accent-purple/15 overflow-hidden ring-1 ring-inset ring-accent-purple/20">
            <div
              className={cn('h-full rounded-full bg-gradient-to-r from-accent-purple via-accent-blue to-accent-cyan shadow-[0_0_12px_rgba(155,123,255,0.55)] transition-[width] duration-700 ease-out',
                job.status !== 'generating' && 'progress-striped')}
              style={{ width: `${Math.max(job.progress, job.status === 'queued' ? 2 : 4)}%` }}
            />
          </div>
          {timeNote && <p className="text-[11px] text-text-muted">{timeNote}</p>}
        </>
      )}

      {(job.logs?.length ?? 0) > 0 && (
        <div>
          <button onClick={() => setShowLog((v) => !v)}
            className="flex items-center gap-1 text-[11px] text-text-muted hover:text-text-secondary">
            <TerminalSquare className="h-3 w-3" />
            {showLog ? 'Hide details' : 'Show details'}
            <ChevronDown className={cn('h-3 w-3 transition-transform', showLog && 'rotate-180')} />
          </button>
          <AnimatePresence initial={false}>
            {showLog && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1, transition: { duration: 0.22, ease: [0.22, 1, 0.36, 1] } }}
                exit={{ height: 0, opacity: 0, transition: { duration: 0.15 } }}
                className="overflow-hidden"
              >
                <div
                  ref={logRef}
                  onScroll={(e) => {
                    const el = e.currentTarget;
                    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
                  }}
                  className="mt-1.5 max-h-36 overflow-y-auto rounded-lg border border-border-dim bg-[#0a0b0e] px-2.5 py-2 font-mono text-[10.5px] leading-relaxed text-[#b9c2d0] select-text"
                  role="log"
                  aria-live="off"
                >
                  {job.logs!.map((l, i) => (
                    <div key={i} className={cn('whitespace-pre-wrap break-words',
                      /error|failed|traceback/i.test(l) ? 'text-accent-red' : /step \d+/i.test(l) ? 'text-[#d7c9ff]' : '')}>
                      <span className="select-none text-[#5d6575]">› </span>{l}
                    </div>
                  ))}
                  {running && <span className="inline-block h-3 w-1.5 animate-pulse bg-[#b9c2d0]/70 align-middle" />}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </motion.div>
  );
}
