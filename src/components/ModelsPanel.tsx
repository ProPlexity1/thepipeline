import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Download, Trash2, CheckCircle2, HardDrive, Pause, Clock, Zap, AlertCircle,
  AlertTriangle, RefreshCw, Sparkles, Eraser,
} from 'lucide-react';
import type { ModelInfo, GPUInfo, StorageReport, SystemInfo, ModelKind } from '../types';
import { cn } from '../utils/cn';
import { formatBytes, formatDuration } from '../api';

interface ModelsPanelProps {
  models: ModelInfo[];
  gpu: GPUInfo | null;
  system: SystemInfo | null;
  storage: StorageReport | null;
  onDownload: (id: string, acceptLicense?: boolean) => void;
  onCancel: (id: string) => void;
  onDelete: (id: string) => void;
  onDeleteOrphan: (name: string) => Promise<void>;
  onRefreshStorage: () => void;
}

type Fit = { ok: boolean; level: 'good' | 'tight' | 'no'; reason: string };

const KINDS: { kind: ModelKind; tab: string; title: string; desc: string }[] = [
  { kind: 'chat', tab: 'Agent', title: 'Agent brains', desc: 'Chat models that plan projects, research, write scripts and prompts, and drive the other models. They understand images too.' },
  { kind: 'video', tab: 'Video', title: 'Video models', desc: 'Turn a text description into a video.' },
  { kind: 'enhancer', tab: 'Enhance', title: 'Enhancers', desc: 'Upscale a finished video. Use from a video\'s preview, or turn on "Enhance" when generating.' },
  { kind: 'image', tab: 'Images', title: 'Image models', desc: 'Make pictures, character designs and reference images for consistent videos.' },
  { kind: 'voice', tab: 'Voices', title: 'Voice models', desc: 'Turn scripts into natural voice-overs, design new voices, keep each character sounding the same.' },
];
const TAB_KEY = 'pipeline.modelsTab';

function hardwareFit(m: ModelInfo, gpu: GPUInfo | null, system: SystemInfo | null): Fit {
  if (!gpu?.detected) return { ok: false, level: 'no', reason: 'Needs an NVIDIA graphics card.' };
  const vram = system?.vram_gb || gpu.vram;
  const ram = system?.ram_gb || 0;
  if (vram + 0.5 < m.minimum_vram_gb)
    return { ok: false, level: 'no', reason: `Needs ${m.minimum_vram_gb} GB of video memory; your GPU has ${Math.round(vram)} GB.` };
  if (ram && m.minimum_ram_gb && ram + 1 < m.minimum_ram_gb)
    return { ok: false, level: 'no', reason: `Needs ${m.minimum_ram_gb} GB of system memory; this PC has ${Math.round(ram)} GB.` };
  if (vram + 0.5 < m.recommended_vram_gb)
    return { ok: true, level: 'tight', reason: 'Runs on your GPU using memory-saving mode.' };
  return { ok: true, level: 'good', reason: 'Runs well on your hardware.' };
}

export default function ModelsPanel({
  models, gpu, system, storage, onDownload, onCancel, onDelete, onDeleteOrphan, onRefreshStorage,
}: ModelsPanelProps) {
  const [confirming, setConfirming] = useState<string | null>(null);
  const [gateOpen, setGateOpen] = useState<string | null>(null);
  const [accepted, setAccepted] = useState(false);
  const kinds = KINDS.filter((k) => models.some((m) => m.kind === k.kind));
  const [tab, setTab] = useState<ModelKind>(() => {
    try { return (localStorage.getItem(TAB_KEY) as ModelKind) || 'video'; } catch { return 'video'; }
  });
  const activeKind = kinds.find((k) => k.kind === tab) || kinds.find((k) => k.kind === 'video') || kinds[0];
  const chooseTab = (k: ModelKind) => {
    setTab(k);
    try { localStorage.setItem(TAB_KEY, k); } catch { /* storage unavailable */ }
  };
  // Best model per kind for this PC: the highest-ranked one that fits.
  const best = new Map<ModelKind, string>();
  for (const k of KINDS) {
    const fitting = models.filter((m) => m.kind === k.kind && hardwareFit(m, gpu, system).ok)
      .sort((a, b) => b.quality_rank - a.quality_rank || Number(b.recommended) - Number(a.recommended));
    if (fitting[0]) best.set(k.kind, fitting[0].id);
  }

  useEffect(() => {
    if (gateOpen) document.getElementById(`gate-${gateOpen}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [gateOpen]);

  const install = (m: ModelInfo) => {
    if (m.license_gate && !m.downloadedBytes) {
      setGateOpen(m.id);
      setAccepted(false);
    } else {
      onDownload(m.id, !!m.license_gate);
    }
  };

  const ownBytes = (id: string) => storage?.models.find((m) => m.id === id)?.own_bytes ?? 0;
  const sharedInfo = (m: ModelInfo) => {
    const rec = storage?.models.find((x) => x.id === m.id);
    return (rec?.shared || []).map((k) => storage?.shared.find((s) => s.key === k)).filter(Boolean) as StorageReport['shared'];
  };
  const usedTotal = storage?.models_bytes ?? 0;
  const orphanBytes = storage?.orphans.reduce((s, o) => s + o.bytes, 0) ?? 0;
  const diskTotal = storage?.disk_total_bytes ?? 1;
  const diskFree = storage?.disk_free_bytes ?? 0;
  const otherUsed = Math.max(diskTotal - diskFree - usedTotal - (storage?.outputs_bytes ?? 0), 0);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <div>
          <h2 className="text-xl font-semibold text-text-primary">Models</h2>
          <p className="text-sm text-text-secondary mt-1">
            Models download once and then run fully on this PC: no account, no limits.
          </p>
        </div>

        {/* Storage */}
        <section className="surface rounded-xl border border-border-dim p-4 space-y-3" aria-label="Storage">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-blue/15">
              <HardDrive className="h-4.5 w-4.5 text-accent-blue" />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-text-primary">Disk space</span>
                <span className="text-xs text-text-muted">{formatBytes(diskFree)} free of {formatBytes(diskTotal)}</span>
              </div>
              <div className="mt-1.5 flex h-2 rounded-full bg-bg-primary overflow-hidden" role="img"
                aria-label={`Models use ${formatBytes(usedTotal)}, videos ${formatBytes(storage?.outputs_bytes ?? 0)}, free ${formatBytes(diskFree)}`}>
                <div className="bg-text-muted/30" style={{ width: `${(otherUsed / diskTotal) * 100}%` }} />
                <div className="bg-accent-purple" style={{ width: `${(usedTotal / diskTotal) * 100}%` }} />
                <div className="bg-accent-cyan" style={{ width: `${((storage?.outputs_bytes ?? 0) / diskTotal) * 100}%` }} />
              </div>
              <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-muted">
                <Legend color="bg-accent-purple" label={`AI models ${formatBytes(usedTotal)}`} />
                <Legend color="bg-accent-cyan" label={`Your videos ${formatBytes(storage?.outputs_bytes ?? 0)}`} />
                <Legend color="bg-text-muted/30" label="Other files" />
              </div>
            </div>
            <button onClick={onRefreshStorage} title="Refresh" aria-label="Refresh storage"
              className="self-start rounded-md p-1.5 text-text-muted hover:text-text-primary hover:bg-bg-hover">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>

          {storage && storage.orphans.length > 0 && (
            <div className="rounded-lg border border-accent-amber/30 bg-accent-amber/8 p-3">
              <div className="flex items-center gap-2 text-sm text-text-primary">
                <Eraser className="h-4 w-4 text-accent-amber" />
                <span className="font-medium">{formatBytes(orphanBytes)} can be freed</span>
              </div>
              <div className="mt-0.5 flex items-start justify-between gap-3">
                <p className="text-xs text-text-secondary">
                  These folders are from older or unsupported models and aren't used by anything in this version.
                </p>
                {storage.orphans.length > 1 && (confirming === 'orphan:*' ? (
                  <span className="flex flex-shrink-0 items-center gap-1 text-xs">
                    <button onClick={async () => { setConfirming(null); for (const o of storage.orphans) await onDeleteOrphan(o.name); }}
                      className="rounded-md bg-accent-red/15 px-2 py-0.5 font-medium text-accent-red hover:bg-accent-red/25">
                      Delete all {formatBytes(orphanBytes)}
                    </button>
                    <button onClick={() => setConfirming(null)} className="rounded-md px-2 py-0.5 text-text-muted hover:bg-bg-hover">Keep</button>
                  </span>
                ) : (
                  <button onClick={() => setConfirming('orphan:*')}
                    className="flex-shrink-0 rounded-md px-2 py-0.5 text-xs font-medium text-accent-amber hover:bg-accent-amber/10">
                    Remove all
                  </button>
                ))}
              </div>
              <ul className="mt-2 divide-y divide-border-dim/60">
                {storage.orphans.map((o) => (
                  <li key={o.name} className="flex items-center justify-between py-1.5 text-xs">
                    <span className="font-mono text-text-secondary">{o.name}</span>
                    <span className="flex items-center gap-3">
                      <span className="text-text-muted">{formatBytes(o.bytes)}</span>
                      {confirming === `orphan:${o.name}` ? (
                        <span className="flex items-center gap-1">
                          <button onClick={() => { onDeleteOrphan(o.name); setConfirming(null); }}
                            className="rounded-md bg-accent-red/15 px-2 py-0.5 font-medium text-accent-red hover:bg-accent-red/25">
                            Delete
                          </button>
                          <button onClick={() => setConfirming(null)} className="rounded-md px-2 py-0.5 text-text-muted hover:bg-bg-hover">
                            Keep
                          </button>
                        </span>
                      ) : (
                        <button onClick={() => setConfirming(`orphan:${o.name}`)}
                          className="rounded-md px-2 py-0.5 text-text-muted hover:text-accent-red hover:bg-accent-red/10">
                          Remove
                        </button>
                      )}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>

        {/* Model types */}
        <div className="flex gap-1 rounded-xl bg-bg-secondary p-1" role="tablist">
          {kinds.map((k) => {
            const count = models.filter((m) => m.kind === k.kind && m.downloaded).length;
            const active = activeKind?.kind === k.kind;
            return (
              <button key={k.kind} role="tab" aria-selected={active} onClick={() => chooseTab(k.kind)}
                className={cn('relative flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                  active ? 'text-text-primary' : 'text-text-muted hover:text-text-secondary')}>
                {active && (
                  <motion.span layoutId="models-tab" className="absolute inset-0 rounded-lg bg-bg-hover"
                    transition={{ type: 'spring', stiffness: 500, damping: 38 }} />
                )}
                <span className="relative">{k.tab}{count > 0 && <span className="ml-1.5 text-xs text-accent-green">{count}</span>}</span>
              </button>
            );
          })}
        </div>

        {/* Model cards for the selected type, best fit for this PC first */}
        <div className="space-y-4">
          {models.filter((m) => m.kind === activeKind?.kind)
            .sort((a, b) => Number(b.id === best.get(b.kind)) - Number(a.id === best.get(a.kind))
              || Number(b.downloaded) - Number(a.downloaded) || b.quality_rank - a.quality_rank)
            .map((model, i) => {
            const sectionHeader = i === 0;
            const fit = hardwareFit(model, gpu, system);
            const installedBytes = ownBytes(model.id) + sharedInfo(model).reduce((s, x) => s + x.bytes, 0);
            const sharedWithOthers = sharedInfo(model).filter((s) => s.used_by_installed.some((u) => u !== model.id));
            const doneBytes = model.downloadedBytes ?? (model.downloadProgress / 100) * model.size * 1024 ** 3;
            const totalBytes = model.totalBytes || model.size * 1024 ** 3;
            const needBytes = Math.max(totalBytes - (model.downloading ? doneBytes : 0), 0);
            const lowDisk = !model.downloaded && storage && needBytes + 3 * 1024 ** 3 > diskFree;

            return (
              <div key={model.id}>
              {sectionHeader && (
                <div className="mb-3">
                  <h3 className="text-sm font-semibold text-text-primary">{activeKind?.title}</h3>
                  <p className="text-xs text-text-secondary mt-0.5">{activeKind?.desc}</p>
                </div>
              )}
              <motion.article
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0, transition: { delay: Math.min(i, 6) * 0.04, duration: 0.28, ease: [0.22, 1, 0.36, 1] } }}
                className={cn('surface rounded-xl border p-5 hover:border-border-active',
                  model.downloaded ? 'border-accent-green/30' : 'border-border-dim')}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="text-sm font-semibold text-text-primary">{model.name}</h3>
                      {model.downloaded && (
                        <span className="flex items-center gap-1 text-xs text-accent-green">
                          <CheckCircle2 className="h-3 w-3" /> Installed
                        </span>
                      )}
                      {best.get(model.kind) === model.id && (
                        <span className="flex items-center gap-1 rounded-full bg-accent-green/10 px-2 py-0.5 text-[10px] font-semibold text-accent-green">
                          <Zap className="h-3 w-3" /> Best for this PC
                        </span>
                      )}
                      {!model.downloaded && best.get(model.kind) !== model.id && fit.level === 'good' && model.recommended_vram_gb > 0 && (
                        <span className="flex items-center gap-1 rounded-full bg-accent-purple/10 px-2 py-0.5 text-[10px] font-semibold text-accent-purple">
                          <Sparkles className="h-3 w-3" /> Good fit
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs text-text-secondary max-w-xl">{model.description}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-text-muted">
                      {model.speed_label && <span className="flex items-center gap-1"><Clock className="h-3 w-3" />{model.speed_label}</span>}
                      {model.kind === 'video' && <>
                        <span>Up to {model.resolution.split('×')[1]}p</span>
                        <span>{model.duration}</span>
                        <span>{model.capabilities.audio_generation ? 'With sound' : 'No sound'}</span>
                      </>}
                      {model.kind === 'chat' && (model.capabilities as any).vision && <span>Understands images</span>}
                      <span>{model.size_gb} GB</span>
                      {model.license && <span>{model.license}</span>}
                    </div>
                    <p className={cn('mt-2 flex items-center gap-1.5 text-xs',
                      fit.level === 'no' ? 'text-accent-red' : fit.level === 'tight' ? 'text-accent-amber' : 'text-accent-green')}>
                      {fit.level === 'no' ? <AlertCircle className="h-3.5 w-3.5" /> : fit.level === 'tight' ? <AlertTriangle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
                      {fit.reason}
                    </p>
                  </div>

                  <div className="flex flex-col items-end gap-1.5 flex-shrink-0">
                    {model.downloaded ? (
                      confirming === model.id ? (
                        <div className="flex items-center gap-1">
                          <button onClick={() => { onDelete(model.id); setConfirming(null); }}
                            className="rounded-lg bg-accent-red/15 px-3 py-1.5 text-xs font-medium text-accent-red hover:bg-accent-red/25">
                            Remove {formatBytes(installedBytes - sharedWithOthers.reduce((s, x) => s + x.bytes, 0))}
                          </button>
                          <button onClick={() => setConfirming(null)} className="rounded-lg px-2 py-1.5 text-xs text-text-muted hover:bg-bg-hover">
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button onClick={() => setConfirming(model.id)}
                          className="flex items-center gap-1.5 rounded-lg border border-border-dim px-3 py-1.5 text-xs text-text-secondary hover:text-accent-red hover:border-accent-red/30 hover:bg-accent-red/5 transition-colors">
                          <Trash2 className="h-3.5 w-3.5" /> Remove
                        </button>
                      )
                    ) : model.downloading ? (
                      <button onClick={() => onCancel(model.id)}
                        className="flex items-center gap-1.5 rounded-lg border border-accent-amber/25 bg-accent-amber/5 px-3 py-1.5 text-xs text-accent-amber hover:bg-accent-amber/10 transition-colors">
                        <Pause className="h-3.5 w-3.5" /> Pause
                      </button>
                    ) : (
                      <button
                        onClick={() => install(model)}
                        disabled={!fit.ok || !!lowDisk || gateOpen === model.id}
                        className={cn('flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
                          fit.ok && !lowDisk && gateOpen !== model.id
                            ? 'bg-gradient-to-r from-accent-purple to-accent-blue text-white shadow-sm hover:shadow-md'
                            : 'bg-bg-tertiary text-text-muted cursor-not-allowed')}
                      >
                        <Download className="h-3.5 w-3.5" />
                        {model.downloadedBytes ? 'Resume' : 'Install'} ({model.size.toFixed(0)} GB)
                      </button>
                    )}
                    {model.downloaded && (
                      <span className="text-[11px] text-text-muted">
                        {formatBytes(installedBytes)} on disk{sharedWithOthers.length ? ' (some shared)' : ''}
                      </span>
                    )}
                    {lowDisk && !model.downloading && (
                      <span className="text-[11px] text-accent-red max-w-[200px] text-right">
                        Not enough free space. Free some up above.
                      </span>
                    )}
                  </div>
                </div>

                {gateOpen === model.id && model.license_gate && !model.downloading && (
                  <div id={`gate-${model.id}`} className="mt-4 rounded-lg border border-accent-purple/30 bg-accent-purple/5 p-3.5">
                    <p className="text-sm font-medium text-text-primary">Licence: {model.license_gate.name}</p>
                    <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-xs text-text-secondary">
                      {model.license_gate.summary.map((line) => <li key={line}>{line}</li>)}
                    </ul>
                    <p className="mt-1.5 text-[11px] text-text-muted select-text">Full text: {model.license_gate.url}</p>
                    <label className="mt-3 flex items-start gap-2 text-xs text-text-primary cursor-pointer">
                      <input type="checkbox" checked={accepted} onChange={(e) => setAccepted(e.target.checked)}
                        className="mt-0.5 accent-accent-purple" />
                      I have read the licence, I am allowed to use this model where I live, and I accept its terms.
                    </label>
                    <div className="mt-3 flex gap-2">
                      <button
                        disabled={!accepted}
                        onClick={() => { onDownload(model.id, true); setGateOpen(null); }}
                        className={cn('rounded-lg px-3 py-1.5 text-xs font-medium',
                          accepted ? 'bg-gradient-to-r from-accent-purple to-accent-blue text-white' : 'bg-bg-tertiary text-text-muted cursor-not-allowed')}
                      >
                        Accept and install ({model.size.toFixed(0)} GB)
                      </button>
                      <button onClick={() => setGateOpen(null)} className="rounded-lg px-3 py-1.5 text-xs text-text-muted hover:bg-bg-hover">
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {model.downloading && (
                  <div className="mt-4 space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-text-secondary">
                        {model.downloadStage === 'verifying' ? 'Checking file integrity…'
                          : model.downloadStage === 'checking' ? 'Preparing…'
                          : model.downloadStage === 'optimizing' ? 'Optimising for faster loading…'
                          : <><span className="font-medium text-text-primary">{model.downloadProgress.toFixed(1)}%</span> · {formatBytes(doneBytes)} of {formatBytes(totalBytes)}</>}
                      </span>
                      <span className="flex items-center gap-3 text-text-muted">
                        <span className="flex items-center gap-1"><Zap className="h-3 w-3" />{model.speedMbps ? `${model.speedMbps.toFixed(1)} MB/s` : '…'}</span>
                        <span className="flex items-center gap-1"><Clock className="h-3 w-3" />{model.etaSeconds ? formatDuration(model.etaSeconds) : '…'}</span>
                      </span>
                    </div>
                    <div className="h-2.5 rounded-full bg-accent-purple/15 ring-1 ring-inset ring-accent-purple/25 overflow-hidden">
                      <div className="h-full rounded-full bg-gradient-to-r from-accent-purple via-accent-blue to-accent-cyan shadow-[0_0_14px_rgba(155,123,255,0.6)] progress-striped transition-[width] duration-700 ease-out"
                        style={{ width: `${Math.max(model.downloadProgress, 1)}%` }} />
                    </div>
                    <p className="text-[11px] text-text-muted">You can keep using the app or pause any time; downloads resume where they stopped.</p>
                  </div>
                )}

                {model.downloadError && !model.downloading && (
                  <div className="mt-3 flex items-start gap-2 rounded-lg border border-accent-red/25 bg-accent-red/5 p-2.5 text-xs text-accent-red">
                    <AlertCircle className="h-4 w-4 flex-shrink-0" />
                    <span className="flex-1">{model.downloadError}</span>
                    {fit.ok && (
                      <button onClick={() => onDownload(model.id, !!model.license_gate)} className="font-medium underline-offset-2 hover:underline">Retry</button>
                    )}
                  </div>
                )}

              </motion.article>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={cn('h-2 w-2 rounded-full', color)} />
      {label}
    </span>
  );
}
