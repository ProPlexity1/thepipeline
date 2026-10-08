import { useEffect, useState } from 'react';
import { FolderOpen, Cpu, HardDrive, RefreshCw, ShieldCheck, Info, Loader2, Globe } from 'lucide-react';
import type { SidecarStatus, StorageReport, SystemInfo, GPUInfo } from '../types';
import { api, post, formatBytes } from '../api';

interface SettingsPanelProps {
  sidecar: SidecarStatus;
  storage: StorageReport | null;
  system: SystemInfo | null;
  gpu: GPUInfo | null;
  onRestartEngine: () => Promise<void>;
  onNotify: (kind: 'error' | 'success' | 'info', text: string) => void;
}

export default function SettingsPanel({ sidecar, storage, system, gpu, onRestartEngine, onNotify }: SettingsPanelProps) {
  const [restarting, setRestarting] = useState(false);
  const [prefs, setPrefs] = useState<{ allow_web_research: boolean; brave_api_key_set: boolean } | null>(null);
  const [braveKey, setBraveKey] = useState('');
  useEffect(() => { api('/settings').then(setPrefs).catch(() => {}); }, [sidecar.running]);
  const savePrefs = async (values: Record<string, unknown>) => {
    try {
      setPrefs(await api('/settings', { method: 'PUT', body: JSON.stringify(values) }));
    } catch (e: any) {
      onNotify('error', e.message);
    }
  };

  const open = async (which: 'models' | 'outputs' | 'logs') => {
    try {
      await post(`/folders/${which}/open`);
    } catch (err: any) {
      onNotify('error', err.message);
    }
  };

  const restart = async () => {
    setRestarting(true);
    await onRestartEngine();
    setRestarting(false);
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-3xl mx-auto p-6 space-y-6">
        <div>
          <h2 className="text-xl font-semibold text-text-primary">Settings</h2>
          <p className="text-sm text-text-secondary mt-1">Where your files live and how the AI engine is doing.</p>
        </div>

        <Section title="Files" icon={<HardDrive className="h-4 w-4" />}>
          <Row label="Your videos" description={storage?.output_dir || '…'}
            extra={storage ? formatBytes(storage.outputs_bytes) : undefined}>
            <OpenButton onClick={() => open('outputs')} />
          </Row>
          <Row label="AI models" description={storage?.models_dir || '…'}
            extra={storage ? formatBytes(storage.models_bytes) : undefined}>
            <OpenButton onClick={() => open('models')} />
          </Row>
        </Section>

        <Section title="Hardware" icon={<Cpu className="h-4 w-4" />}>
          <Row label="Graphics card" description={gpu?.detected ? `${gpu.name} · ${system?.vram_gb ?? gpu.vram} GB video memory` : 'No NVIDIA GPU found'} />
          <Row label="System memory" description={system ? `${system.ram_gb} GB` : '…'} />
          <Row label="Driver" description={gpu?.driver || '…'} />
        </Section>

        <Section title="AI engine" icon={<RefreshCw className="h-4 w-4" />}>
          <Row label="Status" description={sidecar.running ? `Running (v${sidecar.version}, Python ${sidecar.python_version})` : 'Not running'}>
            <button
              onClick={restart}
              disabled={restarting}
              className="flex items-center gap-1.5 rounded-lg border border-border-dim bg-bg-tertiary px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary hover:border-border-active disabled:opacity-60"
            >
              {restarting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
              Restart
            </button>
          </Row>
          <Row label="Logs" description="Useful when reporting a problem.">
            <OpenButton onClick={() => open('logs')} />
          </Row>
        </Section>

        <Section title="Agent web research" icon={<Globe className="h-4 w-4" />}>
          <Row label="Allow web research" description="The agent may search the web and read pages for facts and references. Turn off to keep it fully offline.">
            <input type="checkbox" checked={!!prefs?.allow_web_research} disabled={!prefs}
              onChange={(e) => savePrefs({ allow_web_research: e.target.checked })}
              className="h-4 w-4 accent-[var(--color-accent-purple)]" />
          </Row>
          <Row label="Better search (optional)"
            description={prefs?.brave_api_key_set ? 'A Brave Search key is saved; the agent searches with it first.'
              : 'Free searches work out of the box. For better results, paste a free Brave Search API key (brave.com/search/api).'}>
            <div className="flex items-center gap-1.5">
              <input type="password" value={braveKey} onChange={(e) => setBraveKey(e.target.value)} placeholder={prefs?.brave_api_key_set ? '••••••••' : 'API key'}
                className="w-36 rounded-md border border-border-dim bg-bg-tertiary px-2 py-1 text-xs text-text-primary outline-none focus:border-accent-purple/60" />
              <button onClick={async () => { await savePrefs({ brave_api_key: braveKey.trim() }); setBraveKey(''); }}
                className="rounded-md border border-border-dim bg-bg-tertiary px-2.5 py-1 text-xs text-text-secondary hover:text-text-primary">
                {braveKey ? 'Save' : prefs?.brave_api_key_set ? 'Remove' : 'Save'}
              </button>
            </div>
          </Row>
        </Section>

        <Section title="Privacy" icon={<ShieldCheck className="h-4 w-4" />}>
          <p className="px-4 py-3 text-sm text-text-secondary">
            Everything runs on this PC. Your prompts, images and videos never leave it. The internet is used to
            download models (checked against their published fingerprints) and, if allowed above, for the agent's web research.
          </p>
        </Section>

        <p className="flex items-center gap-1.5 text-xs text-text-muted">
          <Info className="h-3.5 w-3.5" /> The Pipeline 0.2.0
        </p>
      </div>
    </div>
  );
}

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="surface rounded-xl border border-border-dim overflow-hidden">
      <div className="flex items-center gap-2 border-b border-border-dim px-4 py-3">
        <span className="text-accent-purple">{icon}</span>
        <h3 className="text-sm font-medium text-text-primary">{title}</h3>
      </div>
      <div className="divide-y divide-border-dim/50">{children}</div>
    </section>
  );
}

function Row({ label, description, extra, children }: {
  label: string; description: string; extra?: string; children?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4 px-4 py-3">
      <div className="min-w-0">
        <p className="text-sm text-text-primary">{label}{extra && <span className="ml-2 text-xs text-text-muted">{extra}</span>}</p>
        <p className="text-xs text-text-muted truncate" title={description}>{description}</p>
      </div>
      {children}
    </div>
  );
}

function OpenButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex flex-shrink-0 items-center gap-1.5 rounded-lg border border-border-dim bg-bg-tertiary px-3 py-1.5 text-xs text-text-secondary hover:text-text-primary hover:border-border-active"
    >
      <FolderOpen className="h-3.5 w-3.5" /> Open
    </button>
  );
}
