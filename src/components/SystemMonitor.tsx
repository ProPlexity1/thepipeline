import { useEffect, useState } from 'react';
import { api } from '../api';

interface Sample {
  t: number;
  cpu: number;
  ram_used_gb: number;
  ram_total_gb: number;
  gpu: number;
  vram_used_gb: number;
  vram_total_gb: number;
  gpu_temp: number;
}

const POINTS = 60;

/** Live CPU / RAM / GPU / VRAM graphs, like Task Manager's Performance tab. */
export default function SystemMonitor({ enabled }: { enabled: boolean }) {
  const [samples, setSamples] = useState<Sample[]>([]);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const tick = async () => {
      try {
        const data = await api<Sample[]>('/system/stats');
        if (alive) setSamples(data.slice(-POINTS));
      } catch {
        /* engine restarting */
      }
    };
    tick();
    const t = setInterval(tick, 1000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [enabled]);

  const last = samples[samples.length - 1];
  const pct = (used: number, total: number) => (total > 0 ? (used / total) * 100 : 0);

  return (
    <div className="space-y-2.5" aria-label="System usage">
      <Graph label="CPU" color="var(--color-accent-blue)"
        values={samples.map((s) => s.cpu)}
        value={last ? `${Math.round(last.cpu)}%` : '–'} />
      <Graph label="Memory" color="var(--color-accent-purple)"
        values={samples.map((s) => pct(s.ram_used_gb, s.ram_total_gb))}
        value={last ? `${last.ram_used_gb.toFixed(1)}/${Math.round(last.ram_total_gb)} GB` : '–'} />
      <Graph label="GPU" color="var(--color-accent-green)"
        values={samples.map((s) => s.gpu)}
        value={last ? `${Math.round(last.gpu)}% · ${Math.round(last.gpu_temp)}°C` : '–'} />
      <Graph label="VRAM" color="var(--color-accent-cyan)"
        values={samples.map((s) => pct(s.vram_used_gb, s.vram_total_gb))}
        value={last ? `${last.vram_used_gb.toFixed(1)}/${Math.round(last.vram_total_gb)} GB` : '–'} />
    </div>
  );
}

function Graph({ label, values, value, color }: { label: string; values: number[]; value: string; color: string }) {
  const w = 100;
  const h = 28;
  // Right-align like Task Manager: newest sample at the right edge, history scrolls left.
  const pad = Math.max(0, POINTS - values.length);
  const pts = values.map((v, i) => {
    const x = ((pad + i) / (POINTS - 1)) * w;
    const y = h - (Math.min(100, Math.max(0, v)) / 100) * (h - 1) - 0.5;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const line = pts.join(' ');
  const area = pts.length ? `${pts[0].split(',')[0]},${h} ${line} ${w},${h}` : '';
  const id = `g-${label}`;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[11px]">
        <span className="font-medium text-text-secondary">{label}</span>
        <span className="font-mono tabular-nums text-text-primary">{value}</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none"
        className="block h-7 w-full rounded-md border border-border-dim bg-bg-primary" aria-hidden>
        <defs>
          <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.45" />
            <stop offset="100%" stopColor={color} stopOpacity="0.04" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1="0" x2={w} y1={h * f} y2={h * f} stroke="var(--color-border-dim)" strokeWidth="0.5"
            vectorEffect="non-scaling-stroke" />
        ))}
        {pts.length > 1 && (
          <>
            <polygon points={area} fill={`url(#${id})`} />
            <polyline points={line} fill="none" stroke={color} strokeWidth="1.5"
              vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
          </>
        )}
      </svg>
    </div>
  );
}
