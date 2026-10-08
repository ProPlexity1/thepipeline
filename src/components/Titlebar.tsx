import { Sparkles, Wifi, WifiOff } from 'lucide-react';
import type { SidecarStatus } from '../types';
import { cn } from '../utils/cn';

interface TitlebarProps {
  sidecar: SidecarStatus;
}

// The native Windows title bar provides minimise/maximise/close; this is just
// the in-app header strip.
export default function Titlebar({ sidecar }: TitlebarProps) {
  return (
    <div className="flex h-10 items-center justify-between border-b border-border-dim bg-bg-secondary px-4 flex-shrink-0">
      <div className="flex items-center gap-2">
        <div className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-accent-purple to-accent-blue">
          <Sparkles className="h-3.5 w-3.5 text-white" />
        </div>
        <span className="text-sm font-semibold text-text-primary">The Pipeline</span>
      </div>
      <div
        className={cn(
          'flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs',
          sidecar.running ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'
        )}
        role="status"
      >
        {sidecar.running ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
        {sidecar.running ? 'AI engine ready' : 'AI engine offline'}
      </div>
    </div>
  );
}
