import { Wifi, WifiOff } from 'lucide-react';
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
        <img src="/images/logo.png" alt="" className="h-6 w-6 rounded-md" />
        <span className="text-sm font-semibold text-text-primary">ThePipeline</span>
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
