import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Video, Download, Settings, Activity, ChevronDown, Clapperboard, Bot, Image as ImageIcon, Mic } from 'lucide-react';
import type { AppView, GPUInfo } from '../types';
import { cn } from '../utils/cn';
import SystemMonitor from './SystemMonitor';

interface SidebarProps {
  activeView: AppView;
  onViewChange: (view: AppView) => void;
  gpu: GPUInfo | null;
  activeJobCount: number;
  engineReady: boolean;
}

const NAV_ITEMS: { view: AppView; icon: React.ReactNode; label: string }[] = [
  { view: 'agent', icon: <Bot className="h-4 w-4" />, label: 'Agent' },
  { view: 'main', icon: <Video className="h-4 w-4" />, label: 'Video' },
  { view: 'images', icon: <ImageIcon className="h-4 w-4" />, label: 'Images' },
  { view: 'voices', icon: <Mic className="h-4 w-4" />, label: 'Voices' },
  { view: 'editor', icon: <Clapperboard className="h-4 w-4" />, label: 'Editor' },
  { view: 'models', icon: <Download className="h-4 w-4" />, label: 'Models' },
  { view: 'settings', icon: <Settings className="h-4 w-4" />, label: 'Settings' },
];

const PERF_KEY = 'neuralcut.showPerformance';

export default function Sidebar({ activeView, onViewChange, gpu, activeJobCount, engineReady }: SidebarProps) {
  const [showPerf, setShowPerf] = useState(() => {
    try { return localStorage.getItem(PERF_KEY) !== '0'; } catch { return true; }
  });
  const togglePerf = () => {
    setShowPerf((v) => {
      try { localStorage.setItem(PERF_KEY, v ? '0' : '1'); } catch { /* storage unavailable */ }
      return !v;
    });
  };

  return (
    <div className="flex h-full w-48 xl:w-56 flex-col border-r border-border-dim bg-bg-secondary flex-shrink-0">
      <nav className="flex-1 p-3 space-y-1">
        {NAV_ITEMS.map(item => {
          const active = activeView === item.view;
          return (
            <button
              key={item.view}
              onClick={() => onViewChange(item.view)}
              aria-current={active ? 'page' : undefined}
              className={cn(
                'relative flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium',
                active ? 'text-text-primary' : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
              )}
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-lg bg-accent-purple/15 ring-1 ring-inset ring-accent-purple/25"
                  transition={{ type: 'spring', stiffness: 500, damping: 38 }}
                />
              )}
              <span className={cn('relative', active && 'text-accent-purple')}>{item.icon}</span>
              <span className="relative">{item.label}</span>
              {item.view === 'main' && activeJobCount > 0 && (
                <span className="relative ml-auto flex h-5 min-w-5 items-center justify-center rounded-full bg-accent-purple/25 px-1.5 text-xs text-accent-purple">
                  {activeJobCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="border-t border-border-dim p-3">
        <button
          onClick={togglePerf}
          aria-expanded={showPerf}
          title={showPerf ? 'Hide performance graphs' : 'Show performance graphs'}
          className="flex w-full items-center gap-2 rounded-md px-1 py-0.5 text-[11px] text-text-muted hover:text-text-primary"
        >
          <Activity className="h-3.5 w-3.5" />
          <span className="font-medium uppercase tracking-wider">Performance</span>
          <ChevronDown className={`ml-auto h-3.5 w-3.5 transition-transform duration-200 ${showPerf ? '' : '-rotate-90'}`} />
        </button>
        <AnimatePresence initial={false}>
          {showPerf && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1, transition: { duration: 0.25, ease: [0.22, 1, 0.36, 1] } }}
              exit={{ height: 0, opacity: 0, transition: { duration: 0.18 } }}
              className="overflow-hidden"
            >
              <div className="pt-2.5">
                {gpu?.detected && <p className="mb-2 truncate text-[11px] text-text-secondary" title={gpu.name}>{gpu.name}</p>}
                {/* Unmounted while hidden, so it also stops polling. */}
                <SystemMonitor enabled={engineReady} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
