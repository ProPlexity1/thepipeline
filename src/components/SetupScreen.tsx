import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Cpu, Monitor, CheckCircle2, AlertTriangle, Loader2,
  Zap, HardDrive, ArrowRight, Sparkles, Shield
} from 'lucide-react';
import type { GPUInfo, ModelInfo, SidecarStatus } from '../types';
import { cn } from '../utils/cn';

interface SetupScreenProps {
  step: number;
  gpu: GPUInfo | null;
  sidecar: SidecarStatus;
  sidecarError: string | null;
  models: ModelInfo[];
  onDetectGPU: () => void;
  onStartBackend: () => void;
  onComplete: () => void;
}

const SETUP_DONE_KEY = 'neuralcut.setupDone';

function setupDoneBefore() {
  try { return localStorage.getItem(SETUP_DONE_KEY) === '1'; } catch { return false; }
}

export default function SetupScreen({
  step,
  gpu,
  sidecar,
  sidecarError,
  models,
  onDetectGPU,
  onStartBackend,
  onComplete,
}: SetupScreenProps) {
  const [started, setStarted] = useState(false);

  useEffect(() => {
    if (!started) {
      setStarted(true);
      onStartBackend();
      onDetectGPU();
    }
  }, [started, onDetectGPU, onStartBackend]);

  const gpuReady = Boolean(gpu?.detected);
  const allChecked = gpuReady && sidecar.running && models.length > 0;
  const cudaVersion = gpu?.cuda_version || 'Checking';
  const videoModels = models.filter((m) => m.kind === 'video');
  const fitting = videoModels.filter((m) => gpu && gpu.vram + 0.5 >= m.minimum_vram_gb);
  const recommended = fitting.find((m) => m.recommended && m.downloaded) || fitting.find((m) => m.recommended)
    || fitting.find((m) => m.downloaded) || fitting[0];

  const finish = () => {
    try { localStorage.setItem(SETUP_DONE_KEY, '1'); } catch { /* storage unavailable */ }
    onComplete();
  };

  // After the first run, skip straight into the app once everything is ready.
  useEffect(() => {
    if (allChecked && setupDoneBefore()) onComplete();
  }, [allChecked, onComplete]);

  return (
    <div className="flex h-screen w-full items-center justify-center overflow-y-auto bg-bg-primary px-4 py-6">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -left-1/4 -top-1/4 h-1/2 w-1/2 rounded-full bg-accent-purple/5 blur-[120px]" />
        <div className="absolute -bottom-1/4 -right-1/4 h-1/2 w-1/2 rounded-full bg-accent-blue/5 blur-[120px]" />
      </div>

      <AnimatePresence mode="wait">
        {/* Step 0: Splash */}
        {step === 0 && (
          <motion.div
            key="splash"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.5 }}
            className="relative z-10 flex flex-col items-center gap-8"
          >
            <motion.div
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
            >
              <div className="flex h-24 w-24 items-center justify-center rounded-3xl bg-gradient-to-br from-accent-purple to-accent-blue shadow-2xl shadow-accent-purple/30">
                <Sparkles className="h-12 w-12 text-white" />
              </div>
            </motion.div>
            <div className="text-center">
              <h1 className="text-4xl font-bold tracking-tight text-text-primary">The Pipeline</h1>
              <p className="mt-2 text-lg text-text-secondary">Local AI Video Generation</p>
            </div>
            <div className="flex items-center gap-2 text-text-muted">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Starting up…</span>
            </div>
          </motion.div>
        )}

        {/* Step 1: Detecting GPU */}
        {step === 1 && (
          <motion.div
            key="detecting"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
            className="relative z-10 flex flex-col items-center gap-8"
          >
            <div className="relative">
              <motion.div
                className="absolute inset-0 rounded-full bg-accent-blue/20"
                animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <div className="relative flex h-20 w-20 items-center justify-center rounded-full bg-bg-tertiary border border-border-dim">
                <Cpu className="h-10 w-10 text-accent-blue animate-pulse" />
              </div>
            </div>
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-text-primary">Checking your PC</h2>
              <p className="mt-2 text-text-secondary">Looking for a compatible graphics card…</p>
            </div>
            <div className="flex items-center gap-3 rounded-xl bg-bg-secondary border border-border-dim px-6 py-3">
              <Loader2 className="h-5 w-5 animate-spin text-accent-blue" />
              <span className="text-sm text-text-secondary">This only takes a moment</span>
            </div>
          </motion.div>
        )}

        {/* Step 2: GPU Detected - Results */}
        {step === 2 && gpu?.detected && (
          <motion.div
            key="detected"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4 }}
            className="relative z-10 w-full max-w-2xl px-4"
          >
            <div className="surface rounded-2xl border border-border-dim shadow-2xl shadow-black/40 overflow-hidden">
              {/* Header */}
              <div className="border-b border-border-dim px-8 py-6">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-green/20">
                    <CheckCircle2 className="h-5 w-5 text-accent-green" />
                  </div>
                  <div>
                    <h2 className="text-xl font-semibold text-text-primary">GPU Detected</h2>
                    <p className="text-sm text-text-secondary">Your hardware is ready for AI video generation</p>
                  </div>
                </div>
              </div>

              {/* GPU Info */}
              <div className="px-8 py-6 space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <InfoCard icon={<Monitor className="h-4 w-4" />} label="GPU" value={gpu.name} />
                  <InfoCard icon={<HardDrive className="h-4 w-4" />} label="VRAM" value={`${gpu.vram} GB`} />
                  <InfoCard icon={<Shield className="h-4 w-4" />} label="Driver" value={gpu.driver} />
                  <InfoCard icon={<Zap className="h-4 w-4" />} label="CUDA" value={cudaVersion === 'Checking' ? cudaVersion : `v${cudaVersion}`} />
                </div>

                {recommended && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="rounded-xl bg-gradient-to-r from-accent-purple/10 to-accent-blue/10 border border-accent-purple/20 p-4"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <Sparkles className="h-4 w-4 text-accent-purple" />
                      <span className="text-sm font-medium text-text-primary">Recommended for your PC</span>
                    </div>
                    <p className="text-base font-semibold text-text-primary">{recommended.name}</p>
                    <p className="text-xs text-text-secondary mt-0.5">
                      {recommended.downloaded
                        ? 'Already installed and ready to use.'
                        : `${recommended.description} One-time download of about ${recommended.size.toFixed(0)} GB from the Models page.`}
                    </p>
                    <p className="text-xs text-text-muted mt-1">
                      {fitting.length} of {videoModels.length} available models run on your {gpu.vram} GB graphics card.
                    </p>
                  </motion.div>
                )}

                {/* Sidecar Status */}
                <div>
                  <h3 className="text-sm font-medium text-text-muted mb-3 uppercase tracking-wider">Getting ready</h3>
                  <div className="grid grid-cols-2 gap-2">
                    <CheckItem label="Graphics card" checked={gpuReady} />
                    <CheckItem label="AI engine started" checked={sidecar.running} failed={!!sidecarError} />
                    <CheckItem label="Model list loaded" checked={models.length > 0} failed={!!sidecarError} />
                    <CheckItem label="Private: runs offline" checked />
                  </div>
                  {sidecarError && (
                    <div className="mt-3 flex items-start justify-between gap-3 rounded-lg border border-accent-amber/30 bg-accent-amber/10 px-3 py-2 text-xs text-accent-amber">
                      <span>{sidecarError}</span>
                      <button onClick={onStartBackend} className="flex-shrink-0 font-medium hover:underline">Try again</button>
                    </div>
                  )}
                </div>
              </div>

              {/* Footer */}
              <div className="border-t border-border-dim px-8 py-4 flex justify-end">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={finish}
                  disabled={!allChecked}
                  className={cn(
                    'flex items-center gap-2 rounded-xl px-6 py-2.5 font-medium text-sm transition-all',
                    allChecked
                      ? 'bg-gradient-to-r from-accent-purple to-accent-blue text-white shadow-lg shadow-accent-purple/25 hover:shadow-accent-purple/40'
                      : 'bg-bg-tertiary text-text-muted cursor-not-allowed'
                  )}
                >
                  Start creating
                  <ArrowRight className="h-4 w-4" />
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {/* No GPU detected fallback */}
        {step === 2 && (!gpu || !gpu.detected) && (
          <motion.div
            key="no-gpu"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="relative z-10 max-w-md px-4"
          >
            <div className="rounded-2xl bg-bg-secondary border border-accent-amber/30 p-8 text-center">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-accent-amber/20 mb-4">
                <AlertTriangle className="h-8 w-8 text-accent-amber" />
              </div>
              <h2 className="text-xl font-semibold text-text-primary mb-2">No NVIDIA GPU Detected</h2>
              <p className="text-text-secondary text-sm mb-6">
                The Pipeline needs an NVIDIA graphics card with at least 8 GB of video memory.
                If you have one, install or update the NVIDIA driver from nvidia.com, restart, and try again.
              </p>
              <button
                onClick={onDetectGPU}
                className="rounded-xl bg-bg-tertiary border border-border-dim px-5 py-2 text-sm text-text-primary hover:border-border-active"
              >
                Check again
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function InfoCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg bg-bg-tertiary/50 border border-border-dim p-3">
      <div className="flex items-center gap-1.5 text-text-muted mb-1">
        {icon}
        <span className="text-xs uppercase tracking-wider">{label}</span>
      </div>
      <p className="text-sm font-medium text-text-primary truncate">{value}</p>
    </div>
  );
}

function CheckItem({ label, checked, failed }: { label: string; checked: boolean; failed?: boolean }) {
  return (
    <div className="flex items-center gap-2 rounded-lg bg-bg-tertiary/30 border border-border-dim px-3 py-2">
      {checked ? (
        <CheckCircle2 className="h-4 w-4 text-accent-green flex-shrink-0" />
      ) : failed ? (
        <AlertTriangle className="h-4 w-4 text-accent-amber flex-shrink-0" />
      ) : (
        <Loader2 className="h-4 w-4 text-text-muted animate-spin flex-shrink-0" />
      )}
      <span className={cn('text-sm', checked ? 'text-text-primary' : 'text-text-muted')}>{label}</span>
    </div>
  );
}
