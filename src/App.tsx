import { useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';
import { useAppStore, isActiveJob } from './store';
import SetupScreen from './components/SetupScreen';
import Titlebar from './components/Titlebar';
import Sidebar from './components/Sidebar';
import GeneratePanel from './components/GeneratePanel';
import ModelsPanel from './components/ModelsPanel';
import SettingsPanel from './components/SettingsPanel';
import EditorPanel from './components/EditorPanel';
import AgentPanel from './components/AgentPanel';
import ImagesPanel from './components/ImagesPanel';
import VoicesPanel from './components/VoicesPanel';
import { cn } from './utils/cn';

export default function App() {
  const store = useAppStore();

  // Fallback if the user reaches the main app before setup finished starting the engine.
  useEffect(() => {
    if (store.view !== 'setup' && !store.sidecarStatus.running) store.startSidecar();
  }, [store.view]);

  const activeJobCount = store.jobs.filter(isActiveJob).length;

  if (store.view === 'setup') {
    return (
      <div className="h-screen w-screen overflow-hidden bg-bg-primary">
        <SetupScreen
          step={store.setupStep}
          gpu={store.gpu}
          sidecar={store.sidecarStatus}
          sidecarError={store.sidecarError}
          models={store.models}
          onDetectGPU={store.detectGPU}
          onStartBackend={store.startSidecar}
          onComplete={() => store.setView(store.models.some((m) => m.downloaded) ? 'main' : 'models')}
        />
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-bg-primary">
      <Titlebar sidecar={store.sidecarStatus} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          activeView={store.view}
          onViewChange={store.setView}
          gpu={store.gpu}
          activeJobCount={activeJobCount}
          engineReady={store.sidecarStatus.running}
        />

        <AnimatePresence mode="wait">
          <motion.div
            key={store.view}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0, transition: { duration: 0.24, ease: [0.22, 1, 0.36, 1] } }}
            exit={{ opacity: 0, transition: { duration: 0.1, ease: 'easeIn' } }}
            className="flex flex-1 overflow-hidden"
          >
            {store.view === 'main' && (
              <GeneratePanel
                prompt={store.currentPrompt}
                negativePrompt={store.negativePrompt}
                selectedModel={store.selectedModel}
                models={store.models}
                jobs={store.jobs}
                outputs={store.outputs}
                onPromptChange={store.setCurrentPrompt}
                onNegativePromptChange={store.setNegativePrompt}
                onSelectModel={store.setSelectedModel}
                onGenerate={store.startGeneration}
                onCancelJob={store.cancelJob}
                onDismissJob={store.dismissJob}
                onDeleteOutput={store.deleteOutput}
                onRevealOutput={store.revealOutput}
                onGoToModels={() => store.setView('models')}
                onNotify={store.notify}
                onEstimate={store.estimateFor}
                onEnhance={store.enhanceVideo}
                images={store.images}
                onRefreshImages={store.refreshImages}
              />
            )}
            {store.view === 'agent' && (
              <AgentPanel models={store.models} onGoToModels={() => store.setView('models')} onNotify={store.notify} />
            )}
            {store.view === 'images' && (
              <ImagesPanel
                models={store.models}
                images={store.images}
                jobs={store.jobs}
                onGenerate={store.startGeneration}
                onRefresh={store.refreshImages}
                onCancelJob={store.cancelJob}
                onDismissJob={store.dismissJob}
                onGoToModels={() => store.setView('models')}
                onGoToVideo={() => store.setView('main')}
                onNotify={store.notify}
                onAnimatePhoto={store.animatePhoto}
              />
            )}
            {store.view === 'voices' && (
              <VoicesPanel
                audio={store.audio}
                jobs={store.jobs}
                onSpeak={store.speak}
                onRefresh={store.refreshAudio}
                onCancelJob={store.cancelJob}
                onDismissJob={store.dismissJob}
                onGoToModels={() => store.setView('models')}
              />
            )}
            {store.view === 'editor' && (
              <EditorPanel
                outputs={store.outputs}
                images={store.images}
                audio={store.audio}
                models={store.models}
                jobs={store.jobs}
                onRender={store.renderEdit}
                onGenerate={store.startGeneration}
                onCancelJob={store.cancelJob}
                onDismissJob={store.dismissJob}
                onNotify={store.notify}
                onRefreshOutputs={store.refreshOutputs}
              />
            )}
            {store.view === 'models' && (
              <ModelsPanel
                models={store.models}
                gpu={store.gpu}
                system={store.system}
                storage={store.storage}
                onDownload={store.downloadModel}
                onCancel={store.cancelDownload}
                onDelete={store.deleteModel}
                onDeleteOrphan={store.deleteOrphan}
                onRefreshStorage={store.refreshStorage}
              />
            )}
            {store.view === 'settings' && (
              <SettingsPanel
                sidecar={store.sidecarStatus}
                storage={store.storage}
                system={store.system}
                gpu={store.gpu}
                onRestartEngine={store.restartEngine}
                onNotify={store.notify}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Notifications */}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[360px] flex-col gap-2" aria-live="polite">
        <AnimatePresence>
          {store.notices.map((n) => (
            <motion.div
              key={n.id}
              layout
              initial={{ opacity: 0, y: 12, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1, transition: { type: 'spring', stiffness: 420, damping: 32 } }}
              exit={{ opacity: 0, x: 24, transition: { duration: 0.15 } }}
              className={cn(
                'pointer-events-auto flex items-start gap-2.5 rounded-xl border px-3.5 py-3 text-sm shadow-2xl shadow-black/40 backdrop-blur',
                n.kind === 'error' && 'border-accent-red/30 bg-bg-secondary/95 text-text-primary',
                n.kind === 'success' && 'border-accent-green/30 bg-bg-secondary/95 text-text-primary',
                n.kind === 'info' && 'border-border-dim bg-bg-secondary/95 text-text-primary'
              )}
              role={n.kind === 'error' ? 'alert' : 'status'}
            >
              {n.kind === 'error' ? <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-accent-red" />
                : n.kind === 'success' ? <CheckCircle2 className="mt-0.5 h-4 w-4 flex-shrink-0 text-accent-green" />
                : <Info className="mt-0.5 h-4 w-4 flex-shrink-0 text-accent-blue" />}
              <span className="flex-1 leading-snug">{n.text}</span>
              <button onClick={() => store.dismissNotice(n.id)} aria-label="Dismiss" className="text-text-muted hover:text-text-primary">
                <X className="h-3.5 w-3.5" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
