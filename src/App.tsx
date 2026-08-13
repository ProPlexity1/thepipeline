import { useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useAppStore } from './store';
import SetupScreen from './components/SetupScreen';
import Titlebar from './components/Titlebar';
import Sidebar from './components/Sidebar';
import GeneratePanel from './components/GeneratePanel';
import ModelsPanel from './components/ModelsPanel';
import SettingsPanel from './components/SettingsPanel';
import LicensePanel from './components/LicensePanel';

export default function App() {
  const store = useAppStore();

  // Keep this as a fallback if the user reaches the main app before setup finishes.
  useEffect(() => {
    if (store.view === 'main' && !store.sidecarStatus.running) {
      store.startSidecar();
    }
  }, [store.view]);

  const activeJobCount = store.jobs.filter(
    j => j.status !== 'done' && j.status !== 'error' && j.status !== 'idle'
  ).length;

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
          onComplete={() => store.setView('main')}
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
        />

        <AnimatePresence mode="wait">
          <motion.div
            key={store.view}
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -10 }}
            transition={{ duration: 0.15 }}
            className="flex flex-1 overflow-hidden"
          >
            {store.view === 'main' && (
              <GeneratePanel
                prompt={store.currentPrompt}
                negativePrompt={store.negativePrompt}
                selectedModel={store.selectedModel}
                models={store.models}
                jobs={store.jobs}
                galleryItems={store.galleryItems}
                onPromptChange={store.setCurrentPrompt}
                onNegativePromptChange={store.setNegativePrompt}
                onSelectModel={store.setSelectedModel}
                onGenerate={store.startGeneration}
              />
            )}
            {store.view === 'models' && (
              <ModelsPanel
                models={store.models}
                gpu={store.gpu}
                onDownload={store.downloadModel}
                onCancel={store.cancelDownload}
                onDelete={store.deleteModel}
              />
            )}
            {store.view === 'settings' && (
              <SettingsPanel sidecar={store.sidecarStatus} />
            )}
            {store.view === 'license' && (
              <LicensePanel
                license={store.license}
                onValidate={store.validateLicense}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
