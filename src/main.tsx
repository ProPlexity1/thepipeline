import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MotionConfig } from "framer-motion";
import "./index.css";
import App from "./App";

// Settings saved before the rename used "neuralcut." keys; carry them over once.
try {
  for (const key of Object.keys(localStorage)) {
    if (!key.startsWith("neuralcut.")) continue;
    const next = "thepipeline." + key.slice("neuralcut.".length);
    if (localStorage.getItem(next) === null) localStorage.setItem(next, localStorage.getItem(key)!);
    localStorage.removeItem(key);
  }
} catch { /* storage unavailable */ }

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* Always animate: progress and state changes are conveyed through motion,
        and many PCs have Windows animation effects switched off by default. */}
    <MotionConfig reducedMotion="never">
      <App />
    </MotionConfig>
  </StrictMode>
);
