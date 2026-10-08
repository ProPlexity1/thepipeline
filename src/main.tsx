import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MotionConfig } from "framer-motion";
import "./index.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {/* Always animate: progress and state changes are conveyed through motion,
        and many PCs have Windows animation effects switched off by default. */}
    <MotionConfig reducedMotion="never">
      <App />
    </MotionConfig>
  </StrictMode>
);
