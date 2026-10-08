// main.jsx - the JavaScript entry point: mounts <App /> into index.html.

// StrictMode runs extra checks in development to catch common React mistakes.
import { StrictMode } from "react";
// createRoot is React 18+'s API for attaching React to a DOM element.
import { createRoot } from "react-dom/client";
// The top-level component of our UI.
import App from "./App.jsx";
// Global styles (Vite injects the CSS into the page).
import "./styles.css";

// Find <div id="root"> from index.html and render the app inside it.
createRoot(document.getElementById("root")).render(
  // Wrap the app in StrictMode (has no effect in production builds).
  <StrictMode>
    <App />
  </StrictMode>,
);
