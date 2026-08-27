import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// AgriNova AI — Phase 12 frontend build config.
// Talks to the Agents-Pipeline API (default http://localhost:8001) and,
// optionally, the RAG-Pipeline API (default http://localhost:8000) — see
// .env.example. No dev-server proxy is required because both FastAPI
// services already send permissive CORS headers (agent_config.py /
// config.py's *_CORS_ORIGINS, default "*").
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    open: false,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
