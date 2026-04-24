import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  const clientPort = Number(env.CLIENT_PORT || "5173");
  const serverPort = String(env.SERVER_PORT || "8000");
  const backendProxyTarget = (env.VITE_DEV_PROXY_TARGET || env.BACKEND_BASE_URL || `http://localhost:${serverPort}`)
    .replace(/\/$/, "");

  return {
    plugins: [react()],
    envDir: "..",
    base: "/",
    build: {
      outDir: "dist",
      emptyOutDir: true,
      assetsDir: "assets"
    },
    server: {
      port: clientPort,
      proxy: {
        "/api": {
          target: backendProxyTarget,
          changeOrigin: true
        }
      }
    }
  };
});
