import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "");
  const clientPort = Number(env.CLIENT_PORT || "5173");
  const serverPort = String(env.SERVER_PORT || "8000");

  return {
    plugins: [react()],
    envDir: "..",
    define: {
      __SERVER_PORT__: JSON.stringify(serverPort)
    },
    base: "/",
    build: {
      outDir: "../dist",
      emptyOutDir: true,
      assetsDir: "client-assets"
    },
    server: {
      port: clientPort
    }
  };
});
