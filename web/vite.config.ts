import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const studioOrigin = 'http://127.0.0.1:8765'
const developmentOrigin = 'http://127.0.0.1:5173'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: studioOrigin,
        changeOrigin: true,
        configure(proxy) {
          proxy.on('proxyReq', (forwarded, incoming) => {
            if (incoming.headers.origin === developmentOrigin)
              forwarded.setHeader('origin', studioOrigin)
          })
        },
      },
    },
  },
})
