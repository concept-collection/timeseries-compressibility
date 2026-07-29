import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative base so the built site works from any path (GitHub Pages project
// site, a subdirectory, or the filesystem).
export default defineConfig({
  base: './',
  plugins: [react()],
})
