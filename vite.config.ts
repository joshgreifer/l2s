import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import path from 'path';

export default defineConfig({
  root: 'src',
  build: {
    outDir: '../dist/static',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'src/index.html')
      },
      output: {
        entryFileNames: 'index.js'
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname),
      '~': path.resolve(__dirname)
    },
    extensions: ['.js', '.json', '.ts']
  },
  define: {
    APP: {
      TITLE: JSON.stringify("l2s"),
      AUTHOR: JSON.stringify("Error 404 <joshgreifer@gmail.com>")
    }
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: '../pkg/**/*',
          dest: '../pkg'
        },
        {
          src: '../app.py',
          dest: '..'
        },
        {
          src: 'styles/**/*',
          dest: 'styles'
        },
        {
          src: '../static/**/*',
          dest: '.' // copies all files from static to dist/static/
        },
        {
          // Ensure ONNX Runtime's WASM binaries are available under /assets/
          // so that the application can load them via absolute URLs.
          src: '../node_modules/onnxruntime-web/dist/*',
          dest: 'ort'
        }
      ]
    })
  ]
});
