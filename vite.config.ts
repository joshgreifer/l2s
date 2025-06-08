import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import path from 'path';

export default defineConfig({
  root: 'static',
  build: {
    outDir: '../dist/static',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'static/index.html')
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
          src: '*',
          dest: '.'
        }
      ]
    })
  ]
});
