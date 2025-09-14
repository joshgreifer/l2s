import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import path from 'path';

export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: path.resolve(__dirname, 'index.html'),
      output: {
        entryFileNames: 'js/[name].js',
        chunkFileNames: 'js/[name].js',
        assetFileNames: ({ name }) => {
          if (/\.css$/.test(name ?? '')) {
            return 'css/[name].[ext]';
          }
          if (/\.(png|ico|svg|jpg|jpeg|gif)$/.test(name ?? '')) {
            return 'icons/[name].[ext]';
          }
          if (/\.wasm$/.test(name ?? '')) {
            return 'ort/[name].[ext]';
          }
          return 'assets/[name].[ext]';
        }
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src'),
      '~': path.resolve(__dirname, 'src')
    },
    extensions: ['.js', '.json', '.ts']
  },
  define: {
    APP: {
      TITLE: JSON.stringify('l2s'),
      AUTHOR: JSON.stringify('Error 404 <joshgreifer@gmail.com>')
    }
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*',
          dest: 'ort'
        }
      ]
    })
  ]
});
