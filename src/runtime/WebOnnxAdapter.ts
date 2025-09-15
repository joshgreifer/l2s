// src/runtime/WebOnnxAdapter.ts
// If you load ORT via <script src=".../ort.min.js">, use: const ort = (window as any).ort;
import * as ort from 'onnxruntime-web';
import { PixelCoord } from '../util/Coords';
/**
 * Thin wrapper around onnxruntime-web that loads PCA and MLP models and
 * provides batched prediction and model export utilities for gaze inference.
 */

export class WebOnnxAdapter {
  private pcaSession?: ort.InferenceSession;
  private mlpSession?: ort.InferenceSession;
  ready = false;
  private _queue: Promise<unknown> = Promise.resolve();

  async init(mlpBytes?: ArrayBuffer) {
    // Configure ORT's environment. Prefer multi-threading only when the page
    // is cross-origin isolated (the only scenario where SharedArrayBuffer is
    // available) and always avoid proxy workers. When proxy workers were
    // enabled, the main bundle was executed in a WebWorker that lacks a DOM,
    // which surfaced as "document is not defined" errors in the console.
    const env = (window as any).ort?.env ?? ort.env;
    env.logLevel = 'warning';
    env.wasm.simd = true;

    const supportsMultiThreading =
      typeof window !== 'undefined' &&
      window.crossOriginIsolated === true &&
      typeof SharedArrayBuffer !== 'undefined';
    const hardwareConcurrency =
      typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
        ? navigator.hardwareConcurrency
        : 1;

    env.wasm.numThreads = supportsMultiThreading ? Math.min(4, Math.max(1, hardwareConcurrency)) : 1;
    env.wasm.proxy = false;
    // ORT expects its wasm assets relative to this path. Vite copies the
    // binaries into /ort at the public root, so reference that location
    // directly.
    env.wasm.wasmPaths = '/ort/';

    const createSessionOptions = (): ort.InferenceSession.SessionOptions => ({
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
      extra: { 'session.use_ort_model_bytes_directly': '1' },
    });

    this.pcaSession = await ort.InferenceSession.create('/models/pca.onnx', createSessionOptions());

    try {
      if (mlpBytes) {
        this.mlpSession = await ort.InferenceSession.create(mlpBytes, createSessionOptions());
      } else {
        throw new Error('no saved model');
      }
    } catch {
      this.mlpSession = await ort.InferenceSession.create('/models/gaze_mlp.onnx', createSessionOptions());
    }

    // --- SMOKE TEST --- ensure batching works end-to-end
    // const dummy = new ort.Tensor('float32', new Float32Array(2 * 478 * 3), [2, 478, 3]);
    // const pcaOutMap = await this.pcaSession.run({ [this.pcaSession.inputNames[0]]: dummy });
    // const pcaOut = pcaOutMap[this.pcaSession.outputNames[0]] as ort.Tensor;
    // const mlpInput = new ort.Tensor('float32', pcaOut.data as Float32Array, [2, 32]);
    // const mlpOutMap = await this.mlpSession.run({ [this.mlpSession.inputNames[0]]: mlpInput });
    // const mlpOut = mlpOutMap[this.mlpSession.outputNames[0]] as ort.Tensor;
    // console.log('ORT smoke:', {
    //   pcaOutDims: pcaOut.dims,
    //   mlpOutDims: mlpOut.dims,
    //   dataLen: (mlpOut.data as Float32Array).length,
    // });
    this.ready = true;
  }

  private async _predict(batch: PixelCoord[][]): Promise<[number, number][]> {
    if (!this.pcaSession || !this.mlpSession) throw new Error('ORT sessions not initialized');
    if (batch.length === 0) return [];
    const B = batch.length;
    const flat = new Float32Array(B * 478 * 3);
    for (let b = 0; b < B; b++) {
      const landmarks = batch[b];
      if (landmarks.length !== 478) throw new Error(`Expected 478 landmarks, got ${landmarks.length}`);
      for (let i = 0; i < 478; i++) {
        const lm = landmarks[i];
        const base = b * 478 * 3 + i * 3;
        flat[base] = lm[0] ?? 0;
        flat[base + 1] = lm[1] ?? 0;
        flat[base + 2] = lm[2] ?? 0;
      }
    }

    const pcaInput = new ort.Tensor('float32', flat, [B, 478, 3]);
    const pcaOutMap = await this.pcaSession.run({ [this.pcaSession.inputNames[0]]: pcaInput });
    const pcaOut = pcaOutMap[this.pcaSession.outputNames[0]] as ort.Tensor;

    const mlpInput = new ort.Tensor('float32', pcaOut.data as Float32Array, [B, 32]);
    const mlpOutMap = await this.mlpSession.run({ [this.mlpSession.inputNames[0]]: mlpInput });
    const mlpOut = mlpOutMap[this.mlpSession.outputNames[0]] as ort.Tensor;
    const v = mlpOut.data as Float32Array;
    const out: [number, number][] = [];
    for (let b = 0; b < B; b++) {
      out.push([v[b * 2], v[b * 2 + 1]]);
    }
    return out;
  }

  async predict(batch: PixelCoord[][]): Promise<[number, number][]> {
    const run = this._queue.then(() => this._predict(batch));
    this._queue = run.then(() => {}, () => {});
    return run;
  }

  async exportMlpModel(): Promise<ArrayBuffer | null> {
    if (!this.mlpSession) return null;
    const anySession = this.mlpSession as any;
    if (typeof anySession.exportModel === 'function') {
      return await anySession.exportModel();
    }
    try {
      const res = await fetch('/models/gaze_mlp.onnx');
      return await res.arrayBuffer();
    } catch {
      return null;
    }
  }
}

export const webOnnx = new WebOnnxAdapter();
