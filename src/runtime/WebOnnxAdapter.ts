// src/runtime/WebOnnxAdapter.ts
// If you load ORT via <script src=".../ort.min.js">, use: const ort = (window as any).ort;
import type * as ortTypes from 'onnxruntime-web';
import { PixelCoord } from '../util/Coords';

type OrtModule = typeof import('onnxruntime-web');

type OrtConfig = {
  logLevel: 'warning';
  wasm: {
    simd: boolean;
    proxy: boolean;
    multiThread: boolean;
    numThreads: number;
    wasmPaths: string;
  };
};

type GlobalOrt = {
  env?: {
    wasm?: Record<string, unknown>;
    logLevel?: OrtConfig['logLevel'];
  };
};

const ensureGlobalOrt = (): GlobalOrt & { env: { wasm: Record<string, unknown> } } => {
  const globalScope = globalThis as typeof globalThis & { ort?: GlobalOrt };
  if (!globalScope.ort) {
    globalScope.ort = { env: { wasm: {} } };
  } else {
    const ortEnv = globalScope.ort.env ?? (globalScope.ort.env = { wasm: {} });
    if (!ortEnv.wasm) {
      ortEnv.wasm = {};
    }
  }
  return globalScope.ort as GlobalOrt & { env: { wasm: Record<string, unknown> } };
};

const buildOrtConfig = (): OrtConfig => {
  const supportsMultiThreading =
    typeof window !== 'undefined' &&
    window.crossOriginIsolated === true &&
    typeof SharedArrayBuffer !== 'undefined';

  const hardwareConcurrency =
    typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
      ? navigator.hardwareConcurrency
      : 1;

  return {
    logLevel: 'warning',
    wasm: {
      simd: true,
      proxy: false,
      multiThread: supportsMultiThreading,
      numThreads: supportsMultiThreading ? Math.min(4, Math.max(1, hardwareConcurrency)) : 1,
      wasmPaths: '/ort/',
    },
  };
};

const applyConfigToGlobalOrt = (config: OrtConfig) => {
  const ort = ensureGlobalOrt();
  ort.env.logLevel = config.logLevel;
  Object.assign(ort.env.wasm, config.wasm);
};

let ortModulePromise: Promise<OrtModule> | null = null;

const loadOrt = async (config: OrtConfig): Promise<OrtModule> => {
  applyConfigToGlobalOrt(config);
  if (!ortModulePromise) {
    ortModulePromise = import('onnxruntime-web');
  }
  const ort = await ortModulePromise;
  ort.env.logLevel = config.logLevel;
  Object.assign(ort.env.wasm, config.wasm);
  return ort;
};
/**
 * Thin wrapper around onnxruntime-web that loads PCA and MLP models and
 * provides batched prediction and model export utilities for gaze inference.
 */

export class WebOnnxAdapter {
  private ort?: OrtModule;
  private pcaSession?: ortTypes.InferenceSession;
  private mlpSession?: ortTypes.InferenceSession;
  ready = false;
  private _queue: Promise<unknown> = Promise.resolve();

  async init(mlpBytes?: ArrayBuffer) {
    // Configure ORT before importing so the runtime picks the correct WASM
    // variant. Multi-threading is only enabled when cross-origin isolation is
    // available; otherwise we explicitly fall back to the single-threaded
    // ort-wasm-simd build and avoid proxy workers that spawn WebWorkers.
    const config = buildOrtConfig();
    const ort = await loadOrt(config);
    this.ort = ort;

    const createSessionOptions = (): ortTypes.InferenceSession.SessionOptions => ({
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

    const ort = this.ort;
    if (!ort) throw new Error('ORT module not loaded');

    const pcaInput = new ort.Tensor('float32', flat, [B, 478, 3]);
    const pcaOutMap = await this.pcaSession.run({ [this.pcaSession.inputNames[0]]: pcaInput });
    const pcaOut = pcaOutMap[this.pcaSession.outputNames[0]] as ortTypes.Tensor;

    const mlpInput = new ort.Tensor('float32', pcaOut.data as Float32Array, [B, 32]);
    const mlpOutMap = await this.mlpSession.run({ [this.mlpSession.inputNames[0]]: mlpInput });
    const mlpOut = mlpOutMap[this.mlpSession.outputNames[0]] as ortTypes.Tensor;
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
