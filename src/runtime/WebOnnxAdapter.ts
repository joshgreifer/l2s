// src/runtime/WebOnnxAdapter.ts
// If you load ORT via <script src=".../ort.min.js">, use: const ort = (window as any).ort;
import * as ort from 'onnxruntime-web';
import { PixelCoord } from '../util/Coords';

export class WebOnnxAdapter {
  private pcaSession?: ort.InferenceSession;
  private mlpSession?: ort.InferenceSession;
  ready = false;

  async init() {
    // Configure ORT's environment. Keep it single-threaded and avoid proxy
    // workers. When proxy workers were enabled, the main bundle was executed
    // in a WebWorker that lacks a DOM, which surfaced as "document is not
    // defined" errors in the console.
    const env = (window as any).ort?.env ?? ort.env;
    env.logLevel = 'verbose';
    env.wasm.simd = true;
    env.wasm.numThreads = 4;
    env.wasm.proxy = false;
    // ORT expects its wasm assets relative to this path. Vite copies the
    // binaries into /ort at the public root, so reference that location
    // directly.
    env.wasm.wasmPaths = '/ort';

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
      extra: { 'session.use_ort_model_bytes_directly': '1' },
    };

    this.pcaSession = await ort.InferenceSession.create('/models/pca.onnx', sessionOptions);

    this.mlpSession = await ort.InferenceSession.create('/models/gaze_mlp.onnx', sessionOptions);

    // --- SMOKE TEST --- end-to-end
    const dummy = new ort.Tensor('float32', new Float32Array(478 * 3), [1, 478, 3]);
    const pcaOutMap = await this.pcaSession.run({ [this.pcaSession.inputNames[0]]: dummy });
    const pcaOut = pcaOutMap[this.pcaSession.outputNames[0]] as ort.Tensor;
    const mlpInput = new ort.Tensor('float32', pcaOut.data as Float32Array, [1, 32]);
    const mlpOutMap = await this.mlpSession.run({ [this.mlpSession.inputNames[0]]: mlpInput });
    const mlpOut = mlpOutMap[this.mlpSession.outputNames[0]] as ort.Tensor;
    console.log('ORT smoke:', {
      pcaOutDims: pcaOut.dims,
      mlpOutDims: mlpOut.dims,
      dataLen: (mlpOut.data as Float32Array).length,
    });
    this.ready = true;
  }

  async predict(landmarks: PixelCoord[]): Promise<[number, number]> {
    if (!this.pcaSession || !this.mlpSession) throw new Error('ORT sessions not initialized');
    if (landmarks.length !== 478) throw new Error(`Expected 478 landmarks, got ${landmarks.length}`);

    const flat = new Float32Array(478 * 3);
    for (let i = 0; i < 478; i++) {
      const lm = landmarks[i];
      flat[i * 3] = lm[0] ?? 0;
      flat[i * 3 + 1] = lm[1] ?? 0;
      flat[i * 3 + 2] = lm[2] ?? 0;
    }

    const pcaInput = new ort.Tensor('float32', flat, [1, 478, 3]);
    const pcaOutMap = await this.pcaSession.run({ [this.pcaSession.inputNames[0]]: pcaInput });
    const pcaOut = pcaOutMap[this.pcaSession.outputNames[0]] as ort.Tensor;

    const mlpInput = new ort.Tensor('float32', pcaOut.data as Float32Array, [1, 32]);
    const mlpOutMap = await this.mlpSession.run({ [this.mlpSession.inputNames[0]]: mlpInput });
    const mlpOut = mlpOutMap[this.mlpSession.outputNames[0]] as ort.Tensor;
    const v = mlpOut.data as Float32Array;
    return [v[0], v[1]];
  }
}

export const webOnnx = new WebOnnxAdapter();
