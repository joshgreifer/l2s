// src/runtime/WebOnnxAdapter.ts
// If you load ORT via <script src=".../ort.min.js">, use: const ort = (window as any).ort;
import * as ort from 'onnxruntime-web';

export class WebOnnxAdapter {
  private session?: ort.InferenceSession;
  ready = false;

  async init(modelUrl: string) {
    // Keep it simple: single-threaded, no proxy workers. (Threaded is fine too, but this removes that variable.)
    const env = (window as any).ort?.env ?? ort.env;
    env.logLevel = 'verbose';
    env.wasm.simd = true;
    env.wasm.numThreads = 4;
    env.wasm.proxy = true;

    this.session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      extra: { 'session.use_ort_model_bytes_directly': '1' },
    });

    // --- SMOKE TEST ---
    const inputName = this.session.inputNames[0];
    const dummy = new ort.Tensor('float32', new Float32Array(1 * 478 * 3), [1, 478, 3]);
      const out = await this.session.run({ [inputName]: dummy });        // <-- plain object, not Map
      const y = out[this.session.outputNames[0]] as ort.Tensor;          // <-- tensor, not Map/Seq
      console.log('ORT smoke:', { inputName, outName: this.session.outputNames[0], dims: y.dims, dataLen: (y.data as Float32Array).length });
      this.ready = true;
    }

  async predict(flatLandmarks: Float32Array): Promise<[number, number]> {
    if (!this.session) throw new Error('ORT session not initialized');
    if (flatLandmarks.length !== 478 * 3) throw new Error(`Expected 1434 floats, got ${flatLandmarks.length}`);

    const inputName = this.session.inputNames[0];
    const x = new ort.Tensor('float32', flatLandmarks, [1, 478, 3]);
    const out = await this.session.run({ [inputName]: x });            // <-- plain object
    const y = out[this.session.outputNames[0]] as ort.Tensor;          // <-- tensor
    const v = y.data as Float32Array;                                  // <-- use .data, not .getValue()
    return [v[0], v[1]];
  }
}

export const webOnnx = new WebOnnxAdapter();
