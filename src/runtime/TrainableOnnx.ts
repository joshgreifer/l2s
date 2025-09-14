// src/runtime/TrainableOnnx.ts
// Minimal scaffolding to load and train ORT models for PCA and MLP.
import * as ort from 'onnxruntime-web';

interface ArtifactPaths {
  train: ArrayBuffer;
  eval: ArrayBuffer;
  optimizer: ArrayBuffer;
  init: ArrayBuffer;
}

async function fetchArtifacts(prefix: string): Promise<ArtifactPaths> {
  const base = `/ort/${prefix}`;
  const [train, evalModel, optimizer, init] = await Promise.all([
    fetch(`${base}training_model.onnx`).then(r => r.arrayBuffer()),
    fetch(`${base}eval_model.onnx`).then(r => r.arrayBuffer()),
    fetch(`${base}optimizer.onnx`).then(r => r.arrayBuffer()),
    fetch(`${base}checkpoint.onnx`).then(r => r.arrayBuffer()),
  ]);
  return { train, eval: evalModel, optimizer, init };
}

export class TrainableOnnxAdapter {
  private pcaTrainer?: ort.TrainingSession;
  private mlpTrainer?: ort.TrainingSession;

  async init() {
    const env = (window as any).ort?.env ?? ort.env;
    env.wasm.wasmPaths = '/ort/';
    env.wasm.simd = true;
    env.wasm.numThreads = 4;
    env.wasm.proxy = false;

    const [pcaArts, mlpArts] = await Promise.all([
      fetchArtifacts('pca_'),
      fetchArtifacts('gaze_'),
    ]);

    this.pcaTrainer = await ort.TrainingSession.create(pcaArts);
    this.mlpTrainer = await ort.TrainingSession.create(mlpArts);
  }

  async trainPca(landmarks: Float32Array, target: Float32Array) {
    if (!this.pcaTrainer) throw new Error('PCA trainer not initialized');
    const input = new ort.Tensor('float32', landmarks, [1, 478, 3]);
    const label = new ort.Tensor('float32', target, [1, 32]);
    await this.pcaTrainer.trainStep([input, label]);
    await this.pcaTrainer.optimizerStep();
    await this.pcaTrainer.lazyResetGrad();
  }

  async trainMlp(features: Float32Array, target: Float32Array) {
    if (!this.mlpTrainer) throw new Error('MLP trainer not initialized');
    const input = new ort.Tensor('float32', features, [1, 32]);
    const label = new ort.Tensor('float32', target, [1, 2]);
    await this.mlpTrainer.trainStep([input, label]);
    await this.mlpTrainer.optimizerStep();
    await this.mlpTrainer.lazyResetGrad();
  }
}

export const trainableOnnx = new TrainableOnnxAdapter();
