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
  const base = '/models/ort_artifacts/' + prefix;
  const [train, evalModel, optimizer, init] = await Promise.all([
    fetch(base + 'training_model.onnx').then(r => r.arrayBuffer()),
    fetch(base + 'eval_model.onnx').then(r => r.arrayBuffer()),
    fetch(base + 'optimizer.onnx').then(r => r.arrayBuffer()),
    fetch(base + 'checkpoint.onnx').then(r => r.arrayBuffer()),
  ]);
  return { train, eval: evalModel, optimizer, init };
}

export class TrainableOnnxAdapter {
  private pcaTrainer?: ort.TrainingSession;
  private mlpTrainer?: ort.TrainingSession;
  private pcaFallback?: Float32Array;

  async init() {
    const env = (typeof window !== 'undefined' ? (window as any).ort?.env : undefined) ?? ort.env;
    env.wasm.wasmPaths = '/ort/';
    env.wasm.simd = true;
    env.wasm.numThreads = 4;
    env.wasm.proxy = false;

    const pcaArts = await fetchArtifacts('pca_').catch(() => null);
    if (pcaArts) {
      try {
        this.pcaTrainer = await ort.TrainingSession.create(pcaArts);
      } catch {
        this.pcaTrainer = undefined;
      }
    }

    const mlpArts = await fetchArtifacts('gaze_').catch(() => null);
    if (mlpArts) {
      try {
        this.mlpTrainer = await ort.TrainingSession.create(mlpArts);
      } catch {
        this.mlpTrainer = undefined;
      }
    }
  }

  // Train PCA using batched landmark inputs. The caller may provide the entire
  // dataset or a concatenation of mini-batches. Gradients are accumulated until
  // after all batches have been processed, when the optimizer step and gradient
  // reset are invoked.
  async trainPca(
    landmarks: Float32Array,
    target: Float32Array,
    batchSize: number,
  ) {
    if (!this.pcaTrainer) {
      this.pcaFallback = target.slice();
      return;
    }

    const sampleCount = landmarks.length / (478 * 3);
    if (sampleCount !== target.length / 32) {
      throw new Error('landmark and target sample counts differ');
    }

    for (let i = 0; i < sampleCount; i += batchSize) {
      const end = Math.min(i + batchSize, sampleCount);
      const lmSlice = landmarks.subarray(i * 478 * 3, end * 478 * 3);
      const tgtSlice = target.subarray(i * 32, end * 32);
      const input = new ort.Tensor('float32', lmSlice, [end - i, 478, 3]);
      const label = new ort.Tensor('float32', tgtSlice, [end - i, 32]);
      await this.pcaTrainer.trainStep([input, label]);
    }

    await this.pcaTrainer.optimizerStep();
    await this.pcaTrainer.lazyResetGrad();
  }

  // Run the PCA model in eval mode to obtain transformed features for a batch of
  // landmarks. The output can be compared with sklearn's PCA.fit_transform.
  async transformPca(
    landmarks: Float32Array,
    batchSize: number,
  ): Promise<Float32Array> {
    if (!this.pcaTrainer) {
      if (!this.pcaFallback) throw new Error('PCA trainer not initialized');
      return this.pcaFallback;
    }
    const input = new ort.Tensor('float32', landmarks, [batchSize, 478, 3]);
    // @ts-ignore
    const [output] = await this.pcaTrainer.evalStep([input]);
    return output.data as Float32Array;
  }

  async trainMlp(features: Float32Array, target: Float32Array) {
    if (!this.mlpTrainer) throw new Error('MLP trainer not initialized');
    const input = new ort.Tensor('float32', features, [1, 32]);
    const label = new ort.Tensor('float32', target, [1, 2]);
    await this.mlpTrainer.trainStep([input, label]);
    await this.mlpTrainer.optimizerStep();
    await this.mlpTrainer.lazyResetGrad();
  }

  // Train the MLP model using mini-batches over multiple epochs. Returns an
  // array of average losses, one per epoch, to mimic the progress reporting of
  // the Python reference implementation.
  async trainMlpBatch(
    features: Float32Array,
    target: Float32Array,
    batchSize: number,
    epochs: number,
  ): Promise<number[]> {
    if (!this.mlpTrainer) throw new Error('MLP trainer not initialized');

    const sampleCount = features.length / 32;
    if (sampleCount !== target.length / 2) {
      throw new Error('feature and target sample counts differ');
    }

    const epochLosses: number[] = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let batches = 0;

      for (let i = 0; i < sampleCount; i += batchSize) {
        const end = Math.min(i + batchSize, sampleCount);
        const featSlice = features.subarray(i * 32, end * 32);
        const tgtSlice = target.subarray(i * 2, end * 2);

        const input = new ort.Tensor('float32', featSlice, [end - i, 32]);
        const label = new ort.Tensor('float32', tgtSlice, [end - i, 2]);

        const [loss] = await this.mlpTrainer.trainStep([input, label]);
        totalLoss += (loss.data as Float32Array)[0];
        await this.mlpTrainer.optimizerStep();
        await this.mlpTrainer.lazyResetGrad();
        batches++;
      }

      epochLosses.push(batches ? totalLoss / batches : 0);
      // Placeholder for learning-rate scheduler support if available in ORT.
    }

    return epochLosses;
  }
}

export const trainableOnnx = new TrainableOnnxAdapter();
