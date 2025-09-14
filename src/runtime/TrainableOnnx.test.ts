import { expect, test, vi } from 'vitest';

vi.mock('onnxruntime-web', () => ({
  Tensor: class {
    constructor(public type: string, public data: Float32Array, public dims: number[]) {}
  }
}));

import { TrainableOnnxAdapter } from './TrainableOnnx';

test('TrainableOnnxAdapter exposes init and supports batched PCA training', async () => {
  const t = new TrainableOnnxAdapter();
  expect(typeof t.init).toBe('function');

  const trainStep = vi.fn();
  const optimizerStep = vi.fn();
  const lazyResetGrad = vi.fn();
  const evalStep = vi.fn(async () => [{ data: new Float32Array(32) }]);

  // inject fake trainer so we can call trainPca without loading real artifacts
  (t as any).pcaTrainer = { trainStep, optimizerStep, lazyResetGrad, evalStep };

  const landmarks = new Float32Array(2 * 478 * 3); // two samples
  const targets = new Float32Array(2 * 32);

  await t.trainPca(landmarks, targets, 1);
  expect(trainStep).toHaveBeenCalledTimes(2);
  expect(optimizerStep).toHaveBeenCalledTimes(1);
  expect(lazyResetGrad).toHaveBeenCalledTimes(1);

  const out = await t.transformPca(landmarks.subarray(0, 478 * 3), 1);
  expect(out).toBeInstanceOf(Float32Array);
});
