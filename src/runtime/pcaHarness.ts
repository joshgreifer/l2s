// src/runtime/pcaHarness.ts
// Generate a synthetic dataset, train the PCA ONNX model, and compare
// explained variance ratios against sklearn's PCA implementation.

import { spawnSync } from 'child_process';
import { trainableOnnx } from './TrainableOnnx';

function varianceRatios(
  features: Float32Array,
  featureSize: number,
  totalVariance?: number,
): number[] {
  const sampleCount = features.length / featureSize;
  const means = new Float64Array(featureSize);
  for (let i = 0; i < sampleCount; i++) {
    for (let j = 0; j < featureSize; j++) {
      means[j] += features[i * featureSize + j];
    }
  }
  for (let j = 0; j < featureSize; j++) {
    means[j] /= sampleCount;
  }

  const variances = new Float64Array(featureSize);
  for (let i = 0; i < sampleCount; i++) {
    for (let j = 0; j < featureSize; j++) {
      const d = features[i * featureSize + j] - means[j];
      variances[j] += d * d;
    }
  }
  for (let j = 0; j < featureSize; j++) {
    variances[j] /= (sampleCount - 1);
  }
  const denom = totalVariance ?? variances.reduce((a, b) => a + b, 0);
  return Array.from(variances, v => v / denom);
}

function fakeLandmarks(sampleCount: number): Float32Array {
  const arr = new Float32Array(sampleCount * 478 * 3);
  for (let i = 0; i < arr.length; i++) arr[i] = Math.random();
  return arr;
}

function runSklearn(landmarks: Float32Array) {
  const proc = spawnSync('python', ['scripts/sklearn_pca.py'], {
    input: JSON.stringify({ landmarks: Array.from(landmarks) }),
    encoding: 'utf-8',
  });
  if (proc.status !== 0) {
    throw new Error(proc.stderr || 'failed to run sklearn_pca.py');
  }
  const out = JSON.parse(proc.stdout);
  return {
    features: new Float32Array(out.features),
    totalVar: out.total_var as number,
  };
}

export async function compareWithSklearn(sampleCount = 64) {
  const landmarks = fakeLandmarks(sampleCount);
  const { features, totalVar } = runSklearn(landmarks);

  await trainableOnnx.init();
  await trainableOnnx.trainPca(landmarks, features, sampleCount);
  const transformed = await trainableOnnx.transformPca(landmarks, sampleCount);
  const skVariance = varianceRatios(features, 32, totalVar);
  const onnxVariance = varianceRatios(transformed, 32, totalVar);

  return { skVariance, onnxVariance };
}

// @ts-ignore
if (import.meta.main) {
  compareWithSklearn().then(({ skVariance, onnxVariance }) => {
    console.log('sklearn', skVariance);
    console.log('onnx', onnxVariance);
  }).catch(err => {
    console.error(err);
    process.exit(1);
  });
}
