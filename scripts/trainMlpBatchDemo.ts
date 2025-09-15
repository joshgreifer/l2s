import { TrainableOnnxAdapter } from '../src/runtime/TrainableOnnx';

async function main() {
  const adapter = new TrainableOnnxAdapter();
  await adapter.init();

  // Example recorded features and targets. Each feature vector has 32 elements
  // and each target has 2 elements. Replace with real data as needed.
  const features = new Float32Array(Array.from({ length: 64 }, (_, i) => i / 100));
  const targets = new Float32Array([0, 0, 1, 1]);

  const losses = await adapter.trainMlpBatch(features, targets, 1, 3, 0.01);
  losses.forEach((loss, epoch) => {
    console.log(`Epoch ${epoch + 1}: loss=${loss}`);
  });
}

main().catch(err => console.error(err));
