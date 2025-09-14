import { expect, test } from 'vitest';
import { TrainableOnnxAdapter } from './TrainableOnnx';

test('TrainableOnnxAdapter exposes init', () => {
  const t = new TrainableOnnxAdapter();
  expect(typeof t.init).toBe('function');
});
