import { expect, test } from 'vitest';

const run = process.env.RUN_PCA_COMPARE === '1';

(run ? test : test.skip)(
  'PCA training matches sklearn variance ratios',
  async () => {
    const { compareWithSklearn } = await import('./pcaHarness');
    const { skVariance, onnxVariance } = await compareWithSklearn(64);
    skVariance.forEach((v, i) => {
      expect(onnxVariance[i]).toBeCloseTo(v, 4);
    });
  },
);
