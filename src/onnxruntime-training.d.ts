import type { Tensor } from 'onnxruntime-common';

export interface TrainingArtifacts {
  train: ArrayBuffer;
  eval: ArrayBuffer;
  optimizer: ArrayBuffer;
  init: ArrayBuffer;
}

declare module 'onnxruntime-web' {
  export { TrainingArtifacts };
  export class TrainingSession {
    static create(artifacts: TrainingArtifacts): Promise<TrainingSession>;
    trainStep(inputs: Tensor[]): Promise<void>;
    optimizerStep(): Promise<void>;
    lazyResetGrad(): Promise<void>;
  }
}
