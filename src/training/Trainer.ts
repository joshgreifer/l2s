import EventEmitter from "eventemitter3";

import { post_data, train } from "../apiService";
import { Coord, PixelCoord } from "../util/Coords";

export interface iGazeDetectorTrainResult {
    h_loss: number;
    v_loss: number;
    loss: number;
}

export interface iGazeDetectorAddDataResult {
    data_index: number;
    gaze: Coord;
    losses: iGazeDetectorTrainResult;
}

export type BatchItem = {
    landmarks: PixelCoord[];   // [[x,y,z], ...]
    target: [number, number];   // screen-space target coordinates
};

export interface IGazeTrainer extends EventEmitter {
    startTraining(): void;
    stopTraining(): Promise<void>;
    addSample(sample: BatchItem): void;
    readonly isTraining: boolean;
    readonly epoch: number;
}

class RingBufferDataset<T> {
    private buf: T[] = [];
    constructor(private readonly capacity: number) {}

    add(item: T) {
        this.buf.push(item);
        if (this.buf.length > this.capacity) this.buf.shift();
    }

    toArray(): T[] {
        return this.buf.slice();
    }

    get length(): number {
        return this.buf.length;
    }

    get last(): T | undefined {
        return this.buf[this.buf.length - 1];
    }
}

export class Trainer extends EventEmitter implements IGazeTrainer {
    private readonly DATASET_CAPACITY = 2048; // maximum number of samples to retain
    private dataset = new RingBufferDataset<BatchItem>(this.DATASET_CAPACITY);
    private trainingActive = false;
    private trainingLoop?: Promise<void>;
    private epochCounter = 0;

    public get isTraining(): boolean {
        return this.trainingActive;
    }

    public get epoch(): number {
        return this.epochCounter;
    }

    startTraining() {
        if (this.trainingActive) return;
        this.trainingActive = true;
        this.epochCounter = 0;
        this.trainingLoop = this.runTrainingLoop();
    }

    async stopTraining() {
        this.trainingActive = false;
        if (this.trainingLoop) {
            await this.trainingLoop;
            this.trainingLoop = undefined;
        }
    }

    addSample(item: BatchItem) {
        // Ensure that only valid samples with a target are stored.
        if (!item.target) return;
        this.dataset.add(item);
        void post_data([item]).then((features) => {
            if (features) {
                this.emit('prediction', features);
            }
        });
    }

    private readonly BATCH_SIZE = 64;

    private async runTrainingLoop() {
        while (this.trainingActive) {
            const data = this.dataset.toArray();
            // Shuffle data to avoid training on ordered samples
            for (let i = data.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [data[i], data[j]] = [data[j], data[i]];
            }

            const totalBatches = Math.ceil(data.length / this.BATCH_SIZE);
            if (totalBatches) {
                this.emit('epoch-start', { total: totalBatches });
                let h_sum = 0;
                let v_sum = 0;
                let n = 0;
                for (let i = 0; i < totalBatches; i++) {
                    const batch = data.slice(i * this.BATCH_SIZE, (i + 1) * this.BATCH_SIZE);
                    const losses = await train(batch, 1, "train");
                    h_sum += losses.h_loss * batch.length;
                    v_sum += losses.v_loss * batch.length;
                    n += batch.length;
                    this.emit('progress', {
                        current: i + 1,
                        total: totalBatches,
                        losses: {
                            h_loss: h_sum / n,
                            v_loss: v_sum / n,
                            loss: (h_sum / n + v_sum / n) / 2,
                        },
                    });
                }
                const avg = { h_loss: h_sum / n, v_loss: v_sum / n, loss: (h_sum / n + v_sum / n) / 2 };
                this.emit('loss', avg);
                this.epochCounter++;
                this.emit('epoch', this.epochCounter);
            }
            await new Promise((r) => setTimeout(r, 0));
        }
    }
}

