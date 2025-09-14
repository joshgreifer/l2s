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
    target: number[] | null;   // screen-space target (or omit)
};

export interface IGazeTrainer extends EventEmitter {
    startTraining(): void;
    stopTraining(): Promise<void>;
    addSample(sample: BatchItem): void;
    readonly isTraining: boolean;
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

    get last(): T | undefined {
        return this.buf[this.buf.length - 1];
    }
}

export class Trainer extends EventEmitter implements IGazeTrainer {
    private readonly DATASET_CAPACITY = 2048; // maximum number of samples to retain
    private dataset = new RingBufferDataset<BatchItem>(this.DATASET_CAPACITY);
    private trainingActive = false;
    private trainingLoop?: Promise<void>;

    public get isTraining(): boolean {
        return this.trainingActive;
    }

    startTraining() {
        if (this.trainingActive) return;
        this.trainingActive = true;
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
        this.dataset.add(item);
    }

    private async runTrainingLoop() {
        while (this.trainingActive) {
            const batch = this.dataset.toArray();
            const last = this.dataset.last;
            if (batch.length && last) {
                const losses = await train(batch, 1, "train");
                this.emit('loss', losses);
                const features = await post_data([last]);
                if (features) {
                    this.emit('prediction', features);
                }
            }
            await new Promise((r) => setTimeout(r, 0));
        }
    }
}

