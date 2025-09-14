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
        this.dataset.add(item);
        void post_data([item]).then((features) => {
            if (features) {
                this.emit('prediction', features);
            }
        });
    }

    private async runTrainingLoop() {
        while (this.trainingActive) {
            const batch = this.dataset.toArray();
            if (batch.length) {
                const losses = await train(batch, 1, "train");
                this.emit('loss', losses);
                this.epochCounter++;
                this.emit('epoch', this.epochCounter);
            }
            await new Promise((r) => setTimeout(r, 0));
        }
    }
}

