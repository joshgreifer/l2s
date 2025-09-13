import EventEmitter from "eventemitter3";

import { ContinuousTrainer } from "../ContinuousTrainer";
import { post_data } from "../apiService";
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

export class Trainer extends EventEmitter implements IGazeTrainer {
    private continuousTrainer: ContinuousTrainer | undefined = undefined;
    private trainingPromise: Promise<void> | undefined = undefined;

    private sendInFlight = false;
    private batchBuffer: BatchItem[] = [];
    private readonly MAX_BACKLOG_ITEMS = 30; // ~1s of data @30fps

    public get isTraining(): boolean {
        return this.trainingPromise !== undefined;
    }

    startTraining() {
        if (!this.continuousTrainer) {
            this.continuousTrainer = new ContinuousTrainer();
            this.continuousTrainer.on('data', () => {});
            this.trainingPromise = this.continuousTrainer.Start();
        }
    }

    async stopTraining() {
        if (this.continuousTrainer) {
            this.continuousTrainer.Stop();
            if (this.trainingPromise)
                await this.trainingPromise;
            this.continuousTrainer = undefined;
            this.trainingPromise = undefined;
        }
    }

    addSample(item: BatchItem) {
        this.batchBuffer.push(item);
        if (this.batchBuffer.length > this.MAX_BACKLOG_ITEMS) {
            this.batchBuffer.splice(0, this.batchBuffer.length - this.MAX_BACKLOG_ITEMS);
        }
        if (!this.sendInFlight) void this.sendBatchNow();
    }

    private async sendBatchNow() {
        if (this.sendInFlight) return;
        const batch = this.batchBuffer.splice(0, this.batchBuffer.length);
        if (batch.length === 0) return;
        this.sendInFlight = true;
        try {
            const features = await post_data(batch);
            if (features) {
                this.emit('prediction', features);
            }
            this.batchBuffer = batch.concat(this.batchBuffer).slice(-this.MAX_BACKLOG_ITEMS);
        } finally {
            this.sendInFlight = false;
            if (this.batchBuffer.length) void this.sendBatchNow();
        }
    }
}

