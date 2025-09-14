import type { IGazeTrainer, iGazeDetectorTrainResult } from "../training/Trainer";

class TrainingPage {
    private progressBar: HTMLProgressElement;
    private hLossSpan: HTMLSpanElement;
    private vLossSpan: HTMLSpanElement;
    private lossSpan: HTMLSpanElement;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private lossHistory: number[] = [];
    private trainer?: IGazeTrainer;

    constructor() {
        this.progressBar = document.getElementById('trainingProgress') as HTMLProgressElement;
        this.hLossSpan = document.getElementById('hLoss') as HTMLSpanElement;
        this.vLossSpan = document.getElementById('vLoss') as HTMLSpanElement;
        this.lossSpan = document.getElementById('loss') as HTMLSpanElement;
        this.canvas = document.getElementById('lossPlot') as HTMLCanvasElement;
        const ctx = this.canvas.getContext('2d');
        if (!ctx) throw new Error('No 2D context for loss plot');
        this.ctx = ctx;
    }

    public setTrainer(t?: IGazeTrainer) {
        if (this.trainer) {
            this.trainer.off('epoch-start', this.onEpochStart);
            this.trainer.off('progress', this.onProgress);
            this.trainer.off('loss', this.onLoss);
        }
        this.trainer = t;
        if (this.trainer) {
            this.trainer.on('epoch-start', this.onEpochStart);
            this.trainer.on('progress', this.onProgress);
            this.trainer.on('loss', this.onLoss);
        }
    }

    private onEpochStart = ({ total }: { total: number }) => {
        this.progressBar.max = total;
        this.progressBar.value = 0;
    };

    private onProgress = ({ current, losses }: { current: number; losses: iGazeDetectorTrainResult }) => {
        this.progressBar.value = current;
        this.hLossSpan.innerText = losses.h_loss.toFixed(3);
        this.vLossSpan.innerText = losses.v_loss.toFixed(3);
        this.lossSpan.innerText = losses.loss.toFixed(3);
    };

    private onLoss = (losses: iGazeDetectorTrainResult) => {
        this.lossHistory.push(losses.loss);
        this.drawPlot();
    };

    private drawPlot() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);
        if (this.lossHistory.length < 2) return;
        const maxLoss = Math.max(...this.lossHistory);
        ctx.beginPath();
        for (let i = 0; i < this.lossHistory.length; i++) {
            const x = (i / (this.lossHistory.length - 1)) * w;
            const y = h - (this.lossHistory[i] / maxLoss) * h;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = '#03ff10';
        ctx.stroke();
    }
}

export const trainingPage: TrainingPage | undefined =
    typeof document !== 'undefined' ? new TrainingPage() : undefined;
