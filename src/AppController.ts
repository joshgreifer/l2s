import EventEmitter from "eventemitter3";
import {GazeDetector, iGazeDetectorResult} from "./GazeDetector";
import {save_gaze_model} from "./apiService";

export class AppController extends EventEmitter {
    private isGazeDetectionActive: boolean = false;
    public get GazeDetectionActive(): boolean { return this.isGazeDetectionActive; }
    public get isDataAcquisitionActive(): boolean {
        return this.gazeDetector !== undefined && this.gazeDetector.TargetPos !== undefined;
    }
    public get isTrainingActive(): boolean {
        return this.gazeDetector !== undefined && this.gazeDetector.isTraining;
    }

    private targetTimeMs = 5000;
    public set TargetTimeMs(t: number) { this.targetTimeMs = t; }

    public gazeDetector?: GazeDetector;
    private notificationDiv: HTMLDivElement;

    constructor() {
        super();
        this.notificationDiv = <HTMLDivElement>document.querySelector('.notification');
        window.addEventListener('keyup', this.handleKeyboardShortcuts);
    }

    private Notify(message: string): void {
        this.notificationDiv.innerText = message;
        this.notificationDiv.style.opacity = "1";
        window.setTimeout(() => {
            this.notificationDiv.style.opacity = "0";
        }, 5000);
    }

    private handleKeyboardShortcuts = async (evt: KeyboardEvent) => {
        if (evt.key === 's') {
            this.Notify("Saving model.");
            const success = await this.SaveGazeDetectorModel();
            this.Notify(success ? "Saved model." : "Failed to save model.");
        } else if (evt.key === ' ') {
            if (this.isDataAcquisitionActive)
                await this.StopDataAcquisition();
            else
                await this.StartDataAcquisition();
            this.Notify(this.isDataAcquisitionActive ? "Data acquisition started." : "Data acquisition stopped.");
        } else if (evt.key === 'c') {
            if (this.isTrainingActive)
                await this.StopTraining();
            else
                await this.StartTraining();
            this.Notify(this.isTrainingActive ? "Calibration started." : "Calibration stopped.");
        }
    };

    public async Run() {
        await this.StartGazeDetection();
    }

    public async StopGazeDetection() {
        if (this.gazeDetector) {
            await this.gazeDetector.term();
            this.gazeDetector = undefined;
        }
        this.isGazeDetectionActive = false;
    }

    public async StartDataAcquisition() {
        const x_positions: number[] = [0, 0, 0, screen.width / 2 - 50, screen.width / 2 - 50, screen.width / 2 - 50, screen.width / 2, screen.width / 2 + 50, screen.width, screen.width, screen.width];
        const y_positions: number[] = [0, 0, 0, 0 / 2 - 50, screen.height / 2, screen.height / 2 + 50, screen.height, screen.height, screen.height];
        const center_x = screen.width / 2;
        const center_y = screen.height / 2 - 250;
        for (const offset of [-200, -100, 0, 0, 0, 100, 200]) {
            x_positions.push(center_x + offset);
            y_positions.push(center_y + offset);
        }
        const jitter = [-1, -2, -3, -4, 4, 3, 2, 1, 0];

        if (this.gazeDetector)
            this.gazeDetector.TargetPos = { x: 0, y: 0 };

        const new_pos = () => {
            if (this.gazeDetector && this.isGazeDetectionActive && this.gazeDetector.TargetPos) {
                this.gazeDetector.TargetPos = {
                    x: x_positions.randomElement() + jitter.randomElement(),
                    y: y_positions.randomElement() + jitter.randomElement()
                };
                setTimeout(new_pos, this.targetTimeMs);
            }
        };
        new_pos();
    }

    public async StopDataAcquisition() {
        if (this.gazeDetector && this.isGazeDetectionActive)
            this.gazeDetector.TargetPos = undefined;
    }

    public async SaveGazeDetectorModel(): Promise<boolean> {
        return await save_gaze_model();
    }

    public async StartTraining() {
        if (this.gazeDetector && !this.gazeDetector.isTraining)
            await this.gazeDetector.startTraining();
    }

    public async StopTraining() {
        if (this.gazeDetector && this.gazeDetector.isTraining)
            await this.gazeDetector.stopTraining();
    }

    public async StartGazeDetection() {
        if (this.isGazeDetectionActive)
            return;
        this.isGazeDetectionActive = true;
        if (!this.gazeDetector)
            this.gazeDetector = new GazeDetector(<HTMLVideoElement>document.querySelector("#vidCap"),
                <HTMLDivElement>document.querySelector(".landmark_selector"));

        const vidcap_overlay = <HTMLDivElement>document.getElementById('vidCapOverlay');

        this.gazeDetector.on('GazeDetectionComplete', (features: iGazeDetectorResult) => {
                // @ts-ignore
                vidcap_overlay.innerText = `${this.gazeDetector!.FrameRate.toFixed(0)} FPS`;
            }
        );

        await this.gazeDetector.init();
    }
}

