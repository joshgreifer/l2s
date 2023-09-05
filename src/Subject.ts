import EventEmitter from "eventemitter3";

import {GazeDetector, iGazeDetectorResult} from "./GazeDetector";
import {save_gaze_calibration} from "./apiService";


export class Subject extends EventEmitter {

    // private intent: string = 'Greet';

    private isGazeDetectionActive: boolean = false;

    public get GazeDetectionActive(): boolean { return this.isGazeDetectionActive; }
    public get isGazeCalibrationActive(): boolean {
        return this.gazeDetector !== undefined && this.gazeDetector.TargetPos !== undefined;
    };

    private targetTimeMs = 5000;

    public set TargetTimeMs(t:number) { this.targetTimeMs = t;
    }

    public gazeDetector?: GazeDetector;

    public async StopGazeDetection() {
        if (this.gazeDetector) {
            await this.gazeDetector.term();
            this.gazeDetector = undefined;
        }
        this.isGazeDetectionActive = false;
    }

    public StartGazeDetectorCalibration() {

        const this_ = this;

        const x_positions: number[] = [0, 0, 0, screen.width / 2 - 50, screen.width / 2 - 50, screen.width / 2 - 50, screen.width / 2,  screen.width / 2 + 50, screen.width , screen.width, screen.width];
        const y_positions: number[] = [0, 0, 0, 0 / 2 - 50, screen.height / 2, screen.height / 2 + 50, screen.height, screen.height, screen.height ];

        const center_x = screen.width / 2;
        const center_y = screen.height / 2 - 250;

        for (const offset of [-200, -100, 0, 0, 0, 100, 200]) {
            x_positions.push(center_x + offset);
            y_positions.push(center_y + offset);
        }
        const jitter = [-1,-2,-3,-4, 4, 3, 2, 1, 0 ]

       if (this.gazeDetector)
            this.gazeDetector.TargetPos = {x: 0, y: 0}

        const new_pos = () => {
            if (this.gazeDetector && this_.isGazeDetectionActive && this.gazeDetector.TargetPos) {
                this.gazeDetector.TargetPos = {x: x_positions.randomElement() + jitter.randomElement(), y: y_positions.randomElement() + jitter.randomElement()}
                // StopGazeRecognizerCalibration() can break the loop by setting target to undefined
                setTimeout(new_pos, this_.targetTimeMs);
            }
        }
        new_pos();
    }

    public StopGazeDetectorCalibration() {
        if (this.gazeDetector && this.isGazeDetectionActive)
            this.gazeDetector.TargetPos = undefined;
    }
    public async SaveGazeDetectorCalibration(): Promise<boolean> {
        return await save_gaze_calibration();
    }

    public async ToggleGazeDetectorTraining() {
        if (this.gazeDetector)
            await this.gazeDetector.toggleGazeDetectorTraining();
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
                vidcap_overlay.innerText = `${this.gazeDetector.FrameRate.toFixed(0)} FPS`
              }
        );

        await this.gazeDetector.init();

    }

}