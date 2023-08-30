import EventEmitter from "eventemitter3";

import {GazeDetector, iGazeDetectorResult} from "./GazeDetector";
import {save_gaze_calibration} from "./apiService";


export class Subject extends EventEmitter {

    // private intent: string = 'Greet';

    private isGazeDetectionActive: boolean = false;

    public get GazeDetectionActive(): boolean { return this.isGazeDetectionActive; }
    public get isGazeCalibrationActive(): boolean {
        return this.gazeDetector !== undefined && this.gazeDetector.Target !== undefined;
    };

    private targetTimeMs = 3000;

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

    public StartGazeDetectorCalibration(within: Screen | HTMLElement = window.screen) {
        const this_ = this;

        const left = (within === screen) ? 0 : (<HTMLElement>within).offsetLeft;
        const top = (within === screen) ? 0 : (<HTMLElement>within).offsetTop;
        const height = (within === screen) ? screen.height : (<HTMLElement>within).offsetHeight;
        const width = (within === screen) ? screen.width : (<HTMLElement>within).offsetWidth;

        const x_positions: number[] = [50, 200, 400, width / 2 - 50, width / 2 - 50, width / 2 - 50, width / 2,  width / 2 + 50, width - 400, width - 200, width - 55];
        const y_positions: number[] = [50, 200, 400, height / 2 - 50, height / 2, height / 2 + 50, height - 400, height - 200, height - 55];

        const center_x = width / 2;
        const center_y = height / 2 - 250;

        for (const offset of [-200, -100, 0, 0, 0, 100, 200]) {
            x_positions.push(center_x + offset);
            y_positions.push(center_y + offset);
        }
        const jitter = [-1,-2,-3,-4, 4, 3, 2, 1, 0 ]
        if (this.gazeDetector) this.gazeDetector.Target = {x: x_positions.randomElement(), y: y_positions.randomElement()}
        const new_pos = () => {
            if (this_.gazeDetector && this_.isGazeDetectionActive && this_.gazeDetector.Target) {
                this_.gazeDetector.Target = {x: left + x_positions.randomElement() + jitter.randomElement(), y: top + y_positions.randomElement() + jitter.randomElement()}
                // StopGazeRecognizerCalibration() can break the loop by setting target to undefined
                setTimeout(new_pos, this_.targetTimeMs);
            }
        }
        new_pos();
    }

    public StopGazeDetectorCalibration() {
        if (this.gazeDetector && this.isGazeDetectionActive)
            this.gazeDetector.Target = undefined;
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
            this.gazeDetector = new GazeDetector(<HTMLVideoElement>document.getElementById("vidCap"), true);

        const vidcap_overlay = <HTMLDivElement>document.getElementById('vidCapOverlay');

        this.gazeDetector.on('GazeDetectionComplete', (features: iGazeDetectorResult) => {
                // @ts-ignore
                vidcap_overlay.innerText = `${this.gazeDetector.FrameRate.toFixed(0)} FPS`

              }
        );

        await this.gazeDetector.init();

    }

}