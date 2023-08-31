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

    public StartGazeDetectorCalibration(within: Screen | HTMLElement = window.screen) {

        const this_ = this;

        const left = (within === screen) ? 0 : (<HTMLElement>within).offsetLeft;
        const top = (within === screen) ? 0 : (<HTMLElement>within).offsetTop;
        const height = (within === screen) ? screen.height : (<HTMLElement>within).offsetHeight;
        const width = (within === screen) ? screen.width : (<HTMLElement>within).offsetWidth;


       if (this.gazeDetector)
            this.gazeDetector.Target = {x: 0, y: 0}

        const new_pos = () => {
            if (this.gazeDetector && this_.isGazeDetectionActive && this.gazeDetector.Target) {
                const t_width = this.gazeDetector.TargetWidth / 2 + 5;
                const t_height = this.gazeDetector.TargetHeight  / 2 + 5;
                const x_positions: number[] = [t_width, t_width, t_width, width / 2 - 50, width / 2 - 50, width / 2 - 50, width / 2,  width / 2 + 50, width - t_width, width - t_width, width - t_width];
                const y_positions: number[] = [t_height, t_height, t_height, height / 2 - 50, height / 2, height / 2 + 50, height - t_height, height - t_height, height - t_height];

                const center_x = width / 2;
                const center_y = height / 2 - 250;

                for (const offset of [-200, -100, 0, 0, 0, 100, 200]) {
                    x_positions.push(center_x + offset);
                    y_positions.push(center_y + offset);
                }
                const jitter = [-1,-2,-3,-4, 4, 3, 2, 1, 0 ]

                this.gazeDetector.Target = {x: left + x_positions.randomElement() + jitter.randomElement(), y: top + y_positions.randomElement() + jitter.randomElement()}
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
                // const jitter = [-1,-2,-3,-4, 4, 3, 2, 1, 0 ]
                // // @ts-ignore
                // this.gazeDetector.Target =  { x: features.gaze.x + jitter.randomElement(), y: features.gaze.y + jitter.randomElement()};
            }
        );

        await this.gazeDetector.init();

    }

}