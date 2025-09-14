import EventEmitter from "eventemitter3";
import {Coord, PixelCoord, screenToModelCoords} from "./util/Coords";
import type {iGazeDetectorAddDataResult, IGazeTrainer} from "./training/Trainer";
import {NormalizedLandmark} from "@mediapipe/tasks-vision";

/**
 * Packages landmark and target data and forwards samples to the trainer,
 * re-emitting predictions as they arrive.
 */
export class SampleCollector extends EventEmitter {
    private trainer: IGazeTrainer | undefined = undefined;

    private onTrainerPrediction = (features: iGazeDetectorAddDataResult) => {
        this.emit('prediction', features);
    };

    public set Trainer(t: IGazeTrainer | undefined) {
        if (this.trainer) this.trainer.off('prediction', this.onTrainerPrediction);
        this.trainer = t;
        if (t) t.on('prediction', this.onTrainerPrediction);
    }

    public get Trainer(): IGazeTrainer | undefined {
        return this.trainer;
    }

    public collect(landmarks: NormalizedLandmark[], target: Coord | undefined): void {
        const landmarks_as_array: PixelCoord[] = landmarks.map((p) => [p.x, p.y, p.z]);
        const target_model = target ? screenToModelCoords(target) : null;
        this.trainer?.addSample({
            landmarks: landmarks_as_array,
            target: target_model ? [target_model.x, target_model.y] : null,
        });
    }
}

