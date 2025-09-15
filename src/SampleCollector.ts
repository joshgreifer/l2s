import EventEmitter from "eventemitter3";
import { Coord, PixelCoord, screenToModelCoords } from "./util/Coords";
import type { iGazeDetectorAddDataResult, IGazeTrainer } from "./training/Trainer";
import { NormalizedLandmark } from "@mediapipe/tasks-vision";
import { post_data } from "./apiService";

/**
 * Packages landmark and target data and forwards samples to the trainer,
 * re-emitting predictions as they arrive.
 */
export class SampleCollector extends EventEmitter {
    private trainer: IGazeTrainer | undefined = undefined;
    private predicting = false;

    public set Trainer(t: IGazeTrainer | undefined) {
        this.trainer = t;
    }

    public get Trainer(): IGazeTrainer | undefined {
        return this.trainer;
    }

    public collect(landmarks: NormalizedLandmark[], target: Coord | undefined): void {
        const landmarks_as_array: PixelCoord[] = landmarks.map((p) => [p.x, p.y, p.z]);
        const target_model = target ? screenToModelCoords(target) : undefined;
        if (target_model) {
            this.trainer?.addSample({
                landmarks: landmarks_as_array,
                target: [target_model.x, target_model.y],
            });
        }
        if (this.predicting) return;
        this.predicting = true;
        void post_data({
            landmarks: landmarks_as_array,
            target: target_model ? [target_model.x, target_model.y] : undefined,
        })
            .then((features: iGazeDetectorAddDataResult | undefined) => {
                if (features) this.emit('prediction', features);
            })
            .finally(() => {
                this.predicting = false;
            });
    }
}

