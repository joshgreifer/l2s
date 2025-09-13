import type {
    BatchItem,
    iGazeDetectorAddDataResult,
    iGazeDetectorTrainResult,
} from "./training/Trainer";
import { webOnnx } from "./runtime/WebOnnxAdapter";

let data_index = 0;

export async function apiAvailable(): Promise<boolean> {
    return webOnnx.ready;
}

export async function save_gaze_model(): Promise<boolean> {
    // Saving the model locally has not been implemented yet.
    console.warn("save_gaze_model is not implemented in the browser runtime");
    return false;
}

export async function train(
    epochs: number,
    action: "train" | "calibrate",
): Promise<iGazeDetectorTrainResult> {
    // Local training is not yet supported. Return zeroed losses so callers can proceed.
    console.warn("train is not implemented in the browser runtime", { epochs, action });
    return { h_loss: 0, v_loss: 0, loss: 0 };
}

export async function post_data(
    batch: BatchItem[],
): Promise<iGazeDetectorAddDataResult | undefined> {
    if (!webOnnx.ready) {
        console.warn("ONNX model not ready; cannot process gaze data locally.", batch);
        return undefined;
    }

    const last = batch[batch.length - 1];
    const [gx, gy] = await webOnnx.predict(last.landmarks);
    const [tx, ty] = last.target ?? [0, 0];
    const h_loss = Math.abs(gx - tx);
    const v_loss = Math.abs(gy - ty);
    const loss = (h_loss + v_loss) / 2;
    return {
        data_index: data_index++,
        gaze: { x: gx, y: gy },
        losses: { h_loss, v_loss, loss },
    };
}

