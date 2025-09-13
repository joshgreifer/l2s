import type {
    BatchItem,
    iGazeDetectorAddDataResult,
    iGazeDetectorTrainResult,
} from "./training/Trainer";
import { webOnnx } from "./runtime/WebOnnxAdapter";

let data_index = 0;
const modelBias: [number, number] = [0, 0];

export async function apiAvailable(): Promise<boolean> {
    return webOnnx.ready;
}

export async function save_gaze_model(): Promise<boolean> {
    try {
        const data = JSON.stringify({ bias: modelBias });
        const blob = new Blob([data], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "gaze_model.json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return true;
    } catch (err) {
        console.error("Failed to save gaze model", err);
        return false;
    }
}

export async function train(
    batch: BatchItem[],
    epochs: number,
    action: "train" | "calibrate",
): Promise<iGazeDetectorTrainResult> {
    if (!webOnnx.ready) {
        console.warn("ONNX model not ready; cannot train locally.", { batch, epochs, action });
        return { h_loss: 0, v_loss: 0, loss: 0 };
    }

    const lr = action === "calibrate" ? 0.05 : 0.01;
    let h_sum = 0;
    let v_sum = 0;
    let n = 0;

    for (let e = 0; e < epochs; e++) {
        for (const item of batch) {
            if (!item.target) continue;
            const [gx0, gy0] = await webOnnx.predict(item.landmarks);
            const gx = gx0 + modelBias[0];
            const gy = gy0 + modelBias[1];
            const [tx, ty] = item.target;
            const dx = tx - gx;
            const dy = ty - gy;
            modelBias[0] += lr * dx;
            modelBias[1] += lr * dy;
            h_sum += Math.abs(dx);
            v_sum += Math.abs(dy);
            n++;
        }
    }

    const h_loss = n ? h_sum / n : 0;
    const v_loss = n ? v_sum / n : 0;
    const loss = (h_loss + v_loss) / 2;
    return { h_loss, v_loss, loss };
}

export async function post_data(
    batch: BatchItem[],
): Promise<iGazeDetectorAddDataResult | undefined> {
    if (!webOnnx.ready) {
        console.warn("ONNX model not ready; cannot process gaze data locally.", batch);
        return undefined;
    }

    const last = batch[batch.length - 1];
    const [gx0, gy0] = await webOnnx.predict(last.landmarks);
    const gx = gx0 + modelBias[0];
    const gy = gy0 + modelBias[1];
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

