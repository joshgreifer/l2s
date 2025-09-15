import type {
    BatchItem,
    iGazeDetectorAddDataResult,
    iGazeDetectorTrainResult,
} from "./training/Trainer";
import { webOnnx } from "./runtime/WebOnnxAdapter";
import { trainableOnnx } from "./runtime/TrainableOnnx";
import type { PixelCoord } from "./util/Coords";

let data_index = 0;
let savedMlp: ArrayBuffer | null = null;

try {
    if (typeof localStorage !== "undefined") {
        const raw = localStorage.getItem("gaze_mlp_trained");
        if (raw) {
            const parsed = JSON.parse(raw);
            if (parsed.mlp) {
                const arr = Uint8Array.from(parsed.mlp as number[]);
                savedMlp = arr.buffer;
            }
        }
    }
} catch (err) {
    console.warn("Failed to load saved gaze model", err);
}

export function getSavedGazeModel(): ArrayBuffer | null {
    return savedMlp;
}

export async function apiAvailable(): Promise<boolean> {
    return webOnnx.ready;
}

export async function save_gaze_model(): Promise<boolean> {
    try {
        const mlp = await webOnnx.exportMlpModel?.();
        if (!mlp) throw new Error("No MLP model available");
        const payload = {
            mlp: Array.from(new Uint8Array(mlp)),
        };
        const json = JSON.stringify(payload);
        if (typeof localStorage !== "undefined") {
            localStorage.setItem("gaze_mlp_trained", json);
        }
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "gaze_mlp_trained.json";
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
    landmarks: Float32Array,
    targets: Float32Array,
    epochs: number,
    action: "train" | "calibrate",
): Promise<iGazeDetectorTrainResult> {
    if (!webOnnx.ready) {
        console.warn("ONNX model not ready; cannot train locally.", { epochs, action });
        return { h_loss: 0, v_loss: 0, loss: 0 };
    }

    const sampleCount = targets.length / 2;
    const features = await trainableOnnx.transformPca(landmarks, sampleCount);
    const lr = action === "calibrate" ? 0.05 : 0.01;
    await trainableOnnx.trainMlpBatch(features, targets, sampleCount, epochs, lr);

    const mlpBytes = await trainableOnnx.exportMlpModel();
    if (mlpBytes) {
        savedMlp = mlpBytes;
        await webOnnx.init(mlpBytes);
    }

    let h_sum = 0;
    let v_sum = 0;
    for (let i = 0; i < sampleCount; i++) {
        const sample: PixelCoord[] = [];
        for (let j = 0; j < 478; j++) {
            const base = i * 478 * 3 + j * 3;
            sample.push([landmarks[base], landmarks[base + 1], landmarks[base + 2]]);
        }
        const [gx, gy] = (await webOnnx.predict([sample]))[0];
        const tx = targets[i * 2];
        const ty = targets[i * 2 + 1];
        h_sum += Math.abs(gx - tx);
        v_sum += Math.abs(gy - ty);
    }
    const h_loss = sampleCount ? h_sum / sampleCount : 0;
    const v_loss = sampleCount ? v_sum / sampleCount : 0;
    const loss = (h_loss + v_loss) / 2;
    return { h_loss, v_loss, loss };
}

export async function post_data(
    item: BatchItem,
): Promise<iGazeDetectorAddDataResult | undefined> {
    if (!webOnnx.ready) {
        console.warn("ONNX model not ready; cannot process gaze data locally.", item);
        return undefined;
    }

    const [gx, gy] = (await webOnnx.predict([item.landmarks]))[0];
    let h_loss = 0;
    let v_loss = 0;
    let loss = 0;
    if (item.target) {
        const [tx, ty] = item.target;
        h_loss = Math.abs(gx - tx);
        v_loss = Math.abs(gy - ty);
        loss = (h_loss + v_loss) / 2;
    }
    return {
        data_index: data_index++,
        gaze: { x: gx, y: gy },
        losses: { h_loss, v_loss, loss },
    };
}

