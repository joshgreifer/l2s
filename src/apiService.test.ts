import { describe, it, expect, vi } from "vitest";

vi.mock("./runtime/WebOnnxAdapter", () => {
    return {
        webOnnx: {
            ready: true,
            predict: vi.fn().mockResolvedValue([[1, 2]]),
            exportMlpModel: vi.fn().mockResolvedValue(new ArrayBuffer(4)),
            init: vi.fn().mockResolvedValue(undefined),
        },
    };
});

vi.mock("./runtime/TrainableOnnx", () => {
    return {
        trainableOnnx: {
            transformPca: vi.fn(async (_l: Float32Array, batch: number) => new Float32Array(batch * 32)),
            trainMlpBatch: vi.fn(async () => [0.1]),
            exportMlpModel: vi.fn(async () => new ArrayBuffer(4)),
        },
    };
});

const localStore: Record<string, string> = {};
// @ts-ignore
vi.stubGlobal("localStorage", {
    getItem: (k: string) => (k in localStore ? localStore[k] : null),
    setItem: (k: string, v: string) => {
        localStore[k] = v;
    },
});

import { apiAvailable, post_data, train, save_gaze_model } from "./apiService";
import type { BatchItem } from "./training/Trainer";

describe("apiService", () => {
    it("apiAvailable resolves true when webOnnx is ready", async () => {
        const fetchSpy = vi.fn();
        (globalThis as any).fetch = fetchSpy;
        expect(await apiAvailable()).toBe(true);
        expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("post_data uses webOnnx for local predictions", async () => {
        const landmarks: [number, number, number][] = Array.from(
            { length: 478 },
            () => [0, 0, 0],
        );
        const sample: BatchItem = { landmarks, target: [3, 4] };
        const fetchSpy = vi.fn();
        (globalThis as any).fetch = fetchSpy;
        const result = await post_data(sample);
        expect(result).toBeDefined();
        expect(result?.gaze).toEqual({ x: 1, y: 2 });
        expect(result?.losses).toEqual({ h_loss: 2, v_loss: 2, loss: 2 });
        expect(fetchSpy).not.toHaveBeenCalled();
    });

    it("train returns updated losses", async () => {
        const landmarks = new Float32Array(478 * 3);
        const targets = new Float32Array([3, 4]);
        const losses = await train(landmarks, targets, 1, "train");
        expect(losses.loss).toBeGreaterThanOrEqual(0);
    });

    it("save_gaze_model triggers download", async () => {
        const clickSpy = vi.fn();
        // @ts-ignore
        globalThis.document = {};
        // @ts-ignore
        document.body = { appendChild: vi.fn(), removeChild: vi.fn() };
        // @ts-ignore
        document.createElement = vi.fn().mockReturnValue({ click: clickSpy });
        // @ts-ignore
        URL.createObjectURL = vi.fn().mockReturnValue("blob:foo");
        // @ts-ignore
        URL.revokeObjectURL = vi.fn();
        const ok = await save_gaze_model();
        expect(ok).toBe(true);
        expect(clickSpy).toHaveBeenCalled();
    });
});

