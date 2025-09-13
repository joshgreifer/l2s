import { describe, it, expect, vi } from "vitest";

vi.mock("./runtime/WebOnnxAdapter", () => {
    return {
        webOnnx: {
            ready: true,
            predict: vi.fn().mockResolvedValue([1, 2]),
        },
    };
});

import { apiAvailable, post_data } from "./apiService";
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
        const result = await post_data([sample]);
        expect(result).toBeDefined();
        expect(result?.gaze).toEqual({ x: 1, y: 2 });
        expect(result?.losses).toEqual({ h_loss: 2, v_loss: 2, loss: 2 });
        expect(fetchSpy).not.toHaveBeenCalled();
    });
});

