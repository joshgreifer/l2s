import {NormalizedLandmark} from "@mediapipe/tasks-vision";

export interface LandmarkDisplayOption {
    visible: boolean;
    color: string;
}

/**
 * Draws facial landmark overlays on a canvas using configurable display options
 * for each landmark point.
 */
export class LandmarkRenderer {
    private ctx: CanvasRenderingContext2D;
    public displayLandmarkArray: LandmarkDisplayOption[] = [];

    constructor(private canvas: HTMLCanvasElement) {
        this.ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    }

    public render(landmarks: NormalizedLandmark[]): void {
        const overlayCanvas = this.canvas;
        const overlayCtx = this.ctx;
        overlayCtx.clearRect(0, 0, 10000, 10000);
        for (const [idx, point] of landmarks.entries()) {
            const lms = this.displayLandmarkArray[idx];
            if (lms?.visible) {
                overlayCtx.fillStyle = lms.color;
                overlayCtx.setTransform({a: -1, e: overlayCanvas.width});
                overlayCtx.beginPath();
                overlayCtx.arc(point.x * overlayCanvas.width, point.y * overlayCanvas.height, 2, 0, 2 * Math.PI);
                overlayCtx.fill();
            }
        }
    }
}

