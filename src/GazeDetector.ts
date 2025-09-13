
import EventEmitter from "eventemitter3";

import {LandMarkDetector} from "./LandMarkDetector";
import {FaceLandmarker} from "@mediapipe/tasks-vision";
import {ContinuousTrainer} from "./ContinuousTrainer";
import {GazeElement} from "./GazeElement";
import { ui } from "./UI";
import {Coord, PixelCoord} from "./util/Coords";
import {post_data} from "./apiService";



export interface iGazeDetectorTrainResult {
    h_loss: number;
    v_loss: number;
    loss: number;
}

export interface iGazeDetectorAddDataResult {
    data_index: number;
    gaze: Coord; // Predicted gaze of last landmark in batch in screen coordinates
    losses: iGazeDetectorTrainResult;
}


export type BatchItem = {
    landmarks: PixelCoord[];   // [[x,y,z], ...]
    target: number[] | null;    // screen-space target (or omit)
}


// https://github.com/webrtcHacks/tfObjWebrtc/blob/master/static/objDetect.js
export class GazeDetector extends EventEmitter {
//for starting events
    isPlaying: boolean = false;
    gotMetaData: boolean = false;

    training_loss: number = -1;
    frame_rate: number = 0;
    continuousTrainer: ContinuousTrainer | undefined = undefined;
    training_promise: Promise<void> | undefined = undefined;
    private containerDiv: HTMLDivElement;

    static ModelToScreenCoords(point: Coord | PixelCoord): Coord {
        let x, y;

        const isPixelCoord: boolean = Array.isArray(point)
        if (!isPixelCoord) {
            x = (<Coord>point).x;
            y = (<Coord>point).y
            x /= 2;
            y /= 2;
            x += 0.5;
            y += 0.5;
            x *= screen.width;
            y *= screen.height;
        } else {
            x = (<PixelCoord>point)[0];
            y = (<PixelCoord>point)[1];
        }
        return {x: Math.round(x), y: Math.round(y)}
    }

    static screenToModelCoords(point: Coord | undefined): Coord | undefined {
        if (!point)
            return undefined;
        let [x, y] = [point.x, point.y];
        x /= screen.width;
        y /= screen.height;
        x -= 0.5;
        y -= 0.5;
        x *= 2;
        y *= 2;
        return {x: x, y: y}

    }


//create a canvas to grab an image for upload
//     imageCanvas: HTMLCanvasElement;
//     imageCtx: CanvasRenderingContext2D;

    overlayCanvas: HTMLCanvasElement;
    overlayCtx: CanvasRenderingContext2D;

    private target_pos: Coord | undefined = undefined;
    private next_target_pos: Coord | undefined = undefined;
    private targetElement: GazeElement;

    // target won't be set until 'transitionend' event is fired,
    // i.e. when the target element has reached its new position
    public set TargetPos(next_target_pos: Coord | undefined) {

        const targetElement = this.targetElement;
        if (next_target_pos === undefined)
            this.target_pos = undefined;  // Don't wait for transition to end if we're stopping calibration

        // setVisiblePosition() may set the target's position to a different value, it may adjust it to fit screen
        this.next_target_pos = targetElement.setPosition(next_target_pos);

    }

    public get TargetPos(): Coord | undefined {
        return this.next_target_pos;
    }

    public get FrameRate(): number {
        return this.frame_rate;
    }

    public get isTraining(): boolean {
        return this.training_promise !== undefined;
    }

    // public Mode: GazeDetectorMode = 'features';

    videoCaptureElement: HTMLVideoElement;

    public async term() {
        this.removeAllListeners();
        this.isPlaying = false;
        const v = this.videoCaptureElement;
        v.srcObject = v.onloadedmetadata = v.onplaying = null;
    }

    public async init() {
        const this_ = this;
        return new Promise((resolve, reject) => {
            navigator.mediaDevices.getUserMedia({
                audio: false,
                video:
                    {
                        width: {min: 640, ideal: 1280, max: 1280},
                        height: {min: 480, ideal: 960, max: 960},
                        frameRate: {ideal: 30, max: 30}
                    }
            })
                .then(stream => {
                    this_.videoCaptureElement.srcObject = stream;
                    console.log("Got local user video");
                    const v = this_.videoCaptureElement;
                    //Starting events

//check if metadata is ready - we need the video size
                    v.onloadedmetadata = () => {
                        console.log("video metadata ready");
                        this_.gotMetaData = true;
                        const v = this.videoCaptureElement;
                        v.width = v.videoWidth;
                        v.height = v.videoHeight;
                        this_.containerDiv.style.width = v.videoWidth + "px";
                        this_.containerDiv.style.height = v.videoHeight + "px";
                        if (this_.isPlaying)
                            this_.startGazeDetection();
                    };

//see if the video has started playing
                    v.onplaying = () => {
                        console.log("video playing");
                        this_.isPlaying = true;
                        if (this_.gotMetaData) {
                            this_.startGazeDetection();
                        }
                    };
                    this_.once('ProcessedFrame', resolve);

                })
                .catch(err => {
                    console.log('navigator.getUserMedia error: ', err)
                    reject();
                });

        });

    }

    async startTraining() {
        if (!this.continuousTrainer) {
            this.continuousTrainer = new ContinuousTrainer();
            this.continuousTrainer.on('data', (loss) => {
            });

            this.training_promise = this.continuousTrainer.Start();
        }

    }
    async stopTraining() {
        if (this.continuousTrainer) {
            this.continuousTrainer.Stop();
            if (this.training_promise)
                await this.training_promise;
            this.continuousTrainer = this.training_promise = undefined;
        }
    }
    async startGazeDetection() {
        const this_ = this;
        const v = this.videoCaptureElement;
        v.width = v.videoWidth;
        v.height = v.videoHeight;


        const overlayCanvas = this.overlayCanvas;

        const overlayCtx = this.overlayCtx;

        console.log("starting gaze detection");

        overlayCanvas.width = v.videoWidth; // uploadWidth;
        overlayCanvas.height = v.videoHeight; //uploadWidth * (v.videoHeight / v.videoWidth);

        // If user clicks on the overlay canvas, we set the target position to the clicked point in absolute screen coordinates (not local to the canvas)
        overlayCanvas.addEventListener('click', (evt) => {
            const x = evt.screenX;
            const y = evt.screenY;
            this_.TargetPos = {x: x, y: y};

        });
        // set the TargetPos to undefined when the mouse leaves the overlay canvas
        overlayCanvas.addEventListener('mouseleave', () => {
            this_.TargetPos = undefined;
        });
        // set the cursor to a crosshair when hovering over the overlay canvas
        overlayCanvas.style.cursor = 'crosshair';

        let frame_count: number = 0;
        const num_frames_for_frame_rate_measurement = 10;
        let time_at_last_frame_rate_measurement = window.performance.now();


        // --- batching state ---
        let sendInFlight = false;
        let batchBuffer: BatchItem[] = [];

// Prevent unbounded growth if backend stalls
        const MAX_BACKLOG_ITEMS = 30; // ~1s of data @30fps

        function enqueueTrainingSample(item:BatchItem) {
            // Coalesce into an ever-growing batch while backend is busy
            batchBuffer.push(item);
            if (batchBuffer.length > MAX_BACKLOG_ITEMS) {
                // Keep the most recent data if we hit the cap
                batchBuffer.splice(0, batchBuffer.length - MAX_BACKLOG_ITEMS);
            }
            // If backend is "ready" (i.e., no request in flight), ship the batch now
            if (!sendInFlight) void sendBatchNow();
        }

        async function sendBatchNow() {
            if (sendInFlight) return;
            const batch = batchBuffer.splice(0, batchBuffer.length); // drain current buffer
            if (batch.length === 0) return;
            console.log(`Batch size: ${batch.length}`);
            sendInFlight = true;
            try {
                const features = await post_data(batch);
                if (features) {
                    this_.training_loss = features.losses.loss;
                    features.gaze = GazeDetector.ModelToScreenCoords(features.gaze);
                    this_.emit('GazeDetectionComplete', features);
                }

                // (optional) re-queue on transient failure
                batchBuffer = batch.concat(batchBuffer).slice(-MAX_BACKLOG_ITEMS);
            } finally {
                sendInFlight = false;
                // If frames arrived during the send, immediately ship them as the next batch
                if (batchBuffer.length) void sendBatchNow();
            }
        }




        const processFrame = async () => {
            if (this_.isPlaying) {



                const landmarkerResult = await this.landmarkDetector.GetLandmarks();

                const landmarks = landmarkerResult.faceLandmarks[0];
                this_.emit('LandMarkDetectionComplete', landmarks);

                if (landmarks) {

                    // Overlay the landmarks

                    overlayCtx.clearRect(0, 0, 10000, 10000)

                    for (const [idx, point] of landmarks.entries()) {
                        const lms = this.displayLandmarkArray[idx];
                        if (lms.visible) {
                            overlayCtx.fillStyle = lms.color;
                            overlayCtx.setTransform({a: -1, e: overlayCanvas.width})
                            overlayCtx.beginPath();
                            overlayCtx.arc(point.x * overlayCanvas.width, point.y * overlayCanvas.height, 2, 0, 2 * Math.PI);
                            overlayCtx.fill();
                        }
                    }
                    const landmarks_as_array = landmarks.map((p) => [p.x, p.y, p.z]);

                    const target_ = this_.target_pos ? GazeDetector.screenToModelCoords(this_.target_pos) : null;

                    enqueueTrainingSample( { landmarks: landmarks_as_array, target: target_ ? [ target_.x, target_.y] : null });

                }

                if (++frame_count >= num_frames_for_frame_rate_measurement) {
                    frame_count = 0;
                    const now = window.performance.now();
                    this_.frame_rate = 1000.0 * num_frames_for_frame_rate_measurement / (now - time_at_last_frame_rate_measurement);

                    time_at_last_frame_rate_measurement = now;
                }
                this_.emit('ProcessedFrame');
                this.videoCaptureElement.requestVideoFrameCallback(processFrame);
            }
        }

        this.videoCaptureElement.requestVideoFrameCallback(processFrame);

    }

    private landmarkDetector: LandMarkDetector;

    private displayLandmarkArray: { visible: boolean, color: string }[] = [];

    constructor(videoCaptureElement: HTMLVideoElement, landmarkSelectorElement: HTMLDivElement | undefined) {
        super();


        this.landmarkDetector = new LandMarkDetector(videoCaptureElement);

        const plotsDiv: HTMLDivElement = ui.plotsDiv;


        this.videoCaptureElement = videoCaptureElement;


        const gazeElement = <GazeElement>document.createElement('gaze-element');
        const targetElement = <GazeElement>document.createElement('gaze-element');

        gazeElement.setBackground(`radial-gradient(#97dc81, #2f560900)`);
        gazeElement.setRadius(10, 10);
        gazeElement.setTransitionStyle('all 0.2s ease-in-out')

        // Hide the target element
        targetElement.setPosition(undefined);

        targetElement.setTransitionStyle('all 0.4s ease-in-out');
        // Default size for target.
        // Will change size and shape during training to reflect horizontal and vertical loss
        targetElement.setRadius(25, 25);

        const this_ = this;

        // Set the target pos only after it's completed its animation to the new position.
        // This animation allows the eyes to settle on the new target position before we
        // send the landmarks and target to the model trainer
        targetElement.onReachedNewPosition(() => {
            this_.target_pos = this_.next_target_pos
        });

        this.containerDiv = ui.vidCapContainer;
        this.overlayCanvas = ui.overlayCanvas;
        this.overlayCtx = this.overlayCanvas.getContext("2d") as CanvasRenderingContext2D;
        // Append in this order for z-order (gaze above target)
        document.documentElement.appendChild(targetElement);
        document.documentElement.appendChild(gazeElement);


        this.targetElement = targetElement;

        let current_x = 0;
        let current_y = 0;
        this.on('GazeDetectionComplete', (features: iGazeDetectorAddDataResult) => {

            gazeElement.setPosition(features.gaze);

            let rx = features.losses.h_loss * screen.width;
            let ry = features.losses.v_loss * screen.height;

            // Don't allow the target ellipse to be too big (if the loss is big) or too small
            rx = rx.clamp(15, 100);
            ry = ry.clamp(15, 100);


            targetElement.setRadius(rx, ry);
            targetElement.setCaption(`<div>${features.data_index}</div>`)

        });

        // Landmark selector
        if (landmarkSelectorElement) {
            const GroupsElement = <HTMLDivElement>landmarkSelectorElement.children[0];
            const CheckboxesElement = <HTMLDivElement>landmarkSelectorElement.children[1];
            const displayLandmarkArray = this.displayLandmarkArray;
            for (let i = 0; i < 478; ++i) {
                const checkBoxEl = document.createElement('input');
                checkBoxEl.type = 'checkbox';
                checkBoxEl.title = '' + i;
                // mouse pointer changes to crosshair when hovering over checkbox
                checkBoxEl.style.cursor = 'crosshair';
                displayLandmarkArray.push({color: '#ffffff', visible: false});
                checkBoxEl.addEventListener('click', (evt) => {
                    const el = <HTMLInputElement>evt.target;
                    displayLandmarkArray[Number.parseInt(el.title)].visible = el.checked;
                })
                checkBoxEl.addEventListener('mouseenter', (evt) => {
                    const el = <HTMLInputElement>evt.target;
                    displayLandmarkArray[Number.parseInt(el.title)].visible = true;
                })

                checkBoxEl.addEventListener('mouseleave', (evt) => {
                    const el = <HTMLInputElement>evt.target;
                    displayLandmarkArray[Number.parseInt(el.title)].visible = el.checked;
                })
                CheckboxesElement.appendChild(checkBoxEl);
            }

            const myConnections = [];
            const r1 = [127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389,356,];
            const r2 = [246, 161, 160, 159, 158, 157, 173, 8, 398, 384, 385, 385, 387, 388, 466,];
            const r3 = [33, 468, 133, 168, 362, 473, 263,];
            const r4 = [7, 163, 144, 145, 153, 154, 155, 6, 382, 381, 380, 374, 373, 390, 249,];
            const r5 = [132, 147, 187, 207, 206, 165, 167, 164, 393, 391, 426, 427, 411, 376, 361,];
            for (const lms of [
                {connections: [], indexes:  [127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389,356,], color: '#ffffff', name: 'l2s_r1'},
                {connections: [], indexes:  [246, 161, 160, 159, 158, 157, 173, 8, 398, 384, 385, 386, 387, 388, 466,], color: '#ffffff', name: 'l2s_r2'},
                {connections: [], indexes:  [33, 470, 471, 468, 469, 472, 133, 168, 362, 477, 476, 473, 474, 475, 263,], color: '#ffffff', name: 'l2s_r3'},
                {connections: [], indexes:  [7, 163, 144, 145, 153, 154, 155, 6, 382, 381, 380, 374, 373, 390, 249,], color: '#ffffff', name: 'l2s_r4'},
                {connections: [], indexes:  [132, 147, 187, 207, 206, 165, 167, 164, 393, 391, 426, 427, 411, 376, 361,], color: '#ffffff', name: 'l2s_r5'},
                {connections: FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, indexes: [], color: '#800000', name: 'face-oval'},
                {connections: FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, indexes: [], color: '#0010ff', name: 'left-eye'},
                {connections: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, indexes: [], color: '#00507f', name: 'right-eye'},
                {connections: FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, indexes: [], color: '#009f30', name: 'left-eyebrow'},
                {connections: FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, indexes: [], color: '#007f60', name: 'right-eyebrow'},
                {connections: FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, indexes: [],color: '#ffff00', name: 'left-iris'},
                {connections: FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, indexes: [],color: '#8f8f00', name: 'right-iris'},
                {connections: FaceLandmarker.FACE_LANDMARKS_LIPS, indexes: [],color: '#ff8de2', name: 'lips'},

            ]) {
                const groupCheckBoxEl = document.createElement('input');
                const groupCheckBoxLabelEl = document.createElement('label');

                groupCheckBoxLabelEl.appendChild(document.createTextNode(lms.name));

                groupCheckBoxEl.type = 'checkbox';
                groupCheckBoxEl.style.accentColor = lms.color;
                groupCheckBoxLabelEl.appendChild(groupCheckBoxEl);
                groupCheckBoxEl.addEventListener('click', (evt) => {
                    const el = <HTMLInputElement>evt.target;
                    for (const c of lms.connections) {
                        const checkBoxEl = <HTMLInputElement>CheckboxesElement.children[c.start];
                        checkBoxEl.checked = el.checked;
                        checkBoxEl.style.accentColor = lms.color;
                        displayLandmarkArray[c.start] = {visible: el.checked, color: lms.color};
                    }
                    for (const i of lms.indexes) {
                        const checkBoxEl = <HTMLInputElement>CheckboxesElement.children[i];
                        checkBoxEl.checked = el.checked;
                        checkBoxEl.style.accentColor = lms.color;
                        displayLandmarkArray[i] = {visible: el.checked, color: lms.color};
                    }

                })
                GroupsElement.appendChild(groupCheckBoxLabelEl);

                // for (const c of lms.connections) {
                //     const checkBoxEl = <HTMLInputElement>CheckboxesElement.children[c.start];
                //     checkBoxEl.checked = true;
                //     checkBoxEl.style.accentColor = lms.color;
                //     displayLandmarkArray[c.start] = { visible: true, color: lms.color};
                // }
            }

        }
    }
}

