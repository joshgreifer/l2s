import {post_landmarks} from "./apiService";
import EventEmitter from "eventemitter3";

import {LandMarkDetector} from "./LandMarkDetector";
import {BoundingBox, Detection, FaceLandmarker} from "@mediapipe/tasks-vision";
import {ContinuousTrainer} from "./ContinuousTrainer";
import {GazeElement} from "./GazeElement";

export type Coord = { x: number; y: number; }
export type PixelCoord = number[]

export interface iGazeDetectorTrainResult {
    h_loss: number;
    v_loss: number;
    loss: number;
}

export interface iGazeDetectorResult {
    data_index: number;
    faces: number;
    eyes: number;
    gaze: Coord;
    landmarks: PixelCoord[];
    losses: iGazeDetectorTrainResult;
}


const constraints = {
    audio: false,
    video:
        {
            width: {min: 640, ideal: 720, max: 1280},
            height: {min: 480, ideal: 540, max: 1280}
        }
};

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

    static toScreenCoords(point: Coord | PixelCoord): Coord {
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

    static fromScreenCoords(point: Coord | undefined): Coord | undefined {
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
            navigator.mediaDevices.getUserMedia(constraints)
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
                    this_.once('GazeDetectionComplete', resolve);

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


        const processFrame = async () => {
            if (this_.isPlaying) {

                let features: iGazeDetectorResult | undefined = undefined;

                // const faceBoundingBox = await this.landmarkDetector.GetFaceBoundingBox();
                // await copyFaceToTarget(faceBoundingBox);

                // imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, overlayCanvas.width, overlayCanvas.width * (v.videoHeight / v.videoWidth));

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
                    features = await post_landmarks(landmarks_as_array, GazeDetector.fromScreenCoords(this_.target_pos));

                    // features = await post_landmark_features(landmarkFeatures, GazeDetector.fromScreenCoords(this_.target_pos));
                    // if (features) {
                    //     for (let i = 0; i < features.landmarks.length; ++i) {
                    //         features.landmarks[i][0] *= overlayCanvas.width;
                    //         features.landmarks[i][1] *= overlayCanvas.height;
                    //     }
                    // }
                }


                if (features !== undefined) {

                    this_.training_loss = features.losses.loss;
                    features.gaze = GazeDetector.toScreenCoords(features.gaze)
                    this_.emit('GazeDetectionComplete', features);


                    // overlayCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));

                    if (++frame_count >= num_frames_for_frame_rate_measurement) {
                        frame_count = 0;
                        const now = window.performance.now();
                        this_.frame_rate = (now - time_at_last_frame_rate_measurement) * num_frames_for_frame_rate_measurement / 1000.0;
                        time_at_last_frame_rate_measurement = now;
                    }
                }
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

        const plotsDiv: HTMLDivElement = document.querySelector('#plots') as HTMLDivElement;


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

        this.containerDiv = <HTMLDivElement>document.querySelector('.vidcap')
        this.overlayCanvas = <HTMLCanvasElement>document.querySelector('#overlayCanvas')
        this.overlayCtx = this.overlayCanvas.getContext("2d") as CanvasRenderingContext2D;
        // Append in this order for z-order (gaze above target)
        document.documentElement.appendChild(targetElement);
        document.documentElement.appendChild(gazeElement);


        this.targetElement = targetElement;

        let current_x = 0;
        let current_y = 0;
        this.on('GazeDetectionComplete', (features: iGazeDetectorResult) => {

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

