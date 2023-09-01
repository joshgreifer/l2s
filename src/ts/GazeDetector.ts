import {post_landmarks, train} from "./apiService";
import EventEmitter from "eventemitter3";
import {ScopeElement} from "./ScopeElement";
import {Channel, RenderStyle, Scope, SignalFollowBehaviour, TimeAxisFormat} from "../DataPlotting/Scope";
import {DataConnection} from "../DataConnection";
import {LandMarkDetector} from "../LandMarkDetector";
import {BoundingBox, Detection, FaceLandmarker} from "@mediapipe/tasks-vision";

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

export type eyeState_t = 'open' | 'closed' | undefined

export function eyeState(features: iGazeDetectorResult): eyeState_t {
    return  features.faces > 0 ? (features.eyes > 0 ? 'open' : 'closed') : undefined;
}

export function faceVisible(features: iGazeDetectorResult): boolean {
    return features.faces > 0;
}

export class GazeElement extends HTMLElement {
    public setBackground: (css: string) => void;
    public setRadius: (rx: number, ry: number) => void;
    public setCaption: (text: string) => void;

    constructor() {
        super();
        const shadow = this.attachShadow({mode: 'open'}); // sets and returns 'this.shadowRoot'
        const el = <HTMLDivElement>document.createElement('div');
        const style = document.createElement('style');
        // noinspection CssInvalidPropertyValue

        const my_radius = 20;
        el.className = 'private-style1';
        // noinspection CssInvalidFunction,CssInvalidPropertyValue
        style.textContent = `
        .private-style1 {
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            height: ${my_radius * 2}px;
            width: ${my_radius * 2}px;
            border-radius: 100%;
            background-image: radial-gradient(#ff0000, rgba(87,7,7,0));
            font-family: sans-serif;
            font-size: x-small;
        }
`;
        this.setBackground = (css: string) => el.style.background = css;
        this.setRadius = (rx: number, ry: number) => {
            rx = Math.round(rx);
            ry = Math.round(ry);
            el.style.width = 2 * rx + 'px';
            el.style.height = 2 * ry + 'px';
        }
        this.setCaption = (text: string) => el.innerHTML = text;
        shadow.append(style, el);
    }


}

export class ContinuousTrainer extends DataConnection {
    private stop_request: boolean = false;
    public Stop() { this.stop_request = true;  }

    async Start() {
        while (!this.stop_request) {
            let t = window.performance.now();
            const losses = await train(1);
            console.debug("LOSS:", losses);
            this.AddData(new Float32Array([losses.loss, losses.v_loss, losses.h_loss]));
            const t2 = window.performance.now();
            const ms_per_epoch = t2 - t;

            t = t2;
        }
    }
    constructor(interval_period_secs = 10) {

        super(1, 3, Float32Array, 1000);
        this.on('data', (data) => console.log(data))

    }
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
    lossDisplayScope: Scope;
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
        return { x: Math.round(x), y: Math.round(y) }
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
        return { x: x, y: y }

    }


//create a canvas to grab an image for upload
    imageCanvas: HTMLCanvasElement;
    imageCtx: CanvasRenderingContext2D;

    overlayCanvas: HTMLCanvasElement;
    overlayCtx: CanvasRenderingContext2D;

    private target: Coord | undefined = undefined;
    private next_target: Coord | undefined = undefined;
    private targetElement: GazeElement;

    public get TargetWidth(): number {
        return this.targetElement.clientWidth;
    }

    public get TargetHeight(): number {
        return this.targetElement.clientHeight;
    }
    // target won't be set until 'transitionend' event is fired,
    // i.e. when the target element has reached its new position
    public set Target(next_target: Coord | undefined) {
        console.log("Target:", next_target, GazeDetector.fromScreenCoords(next_target))
        this.next_target = next_target;

        const targetElement = this.targetElement;
        if (next_target === undefined) {
            this.target = undefined;  // Don't wait for transition to end if we're stopping calibration
            // this.Mode = 'features';
            targetElement.style.left = '-10000px';
            targetElement.style.top =  '-10000px';
        } else {

            const width = targetElement.clientWidth;
            const height = targetElement.clientHeight;

            targetElement.style.left = Math.round(next_target.x - width / 2) + 'px';
            targetElement.style.top = Math.round(next_target.y - height / 2) + 'px';

        }


    }

    public get Target() : Coord | undefined { return this.next_target; }

    public get FrameRate() : number { return this.frame_rate; }

    public get isTraining(): boolean { return this.target !== undefined; }

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
                        this_.containerDiv.style.height =v.videoHeight + "px";
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
    async toggleGazeDetectorTraining() {
        if (this.continuousTrainer) {
            this.continuousTrainer.Stop();
            if (this.training_promise)
                await this.training_promise;
            this.continuousTrainer = this.training_promise = undefined;
        } else {
            this.continuousTrainer = new ContinuousTrainer();
            this.continuousTrainer.on('data', (loss) => { this.lossDisplayScope.Title = `Loss: ${loss[0].toFixed(4)}`});

            this.training_promise = this.continuousTrainer.Start();
        }
        this.lossDisplayScope.Connection = this.continuousTrainer;  // if undefined, will disconnect scope
    }
    async startGazeDetection() {
        const this_ = this;
        const v = this.videoCaptureElement;
        v.width = v.videoWidth;
        v.height = v.videoHeight;


        const imageCanvas = this.imageCanvas;
        const overlayCanvas = this.overlayCanvas;
        const imageCtx = this.imageCtx;
        const overlayCtx = this.overlayCtx;

        console.log("starting gaze detection");


        imageCanvas.width = v.videoWidth; // uploadWidth;
        imageCanvas.height = v.videoHeight; //uploadWidth * (v.videoHeight / v.videoWidth);
        overlayCanvas.width = v.videoWidth; // uploadWidth;
        overlayCanvas.height = v.videoHeight; //uploadWidth * (v.videoHeight / v.videoWidth);

        let frame_count: number = 0;
        const num_frames_for_frame_rate_measurement = 10;
        let time_at_last_frame_rate_measurement = window.performance.now();

        const copyFaceToTarget = async (bbox?: BoundingBox) => {
            if (bbox) {
                const img = imageCtx.getImageData(bbox.originX, bbox.originY, bbox.width, bbox.height);
            }

        };

        const processFrame = async () => {
            if (this_.isPlaying) {

                let features: iGazeDetectorResult | undefined = undefined;

                // const faceBoundingBox = await this.landmarkDetector.GetFaceBoundingBox();
                // await copyFaceToTarget(faceBoundingBox);

                // imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, overlayCanvas.width, overlayCanvas.width * (v.videoHeight / v.videoWidth));

                const landmarkerResult = await this.landmarkDetector.GetLandmarks();

                this_.emit('LandMarkDetectionComplete', landmarkerResult.faceLandmarks[0]);
                const landmarkFeatures = LandMarkDetector.GetFeaturesFromLandmarks(landmarkerResult);
                if (landmarkFeatures) {

                    // Overlay the landmarks

                    overlayCtx.clearRect(0, 0, 10000, 10000)
                    overlayCtx.fillStyle = '#ff0000';
                    for (const [idx, point] of landmarkerResult.faceLandmarks[0].entries()) {
                        if (this.displayLandmarkArray[idx]) {
                            overlayCtx.setTransform({a: -1, e: overlayCanvas.width})
                            overlayCtx.beginPath();
                            overlayCtx.arc(point.x * overlayCanvas.width, point.y * overlayCanvas.height, 3, 0, 2 * Math.PI);
                            overlayCtx.fill();
                        }
                    }


                    features = await post_landmarks(landmarkFeatures, GazeDetector.fromScreenCoords(this_.target));
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

    private displayLandmarkArray: boolean[] = [];
    constructor(videoCaptureElement: HTMLVideoElement, landmarkDisplaySelectorElement: HTMLDivElement | undefined) {
        super();


        this.landmarkDetector = new LandMarkDetector(videoCaptureElement);

        const plotsDiv: HTMLDivElement = document.querySelector('#plots') as HTMLDivElement;


        this.videoCaptureElement = videoCaptureElement;


        const lossDisplayElement = <ScopeElement>document.querySelector('#loss-display');
        const scope: Scope = lossDisplayElement.Scope;
        scope.AutoYAxisAdjustChannel = 0;
        scope.AutoScaleY = true;
        scope.DataHeight = 25;
        scope.DataWidth = 600;
        scope.DataY = 0;
        scope.SignalFollowBehaviour = SignalFollowBehaviour.Fit;
        scope.SampleUnitMultiplier = 1/10000;
        scope.TimeAxisFormat = TimeAxisFormat.Seconds;

        scope.ChannelInfos = [
            {
                Color: '#003aa9',
                Name: 'Loss',
                Visible: true,
                RenderStyle: RenderStyle.Step,
                LineWidth: 10
            },
            {
                Color: '#446666',
                Name: 'X Loss',
                Visible: true,
                RenderStyle: RenderStyle.Step,
                LineWidth: 10
            },
            {
                Color: '#882c2c',
                Name: 'Y Loss',
                Visible: true,
                RenderStyle: RenderStyle.Step,
                LineWidth: 10
            }

        ]
        scope.BackColor = '#cedada';
        scope.ForeColor = '#2c2c2c'
        scope.TitleColor = '#2c2c2c'

        this.lossDisplayScope = scope;


        // scope.Connection = new PollingDataConnection(2.0, 1, () => [100 * this.training_loss])

        // scope.Connection = new PollingDataConnection(2.0, 1, async () => [100 * this.training_loss])

        const resizeObserver: ResizeObserver = new ResizeObserver((entries: ResizeObserverEntry[]) => {
                for (const scope_el of [lossDisplayElement]) {
                    for (const entry of entries) {
                        if (entry.contentRect) {
                            for (const s of [scope]) {
                                scope_el.Scope.Resize(entry.contentRect.width, Number.parseInt(scope_el.getAttribute('height') as string));
                            }
                        }
                    }}
            }

        );
        resizeObserver.observe(plotsDiv);

        const gazeElement = <GazeElement>document.createElement('gaze-element');
        const targetElement = <GazeElement>document.createElement('gaze-element');
        gazeElement.style.position = 'absolute';
        gazeElement.setBackground(
            `radial-gradient(#97dc81, #2f5609)`
        );
        gazeElement.setRadius(10, 10);
        targetElement.style.position = 'absolute';

        // targetElement.setBackground(
        //     "radial-gradient(#ffc42f, #570707)"
        // );
        targetElement.setRadius(25, 25);
        const this_ = this;
        targetElement.addEventListener('transitionend', () => {
            this_.target = this_.next_target
        });
        this.containerDiv = <HTMLDivElement>document.querySelector('.vidcap')
        this.imageCanvas = document.createElement('canvas');
        this.imageCtx = this.imageCanvas.getContext("2d") as CanvasRenderingContext2D;
        this.overlayCanvas = <HTMLCanvasElement>document.querySelector('#overlayCanvas')
        this.overlayCtx = this.overlayCanvas.getContext("2d") as CanvasRenderingContext2D;
        // Append in this order for z-order (gaze above target)
        document.documentElement.appendChild(targetElement);
        document.documentElement.appendChild(gazeElement);
        targetElement.style.left = '-10000px';
        targetElement.style.top = '-10000px';
        targetElement.style.transition = 'all 0.6s ease-out';
        this.targetElement = targetElement;

        let current_x = 0;
        let current_y = 0;
        this.on('GazeDetectionComplete', (features: iGazeDetectorResult) => {

            const momentum = 0.2;

            let x = features.gaze.x;
            if (x > screen.width - gazeElement.clientWidth)
                x =  screen.width - gazeElement.clientWidth;
            if (x < 0)
                x = 0;
            let y = features.gaze.y;
            if (y > screen.height - gazeElement.clientHeight)
                y =  screen.height - gazeElement.clientHeight;
            if (y < 0)
                y = 0;

            // Make a smooth transition between new and current gaze positions, so that the dot moves less jerkily
            current_x = Math.round(current_x + (x-current_x) * momentum);
            current_y = Math.round(current_y + (y-current_y) * momentum);

            gazeElement.style.left = current_x + 'px';
            gazeElement.style.top = current_y + 'px';
            let rx = features.losses.h_loss * screen.width;
            let ry = features.losses.v_loss * screen.height;
            if (rx < 15) rx = 15;
            if (ry < 15) ry = 15;
            if (rx > 100) rx = 100;
            if (ry > 100) ry = 100;

            targetElement.setRadius(rx , ry);
            targetElement.setCaption(`<div>${features.data_index}</div>`)

        });

        // Landmark selector
        if (landmarkDisplaySelectorElement) {
            const displayLandmarkArray = this.displayLandmarkArray;
            for (let i = 0; i < 478; ++i) {
                const checkBoxEl = document.createElement('input');
                checkBoxEl.type = 'checkbox';
                checkBoxEl.title = '' + i;
                displayLandmarkArray.push(false);
                checkBoxEl.addEventListener('click', (evt)=> {
                    const el = <HTMLInputElement>evt.target;
                    displayLandmarkArray[Number.parseInt(el.title)] = el.checked;
                })
                landmarkDisplaySelectorElement.appendChild(checkBoxEl);
            }

            for (const c of FaceLandmarker.FACE_LANDMARKS_FACE_OVAL) {
                const checkBoxEl = <HTMLInputElement>landmarkDisplaySelectorElement.children[c.start];
                checkBoxEl.checked = true;
                displayLandmarkArray[c.start] = true;
            }
        }
    }
}

