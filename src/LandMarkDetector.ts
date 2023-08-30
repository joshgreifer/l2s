import vision from "@mediapipe/tasks-vision";
const {FaceLandmarker, FilesetResolver, DrawingUtils} = vision;

export interface LandmarkFeatures {
    face_oval: Number[][];
    left_eye: number[][];
    right_eye: number[][];
    left_iris: number[][];
    right_iris: number[][];
    eye_blendshapes: number[];
}

export class LandMarkDetector {
    private faceLandmarker! : vision.FaceLandmarker;

    public loaded: Promise<void>;

    constructor(private el: HTMLVideoElement) {
        this.loaded = this._load();
    }

    public async GetLandmarks(): Promise<vision.FaceLandmarkerResult> {
            await this.loaded;
            return this.faceLandmarker.detectForVideo(this.el, performance.now());
    }

    public static GetFeaturesFromLandmarks(result: vision.FaceLandmarkerResult) : LandmarkFeatures | undefined {

        try {
            let face_oval: number[][] = [];

            for (const c of FaceLandmarker.FACE_LANDMARKS_FACE_OVAL) {
                const landmark = result.faceLandmarks[0][c.start]
                face_oval.push([landmark.x, landmark.y, landmark.z])
            }

            let left_eye: number[][] = [];

            for (const c of FaceLandmarker.FACE_LANDMARKS_LEFT_EYE) {
                const landmark = result.faceLandmarks[0][c.start]
                left_eye.push([landmark.x, landmark.y, landmark.z])
            }
            let right_eye: number[][] = [];

            for (const c of FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE) {
                const landmark = result.faceLandmarks[0][c.start]
                right_eye.push([landmark.x, landmark.y, landmark.z])
            }

            let left_iris: number[][] = [];

            for (const c of FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS) {
                const landmark = result.faceLandmarks[0][c.start]
                left_iris.push([landmark.x, landmark.y, landmark.z])
            }
            let right_iris: number[][] = [];

            for (const c of FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS) {
                const landmark = result.faceLandmarks[0][c.start]
                right_iris.push([landmark.x, landmark.y, landmark.z])
            }

            let eye_blendshapes: number[] = [];
            const blendShapes = result.faceBlendshapes[0].categories;
            for (const c of blendShapes) {
                if (c.categoryName.startsWith("eye")) {
                    eye_blendshapes.push(c.score);
                }
            }
            return {
                face_oval: face_oval,
                left_eye: left_eye,
                right_eye: right_eye,
                left_iris: left_iris,
                right_iris: right_iris,
                eye_blendshapes: eye_blendshapes
            }
        } catch(e) {
            return undefined;
        }

    }
    public async _load()  {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        this.faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });
    }
}