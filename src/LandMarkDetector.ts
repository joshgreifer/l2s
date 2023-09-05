import vision, {BoundingBox} from "@mediapipe/tasks-vision";
const {FaceLandmarker, FilesetResolver, FaceDetector} = vision;

// Copied from vision.d.ts

interface Connection {
    start: number;
    end: number;
}

export interface LandmarkFeatures {
    face_oval: Number[][];
    left_eye: number[][];
    right_eye: number[][];
    left_iris: number[][];
    right_iris: number[][];
    nose: number[][];
    eye_blendshapes: number[];
}

export class LandMarkDetector {
    private faceLandmarker! : vision.FaceLandmarker;

    private faceDetector! : vision.FaceDetector;

    public Loaded: Promise<void>;

    constructor(private el: HTMLVideoElement) {
        this.Loaded = this.load();
    }

    public async GetLandmarks(): Promise<vision.FaceLandmarkerResult> {
        await this.Loaded;
        return this.faceLandmarker.detectForVideo(this.el, performance.now());
    }

    public async GetFaceBoundingBox(): Promise<BoundingBox | undefined> {
        await this.Loaded;
        const result = await this.faceDetector.detectForVideo(this.el, performance.now());
        return result.detections[0].boundingBox;
    }


    public static PackLandmarksIntoFeatureRect(result: vision.FaceLandmarkerResult)  {

        // Get subset of landmarks
        let indices = [];
        indices.push(...FaceLandmarker.FACE_LANDMARKS_FACE_OVAL.map((c) => c.start))
        indices.push(...FaceLandmarker.FACE_LANDMARKS_LEFT_EYE.map((c) => c.start))
        indices.push(...FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE.map((c) => c.start))
        // Nose-line and iris centers
        indices.push(...[0, 1, 2, 3, 473, 468])

        const landmarks = []
        for (let index of indices)
            landmarks.push (result.faceLandmarks[0][index])

        result.facialTransformationMatrixes
        let rows = new Array<number>(landmarks.length)

        landmarks.sort((a, b) => a.x - b.x)
        let min_x_diff = Number.MAX_VALUE;
        for (let i = 0; i < landmarks.length - 1; ++i)
            min_x_diff = Math.min(min_x_diff, landmarks[i + 1].x - landmarks[i].x)
         landmarks.sort((a, b) => a.y - b.y)
        let min_y_diff = Number.MAX_VALUE;
        for (let i = 0; i < landmarks.length - 1; ++i)
            min_x_diff = Math.min(min_x_diff, landmarks[i + 1].y - landmarks[i].y)



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

            let nose: number[][] = [];

            for (const c of [ 0, 1, 2, 3]) {
                const landmark = result.faceLandmarks[0][c]
                nose.push([landmark.x, landmark.y, landmark.z])
            }

            return {
                face_oval: face_oval,
                left_eye: left_eye,
                right_eye: right_eye,
                left_iris: left_iris,
                right_iris: right_iris,
                nose: nose,
                eye_blendshapes: eye_blendshapes
            }
        } catch(e) {
            return undefined;
        }

    }
    private async load()  {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        // Create a face landmarker
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
        // Create a face detector too, currently unused
        this.faceDetector = await FaceDetector.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
                delegate: "GPU"
            },
            runningMode: "VIDEO"
        });
    }
}