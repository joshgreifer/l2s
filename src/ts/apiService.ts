import {iGazeDetectorResult, iGazeDetectorTrainResult} from "./GazeDetector";
import {LandmarkFeatures} from "../LandMarkDetector";

export class HttpError extends Error {
    constructor(response: Response, public code= response.status) {
        super(response.statusText);
    }
}
interface iHttpServerError {
    exception: string;
    traceback: string[];
}
export class HttpServerError extends Error {
    constructor(serverError: iHttpServerError) {
        super(serverError.exception);
    }
}
export async function apiAvailable() : Promise<Boolean> {
    try {
        return (await fetch(`/api`)).status === 200;

    } catch (e) {
        return false;
    }
}
async function fetch_handling_server_error(input: RequestInfo, init: RequestInit | undefined) : Promise<Response> {
    let resp = await fetch(input, init);
    if (resp.status === 500) {
        const err = await resp.json() as iHttpServerError;
        throw new HttpServerError(err);
    }
    return resp;
}

export async function save_gaze_calibration() : Promise<boolean> {
    const api_response = await fetch(`/api/gaze/save`, {

        method: 'post',
        headers: {
            'Accept': 'application/json',
        }
    } );
    const res = await api_response.json();
    return res.status === 'success';

}


export async function train(epochs: number) : Promise<iGazeDetectorTrainResult> {


    const api_response = await fetch(`/api/gaze/train/${epochs}`, {

        method: 'post',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ epochs: epochs } )
    });
    const loss_json: iGazeDetectorTrainResult = await api_response.json();
    console.log("loss_json:", loss_json)
    return loss_json;

}

export async function post_landmarks(landmarks: LandmarkFeatures, target: {  x: number, y: number } | undefined = undefined) : Promise<iGazeDetectorResult | undefined> {

    let target_post = target ? { target_x: target.x.toString() , target_y: target.y.toString()} :  { target_x: "undefined", target_y: "undefined"}

    const landmarks_and_target = {
        landmarks: landmarks,
        target: target_post
    }

    const api_response = await fetch(`/api/gaze/landmarks`, {

        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(landmarks_and_target)
    });
    return await api_response.json();
}

