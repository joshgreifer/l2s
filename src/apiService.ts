import {
    BatchItem,
    iGazeDetectorAddDataResult,
    iGazeDetectorTrainResult
} from "./GazeDetector";
import { webOnnx } from "./runtime/WebOnnxAdapter";

let data_index = 0;

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
    if (webOnnx.ready)
        return true;
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

export async function save_gaze_model() : Promise<boolean> {
    const api_response = await fetch(`/api/gaze/save`, {

        method: 'post',
        headers: {
            'Accept': 'application/json',
        }
    } );
    const res = await api_response.json();
    return res.status === 'success';

}


export async function train(epochs: number, action: "train" | "calibrate") : Promise<iGazeDetectorTrainResult> {


    const api_response = await fetch(`/api/gaze/${action}/${epochs}`, {

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
export async function post_data(batch: BatchItem[]) : Promise<iGazeDetectorAddDataResult | undefined> {

    if (webOnnx.ready) {
        const last = batch[batch.length - 1];
        const [gx, gy] = await webOnnx.predict(last.landmarks);
        const [tx, ty] = last.target ?? [0, 0];
        const h_loss = Math.abs(gx - tx);
        const v_loss = Math.abs(gy - ty);
        const loss = (h_loss + v_loss) / 2;
        return {
            data_index: data_index++,
            gaze: { x: gx, y: gy },
            losses: { h_loss, v_loss, loss }
        };
    }
    console.log("Using Python backend, ONNX model not available.", batch);
    const api_response = await fetch(`/api/gaze/data`, {

        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(batch)
    });
    return await api_response.json();
}




