import {DataConnection} from "./DataConnection";
import {train} from "./apiService";

export class ContinuousTrainer extends DataConnection {
    private stop_request: boolean = false;
    public Stop() { this.stop_request = true;  }

    async Start() {
        while (!this.stop_request) {
            let t = window.performance.now();
            const losses = await train(1, "calibrate");
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