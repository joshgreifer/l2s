import {DataConnection} from "./DataConnection";


export class PollingDataConnection extends DataConnection {

    constructor(Fs: number, NumChannels: number, dataGenerator: ()=>number[]) {

        super(Fs, NumChannels, Float32Array);
        const data = new Float32Array(NumChannels);
        const interval_ms = 1000 / Fs;
        let t = window.performance.now();
        const poll = () => {
            const t2 = window.performance.now();
            if (t2 - t >= interval_ms) {
                t = t2;

                const data_as_array = dataGenerator();
                for (let c=0; c < data_as_array.length; ++c)
                    data[c] = data_as_array[c];
                this.AddData(data);
            }
            window.requestAnimationFrame(poll);
        }
        window.requestAnimationFrame(poll);
    }
}
