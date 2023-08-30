/**
 * @file DataConnection.ts
 * @author Josh Greifer <joshgreifer@gmail.com>
 * @copyright © 2020 Josh Greifer
 * @licence
 * Copyright © 2020 Josh Greifer
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:

 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * @summary A multi-channel real-time data buffer, optimised for speed.
 * Internally uses ndarray views and implements a circular buffer.
 * When data is added (using .addData(),  the DataConnection emits a
 * 'data' event.   Subscribers to this event should then access the data in the buffer
 * using its Data() method, using start and end time offsets.
 *
 */

import NDArray from 'ndarray';
import EventEmitter from "eventemitter3";

export declare type BufferData = Int8Array | Int16Array | Int32Array |
    Uint8Array | Uint16Array | Uint32Array |
    Float32Array | Float64Array | Uint8ClampedArray;

export declare type BufferDataConstructor = Int8ArrayConstructor | Int16ArrayConstructor | Int32ArrayConstructor |
    Uint8ArrayConstructor | Uint16ArrayConstructor | Uint32ArrayConstructor |
    Float32ArrayConstructor | Float64ArrayConstructor | Uint8ClampedArrayConstructor;

class DataBuffer {
    private readonly NUM_CHANNELS: number;
    private readonly SZ: number;
    private readonly buf_:  BufferData;
    private idx_: number = 0;  // The offset into buffer of where to put new data
    private nBuffersRead_: number = 0;  // The number of times we've filled the buffer
    private sampleCount_: number = 0; // total number of data items read (grows past SZ)

    get RowCount(): number {
        return this.sampleCount_ / this.NUM_CHANNELS;
    }
    get BuffersRead(): number {
        return this.nBuffersRead_;
    }

    get Data(): BufferData { return this.buf_.subarray(0, this.sampleCount_ % this.SZ); }

    put(new_data: BufferData): void {
        // assert new_data.length % num_channels === 0
        const N = this.buf_.length / 2;
        const n = new_data.length;
        // assert (n <= N);
        let idx = this.idx_;
        this.buf_.set(new_data, idx);
        // if data has gone past end of 1st half of underlying buffer,
        // copy the overflowed data to start of the buffer (i.e. make it a circular buffer)
        const extra = (idx + n) - N;
        if (extra > 0) {
            this.buf_.set(new_data.subarray(n-extra), 0);
        }

        if ((idx += n) >= N) {
            ++this.nBuffersRead_;
            idx -= N;
        }
        this.idx_ = idx;
        this.sampleCount_ += n;

    }

    // t is sample_number, not time - return actual index into buffer at sample number t
    index(t: number): number { return (t % this.SZ)  * this.NUM_CHANNELS; }

    raw_view(t: number, n: number): BufferData {
        t %= this.SZ;
        t *=  this.NUM_CHANNELS;
        n *= this.NUM_CHANNELS;
        return this.buf_.subarray(t, t+n);
    }

    // returns view of sample buffer starting at time t
    view(t: number, n: number) :NDArray.NdArray {
        const N = this.SZ;

        const nd_array: NDArray.NdArray = NDArray(this.buf_, [2 * N, this.NUM_CHANNELS]);
        //       const idx = this.idx_ / this.NUM_CHANNELS;
        t %= N;
        //       t += idx;
        //        t -= this.StartCount;
        //        if (t+n < 0 || t >= this.SZ)
        //            return null;
        return nd_array.lo(t, 0);
    }

    reset(): void { this.nBuffersRead_ =  this.idx_ = this.sampleCount_ = 0; }

    constructor(sz: number, num_channels: number, /* bits_per_sample: number, */ array_constructor: BufferDataConstructor) {
        this.NUM_CHANNELS = num_channels;
        this.SZ = sz;
            this.buf_ = new array_constructor(2* sz * num_channels);

        this.reset();
    }
}


export interface iDataConnection extends EventEmitter
{
    readonly Fs: number;
    readonly NumChannels: number;
    readonly HasData: boolean;
    readonly CurrentTimeSecs: number
    readonly StartTimeSecs: number
    readonly NumFramesRead: number
    Reset(): void
    AddData(data: BufferData): void;
    Data(start_time_seconds: number, duration_seconds: number) : NDArray.NdArray;
    DataRaw() : BufferData;
    TimeToIndex(t: number) : number;
    ValueAtTime(t: number, channel?: number) : number;
}

export class DataConnection extends EventEmitter implements iDataConnection {

    public MeasuredFs: number = 0;

    protected static readonly BUFFER_SIZE_SECONDS = 3600;
    protected static readonly PERFORMANCE_MEASUREMENT_POLL_FREQ_SECONDS = 5;
    constructor(public readonly Fs: number, public readonly NumChannels: number, array_constructor: BufferDataConstructor, buffer_size_secs = DataConnection.BUFFER_SIZE_SECONDS) {
        super();
        this.buf_ = new DataBuffer(Math.round(buffer_size_secs * this.Fs), this.NumChannels, array_constructor);
    }

    AddData(data: BufferData): void {
        if (this.buf_.RowCount == 0) {
            this.emit('started');
        }


        this.buf_.put(data);
        if ((this.buf_.RowCount % Math.round(this.Fs * DataConnection.PERFORMANCE_MEASUREMENT_POLL_FREQ_SECONDS)) == 0)
            this.startPerformanceMeasurement()
        this.emit('data', data);

    }

    get HasData(): boolean { return this.buf_.RowCount > 0; }
    get CurrentTimeSecs(): number { return this.buf_.RowCount / this.Fs; }
    get StartTimeSecs(): number { return this.buf_.BuffersRead * DataConnection.BUFFER_SIZE_SECONDS; }
    get NumFramesRead(): number { return this.buf_.RowCount; }
    protected buf_: DataBuffer;


    Reset() {
        this.buf_.reset();
        this.emit('reset');

    }

    Data(start_time_seconds: number, duration_seconds: number) : NDArray.NdArray
    {
        return this.buf_.view(start_time_seconds * this.Fs, duration_seconds * this.Fs);
    }

    DataRaw() : BufferData
    {
        return this.buf_.Data;
    }

    FrameOffsetToIndex(frame_offset: number) : number { return this.buf_.index(frame_offset); }
    TimeToIndex(t: number) : number { return this.buf_.index(Math.round(t * this.Fs)); }

    ValueAtTime(t: number, channel: number = 0) {
        return this.Data(t, 1/this.Fs).pick(channel).get(-this.NumChannels);
    }

    perf_start_frames_read: number = 0;
    perf_start_time: number = 0;

    public startPerformanceMeasurement() {
        this.perf_start_time = window.performance.now();
        this.perf_start_frames_read = this.NumFramesRead;

    }
    public measurePerformance(): void {

        const now = window.performance.now();
        const frames_we_should_have_read = (now - this.perf_start_time) / 1000 * this.Fs;
        const frames_read = this.NumFramesRead - this.perf_start_frames_read;

        const speed = (frames_read / frames_we_should_have_read) || 1;



        // const  fs = speed * this.Fs;
        this.MeasuredFs = frames_read / ((now - this.perf_start_time) / 1000) ;

    }

}

