// Core of this code comes from https://github.com/niklasvh/base64-arraybuffer/blob/master/lib/base64-arraybuffer.js




export declare type BufferData = Int8Array | Int16Array | Int32Array |
    Uint8Array | Uint16Array | Uint32Array |
    Float32Array | Float64Array | Uint8ClampedArray;

export declare type BufferDataConstructor = Int8ArrayConstructor | Int16ArrayConstructor | Int32ArrayConstructor |
    Uint8ArrayConstructor | Uint16ArrayConstructor | Uint32ArrayConstructor |
    Float32ArrayConstructor | Float64ArrayConstructor | Uint8ClampedArrayConstructor;

export declare type BufferTypeName = 'Int8Array' | 'Int16Array' | 'Int32Array' |
    'Uint8Array' | 'Uint16Array' | 'Uint32Array' |
    'Float32Array' | 'Float64Array' | 'Uint8ClampedArray';

export interface BufferDataStringified {
    dtype: BufferTypeName

    base64: string;
}
class _BufferUtils {

    public encode: (bufferData: BufferData) => BufferDataStringified;
    public decode: (stringified: BufferDataStringified) => BufferData;


    constructor() {

        const factories : { [Key: string] : BufferDataConstructor } = {
            'Buffer' : Uint8Array,
            'Int8Array' : Int8Array,
            'Uint8Array': Uint8Array,
            'Int16Array' : Int16Array,
            'Uint16Array': Uint16Array,
            'Int32Array': Int32Array,
            'Uint32Array': Uint32Array,
            'Float32Array': Float32Array,
            'Float64Array': Float64Array,
            'Uint8ClampedArray': Uint8ClampedArray,
        }

        const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const lookup = new Uint8Array(256);


        for (let i = 0; i < chars.length; i++) {
            lookup[chars.charCodeAt(i)] = i;
        }

        this.encode = (bufferData: BufferData): BufferDataStringified => {
            const arrayBuffer = bufferData.buffer;
            const dtype =  Object.getPrototypeOf(bufferData).constructor.name;
            const len = bufferData.byteLength;

            const bytes = new Uint8Array(arrayBuffer);

            const base64a = [];

            let j = 0;
            for (let i = 0; i < len; i += 3) {
                base64a[j++] = chars[bytes[i] >> 2];
                base64a[j++] = chars[((bytes[i] & 3) << 4) | (bytes[i + 1] >> 4)];
                base64a[j++] = chars[((bytes[i + 1] & 15) << 2) | (bytes[i + 2] >> 6)];
                base64a[j++] = chars[bytes[i + 2] & 63];
            }
            let base64 = base64a.join('');

            if ((len % 3) === 2) {
                base64 = base64.substring(0, base64.length - 1) + "=";
            } else if (len % 3 === 1) {
                base64 = base64.substring(0, base64.length - 2) + "==";
            }

            return { dtype: dtype, base64: base64 };
        }

        this.decode = (stringified: BufferDataStringified): BufferData => {

            const base64 = stringified.base64;
            let bufferLength = base64.length * 0.75;

            const len = base64.length;


            if (base64[base64.length - 1] === "=") {
                bufferLength--;
                if (base64[base64.length - 2] === "=") {
                    bufferLength--;
                }
            }

            const arraybuffer = new ArrayBuffer(bufferLength);
            const bytes = new Uint8Array(arraybuffer);

            let p = 0;
            for ( let i = 0; i < len;) {
                const encoded_0 = lookup[base64.charCodeAt(i++)];
                const encoded_1 = lookup[base64.charCodeAt(i++)];
                const encoded_2 = lookup[base64.charCodeAt(i++)];
                const encoded_3 = lookup[base64.charCodeAt(i++)];
                bytes[p++] = (encoded_0 << 2) | (encoded_1 >> 4);
                bytes[p++] = ((encoded_1 & 15) << 4) | (encoded_2 >> 2);
                bytes[p++] = ((encoded_2 & 3) << 6) | (encoded_3 & 63);
            }

            const factory = factories[stringified.dtype];

            return new factory(arraybuffer);
        }
      }

}

export const BufferUtils = new _BufferUtils();