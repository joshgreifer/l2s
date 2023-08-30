/*
 Color lookup table
 */


export class Clut {
    SZ:number;
    rgba:Uint32Array;
    r:Uint8Array;
    g:Uint8Array;
    b:Uint8Array;

    constructor(SZ:number, type : string, r :Uint8Array, g:Uint8Array, b:Uint8Array) {
        this.SZ = SZ;
        this.r = new Uint8Array(SZ+1);
        this.g = new Uint8Array(SZ+1);
        this.b = new Uint8Array(SZ+1);
        this.rgba = new Uint32Array(SZ+1);

        if (type == 'interpolate') {
            const f_inc = 1.0 / SZ;

            let f = 0.0;
            for (let i = 0; i < SZ; ++i) {
                this.r[i] = this.get_interpolated_val(r, f);
                this.g[i] = this.get_interpolated_val(g, f);
                this.b[i] = this.get_interpolated_val(b, f);
                this.rgba[i] = this.r[i] + this.g[i] * 0x100 + this.b[i] * 0x10000 + 0xff000000;
                f += f_inc;
            }
        }
        else if (type == 'repeat') {
            for (let i = 0; i < SZ; ++i) {
                this.r[i] = r[i % r.length];
                this.g[i] = g[i % g.length];
                this.b[i] = b[i % b.length];
                this.rgba[i] = this.r[i] + this.g[i] * 0x100 + this.b[i] * 0x10000 + 0xff000000;
            }
        }
        this.r[SZ] = this.r[SZ-1];
        this.g[SZ] = this.g[SZ-1];
        this.b[SZ] = this.b[SZ-1];
        this.rgba[SZ] = this.rgba[SZ-1];
    }

    private get_interpolated_val(a:Uint8Array, idx:number):number {

        const LAST_IDX = a.length - 1;
        if (idx >= 1.0) {
            return a[LAST_IDX];
        } else if (idx < 0.0)
            idx = 0.0;

        let fractional_idx = idx * LAST_IDX;

        const internal_idx = Math.floor(fractional_idx);
        fractional_idx -= internal_idx; // leave fractional part
        // linear interpolate
        return a[internal_idx] + (a[internal_idx + 1] - a[internal_idx]) * fractional_idx;
    }

    css_color(v: number, alpha: number = 1.0) {
        if (v < 0.0 || v > 1.0)
            throw new Error("clut.css_color: Parameter must be between 0 and 1.0");
        v *= this.SZ;
        v = Math.floor(v);
        return "rgba(" + this.r[v] +"," + this.g[v] +"," + this.b[v] + ", " + alpha + ")";
    }
}

export namespace Presets {
    export const MADNESS: Clut = new Clut(1024, 'interpolate',
        new Uint8Array([0, 0x77, 0xff, 0x11, 0xff, 0, 0xff, 0, 0xff, 0xff, 0x11, 0xff, 0, 0xff, 0x77, 0xff]),
        new Uint8Array([0, 0x00, 0x66, 0x99, 0xee, 0x66, 0, 0xff, 0xff, 0x66, 0x33, 0xee, 0x66, 0x00, 0xff, 0xff]),
        new Uint8Array([0, 0x11, 0x00, 0x00, 0x66, 0x99, 0xff, 0xee, 0xff, 0x88, 0x99, 0x66, 0x99, 0xff, 0x77, 0xff])
    );
    export const HEATMAP: Clut = new Clut(1024, 'interpolate',
        new Uint8Array([0, 0, 0, 0, 0, 0x7f, 0xff, 0xff, 0xff, 0x7f]),
        new Uint8Array([0, 8, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, 0, 0]),
        new Uint8Array([0x9f, 0xff, 0xff, 0xff, 0x7f, 0, 0, 0, 0, 0])
    );

    export const GRAYSCALE: Clut = new Clut(1024, 'interpolate',
        new Uint8Array([0, 0xff]),
        new Uint8Array([0, 0xff]),
        new Uint8Array([0, 0xff])
    );

    export const GRAYSCALE_INVERSE: Clut = new Clut(1024, 'interpolate',
        new Uint8Array([0xff, 0]),
        new Uint8Array([0xff, 0]),
        new Uint8Array([0xff, 0])
    );

    export const TRAFFIC_LIGHTS: Clut = new Clut(1024, 'interpolate',
        new Uint8Array([0xcf, 0xff, 0xff, 0]),
        new Uint8Array([0,    0x4f, 0xcf, 0x8f]),
        new Uint8Array([0,    0x0,  0x00,    0])
    );

}