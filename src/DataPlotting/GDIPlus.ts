
/*****************************************************************************
 GDI+
 */
export type ColorOrGradient = string | CanvasGradient;


export enum DashStyle {
    Dash,
    DashDot,
    DashDotDot,
    Dot,
    Solid,
    Custom
}
export class Pen {
    Color: ColorOrGradient;
    DashPattern?: number[];
    Width: number;

    constructor( color: ColorOrGradient) {
        this.Color = color;
        this.Width = .5;
    }

}

export class Rect {

    public x: number;
    public y: number;
    public width: number;
    public height: number;

    public get right(): number {
        return this.x + this.width;
    }
    public set right(v: number) {
        this.width = v - this.x;
    }

    public get bottom(): number {
        return this.y + this.height;
    }
    public set bottom(v: number) {
        this.height = v - this.y;
    }

    public get Center(): Point { return { x: this.x + this.width /2, y: this.y + this.height /2}; }

    public Contains(x: number, y:number): boolean {
        return x >= this.x && y >= this.y && x <= this.right && y <= this.bottom;
    }
    public Equals(r: Rect): boolean {
        return r.x === this.x && r.y === this.y && r.width === this.width && r.height === this.height;
    }

    constructor(x: number, y: number, w: number, h: number) {
        this.x = x;
        this.y = y;
        this.width = w;
        this.height = h;
    }

    AssignFrom(r: Rect): void {
        this.x = r.x;
        this.y = r.y;
        this.width = r.width;
        this.height = r.height;
    }

    Clone(): Rect {
        return new Rect(this.x,this.y,this.width,this.height);
    }
}

export interface Point {
    x: number;
    y: number;

}

export class TransformMatrix {
    a!: number;
    b!: number;
    c!: number;
    d!: number;
    e!: number;
    f!: number;
}

export class GCH  {



    static setClip(ctx: CanvasRenderingContext2D, r: Rect) {
        ctx.rect(r.x, r.y, r.width, r.height);
        ctx.clip();
    }


    /*
    Convert between source rect and affine transformation matrix
    The destination matrix is
    a b 0
    c d 0
    e f 1
    Where e and f are x and y translation,

    and | a b |
        | c d |  is the scale/skew sub-matrix.
     */
    static GetWorldToGraphicTransform(graph_rect: Rect, world_rect: Rect) : TransformMatrix {
        const xscale = graph_rect.width / world_rect.width;
        const yscale = graph_rect.height / world_rect.height;

        return {
            a: xscale,
            b: 0.0,
            c: 0.0,
            d: yscale,
            e: graph_rect.x - (world_rect.x * xscale),
            f: graph_rect.y - (world_rect.y * yscale)
        };

    }


    static  ApplyTransform(ctx:CanvasRenderingContext2D, xform:TransformMatrix ) : void {
        ctx.setTransform(xform.a, xform.b, xform.c, xform.d, xform.e, xform.f);
    }

    static SetTransformRect(ctx:CanvasRenderingContext2D,  dest_rect: Rect, src_rect: Rect) : TransformMatrix {

        const xform: TransformMatrix = GCH.GetWorldToGraphicTransform(dest_rect, src_rect);
        GCH.ApplyTransform(ctx, xform);

        return xform;
    }

    static SetOrigin(ctx: CanvasRenderingContext2D, x: number, y: number ) {
        ctx.setTransform(1, 0, 0, 1, x, y);
    }

    static SetIdentityTransform(ctx: CanvasRenderingContext2D) {
        // Set to identity transform
        ctx.setTransform(1, 0, 0, 1, 0, 0);
    }

    static FillRectangleCoords(ctx: CanvasRenderingContext2D, color: ColorOrGradient, x: number, y: number, w: number, h: number) {
        ctx.fillStyle = color;
        ctx.fillRect(x, y, w, h);
    }

    static FillRectangle(ctx: CanvasRenderingContext2D, color: ColorOrGradient, r: Rect) {
        ctx.fillStyle = color;
        ctx.fillRect(r.x, r.y, r.width, r.height);
    }

    static FillEllipse(ctx: CanvasRenderingContext2D, color: ColorOrGradient, cx: number, cy: number, rx: number, ry: number) {
        const context = ctx;
        context.save(); // save state
        context.beginPath();

        context.translate(cx-rx, cy-ry);
        context.scale(rx, ry);
        context.arc(1, 1, 1, 0, 2 * Math.PI, false);

        context.restore(); // restore to original state
        //       context.stroke();
        context.fillStyle = color;
        context.fill();
    }

    static FillCircle(ctx: CanvasRenderingContext2D, color: ColorOrGradient, cx: number, cy: number, r: number) {
        const context = ctx;
        context.save(); // save state
        context.beginPath();

        context.translate(cx-r, cy-r);
        context.scale(r, r);
        context.arc(1, 1, 1, 0, 2 * Math.PI, false);

        context.restore(); // restore to original state
        //       context.stroke();
        context.fillStyle = color;
        context.fill();
    }
    static DrawCircle(ctx: CanvasRenderingContext2D, color: ColorOrGradient, cx: number, cy: number, r: number) {
        const context = ctx;
        context.save(); // save state
        context.beginPath();

        context.translate(cx-r, cy-r);
        context.scale(r, r);
        context.arc(1, 1, 1, 0, 2 * Math.PI, false);

        context.restore(); // restore to original state
        //       context.stroke();
        context.strokeStyle = color;
        context.stroke();
    }

    static DrawRectangleCoords(ctx: CanvasRenderingContext2D, pen: Pen, x: number, y: number, w: number, h: number) {
        const gc = ctx;
        gc.lineWidth = pen.Width;
        gc.strokeStyle = pen.Color;

        gc.setLineDash(pen.DashPattern || []);
        gc.strokeRect(x, y, w, h);
    }
    static DrawRectangle(ctx: CanvasRenderingContext2D,pen: Pen, r: Rect) {
        const gc = ctx;
        gc.lineWidth = pen.Width;
        gc.strokeStyle = pen.Color;
        gc.setLineDash(pen.DashPattern || []);
        gc.strokeRect(r.x, r.y, r.width, r.height);
    }

    static DrawString(ctx: CanvasRenderingContext2D, text: string, color: ColorOrGradient, x: number, y: number, alignment: TextAlign) {
        if (alignment === undefined) {
            alignment = { H: TextHorizontalAlign.Left, V:TextVerticalAlign.Bottom };
        } else {
            if (alignment.H === undefined)  {alignment.H = TextHorizontalAlign.Left; }
            if (alignment.V === undefined)  {alignment.V = TextVerticalAlign.Bottom; }
        }

        ctx.fillStyle = color;
        ctx.textAlign = alignment.H;
        ctx.textBaseline = alignment.V;
        ctx.fillText(text, x, y);
    }

    static DrawLine(ctx: CanvasRenderingContext2D, pen: Pen, x1: number, y1: number, x2: number, y2: number) {
        const gc = ctx;
        gc.strokeStyle = pen.Color;
        gc.lineWidth = pen.Width;
        gc.setLineDash(pen.DashPattern || []);
        gc.beginPath();
        gc.moveTo(x1, y1);
        gc.lineTo(x2, y2);
        gc.stroke();
    }



}
export class TextAlign  {
    H!: TextHorizontalAlign;
    V!: TextVerticalAlign;
}

export enum TextHorizontalAlign { Left = 'left', Right=  'right' , Center= 'center', Start= 'start' , End= 'end' }
export enum TextVerticalAlign {  Top='top', Hanging='hanging', Middle='middle', Alphabetic='alphabetic', Ideographic='ideographic', Bottom='bottom' }

