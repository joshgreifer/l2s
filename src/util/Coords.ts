/*
Handle coordinate transformations and calculations
 */

export type Coord = { x: number; y: number; };
export type PixelCoord = [number, number, number];

export function modelToScreenCoords(point: Coord | PixelCoord): Coord {
    let x: number, y: number;
    if (Array.isArray(point)) {
        x = point[0];
        y = point[1];
    } else {
        x = point.x;
        y = point.y;
        x = ((x / 2) + 0.5) * screen.width;
        y = ((y / 2) + 0.5) * screen.height;
    }
    return { x: Math.round(x), y: Math.round(y) };
}

export function screenToModelCoords(point: Coord | undefined): Coord | undefined {
    if (!point) return undefined;
    let { x, y } = point;
    x = ((x / screen.width) - 0.5) * 2;
    y = ((y / screen.height) - 0.5) * 2;
    return { x, y };
}

