
/*
    An HTML element whose position on the screen represents either the user's estimated gaze position,
    or the target (label for training) of the gaze.
    Implemented as an elliptically-shaped DIV with an optional centered caption.
    Position setter/getter will set/get its center position.
    Aside from adding it to the DOM with create-element, it should not be manipulated directly by
    HTML/CSS  methods.
    Instead, its properties should be used.
 */
import {Coord, PixelCoord} from "./GazeDetector";

export class GazeElement extends HTMLElement {

    public setBackground: (css: string) => void;
    public setRadius: (rx: number, ry: number) => void;
    public setCaption: (text: string) => void;
    public setPosition: (coord: (Coord | undefined)) => (Coord | undefined);
    public setTransitionStyle: (css: string) => void;
    public onReachedNewPosition: (cb: () => any) => void;


    constructor() {
        super();
        const shadow = this.attachShadow({mode: 'open'}); // sets and returns 'this.shadowRoot'
        const el = <HTMLDivElement>document.createElement('div');
        const style = document.createElement('style');



        el.className = 'private-style1';
        // noinspection CssInvalidFunction,CssInvalidPropertyValue
        style.textContent = `
        .private-style1 {
            position: absolute;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            height: 20px;
            width: 20px;
            border-radius: 100%;
            background-image: radial-gradient(#ff0000, rgba(87,7,7,0));
            font-family: sans-serif;
            font-size: x-small;
        }
`;
        this.setBackground = (css: string) => el.style.background = css;
        this.setRadius = (rx: number, ry: number) => {
            rx = Math.round(rx);
            ry = Math.round(ry);
            el.style.width = 2 * rx + 'px';
            el.style.height = 2 * ry + 'px';
        }
        this.setCaption = (text: string) => el.innerHTML = text;

        this.setPosition = (coord: Coord | undefined) : Coord | undefined => {

            if (coord === undefined) {
                // move it off-screen
                el.style.left = '-10000px';
                el.style.top = '-10000px';
                return undefined;
            } else {
                const rx = Math.ceil(el.clientWidth / 2);
                const ry = Math.ceil(el.clientHeight / 2);
                let x = Math.floor(coord.x - rx);
                let y = Math.floor(coord.y - ry);

                // adjust x and y so that element doesn't overflow screen
                x = x.clamp(0, screen.width - 2 * rx);
                y = y.clamp(0, screen.height - 2 * ry);
                el.style.left = x + 'px';
                el.style.top = y + 'px';
                return {x: x, y: y}
            }
        }


        this.onReachedNewPosition = (cb: () => any ) => {
            el.addEventListener('transitionend', cb);
        }

        this.setTransitionStyle = (css: string) => { el.style.transition = css;}

        shadow.append(style, el);

    }
}

customElements.define('gaze-element', GazeElement);