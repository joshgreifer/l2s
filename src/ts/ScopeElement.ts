/**
 * @file CustomElement.ts
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
 * @summary HTML Custom element wrapping a Scope.  It hooks window Resize() events
 * and delegates them to the Scope, so it can resize its internal canvases and data buffers.
 *
 */

import {Marker, Scope} from "../DataPlotting/Scope";

export class ScopeElement extends HTMLElement {

    private readonly scope_: Scope;

    public get Scope() : Scope { return this.scope_; }

    constructor() {
        super();
        const shadow = this.attachShadow({mode: 'open'}); // sets and returns 'this.shadowRoot'
        const el = <HTMLDivElement>document.createElement('div');
        const scope_el = <HTMLDivElement>document.createElement('div');
        const has_labels: boolean = this.hasAttribute('labels');
        const labels_el = has_labels ? <HTMLDivElement>document.createElement('div') : undefined;
        const marker_editor_dialog_el = <HTMLDialogElement>document.createElement('dialog');

        const height = this.hasAttribute('height') ? this.getAttribute('height') : '80vh';
        const width = this.hasAttribute('width') ? this.getAttribute('width') : '70vh';
        const title = this.hasAttribute('title') ? this.getAttribute('title') : '(unnamed)';

        marker_editor_dialog_el.innerHTML = `
            <form method="dialog">
                <section>
                <p>
                    <label for="label">Label:</label>
                    <input id="label" name="label" type="text">
                </p>
                </section>
                <menu>
                    <button id="cancel" type="reset">Cancel</button>
                    <button type="submit">Confirm</button>
                </menu>
            </form>
        `;

        // TODO: Get dialog working
        // Marker.editDialog = marker_editor_dialog_el;

        el.appendChild(marker_editor_dialog_el);
        el.appendChild(scope_el);
        if (labels_el) {
            el.appendChild(labels_el);
            labels_el.className = 'labels';
        }
        const style = document.createElement('style');
        // noinspection CssInvalidPropertyValue
        el.className = 'container';
        scope_el.className = 'plot';
        const scope = new Scope(scope_el, title || '(unnamed');
        // noinspection CssInvalidFunction,CssInvalidPropertyValue
        style.textContent = `

        .container {
            gap: 0;
            width: inherit;
            height: inherit;
            border: 1px solid rgba(170,170,170, 0.3);
            border-radius: 10px;
            margin: 0;
            overflow: hidden;
            display: grid;

            background-clip: content-box;
            grid-template-rows: 1fr ${labels_el ? 24 : 0}px;
            grid-template-columns: 1fr;
            grid-template-areas:
                "plot"
                "labels"
        }

        .labels {
            border-top: 1px solid rgb(71,71,71);
            background-color: ${scope.AxesBackColor};
            grid-area: labels;
            position: relative;
        }


        .cue {
            font-family: monospace;
            font-size: 12px;
            border-radius: 0 3px 3px 0;
            border-left: solid rgb(229,205,82);
            position: absolute;
            display: inline;
            top: 0px;
            padding: 5px;
            /*width: 100px;*/
            color: white;
            background-color: rgb(84,84,84);
        }

        .plot {
            grid-area: plot;
         }
            
`;

        this.scope_ = scope;

        const t2x = (t: number) => {
            return scope.d2gX(t) + scope.GraphBounds.x;
        };


        const d2w = (d: number) => {
            return scope.duration2pixels(d);
        }

        scope.on('reset', () => { if (labels_el) while (labels_el.firstChild) labels_el.removeChild(labels_el.firstChild); });

        scope.on('marker-added', (marker: Marker) => {
            if (labels_el) {
                const cue_el = <HTMLDivElement>document.createElement('div');
                marker.on('label-changed', (new_label) => {
                    cue_el.innerHTML = new_label;
                    cue_el.style.borderLeftColor = marker.color;
                });
                cue_el.dataset.time = '' + marker.time;
                cue_el.classList.add('cue');
                cue_el.innerHTML = marker.label;
                cue_el.style.left = t2x(marker.time) + 'px';
                cue_el.style.borderLeftColor = marker.color;
                labels_el.appendChild(cue_el);
            }

        });


        // Update the position of the cues when time axis changes
        scope.on('TimeAxisChanged', () => {
            const cue_els = el.querySelectorAll('.cue');
            for (const el of cue_els) {
                const cue_el = el as HTMLDivElement;
                cue_el.style.left = t2x(Number.parseFloat(cue_el.dataset.time as string)) + 'px';
            }

        });
        // Update the position of the cues when y axis changes
        // scope.on('YAxisChanged', () => {
        //     const cue_els = el.querySelectorAll('.cue');
        //     for (const el of cue_els) {
        //         const cue_el = el as HTMLDivElement;
        //         cue_el.style.top = t2y(Number.parseFloat(cue_el.dataset.time as string)) + 'px';
        //     }
        //
        // });
        shadow.append( style, el);
        // https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver


        // window.addEventListener('resize', (ev => {
        //     // this.scope_.Resize(240, 120 - (labels_el? 24: 0) )
        //     // scope_el.style.width = el.clientWidth + 'px';
        //
        //     this.scope_.Resize(el.clientWidth, el.clientHeight - (labels_el? 24: 0) )
        //     // if (labels_el)
        //     //     labels_el.style.width = el.clientWidth + 'px'
        //
        // }));
        this.scope_.Resize(el.clientWidth, el.clientHeight - (labels_el? 24: 0));

    }


}
