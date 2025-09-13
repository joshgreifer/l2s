

import {GazeElement} from "./GazeElement";
import {Subject} from "./Subject";
import {Session} from "./Session";

import "./util/index";
import {TabNavigator} from "./util/nav";
import {apiAvailable} from "./apiService";
import {Fullscreen} from "./util/util";
import * as util from "node:util";
import {webOnnx} from "./runtime/WebOnnxAdapter";
import { PixelCoord } from "./util/Coords";

customElements.define('gaze-element', GazeElement);
// status fields and start button in UI

let statusDiv: HTMLDivElement;
let startGazeDetectionButton!: HTMLButtonElement;
let indicatorApiAvailableEl: HTMLDivElement;


function setUpDomElementVars() {
    startGazeDetectionButton = <HTMLButtonElement>document.querySelector("#startGazeDetectionButton");

    statusDiv = <HTMLDivElement>document.querySelector("#statusDiv");

    indicatorApiAvailableEl = <HTMLDivElement>document.querySelector('.api-avail-indicator');



}



TabNavigator.switchToPage('page-face');


let subject: Subject | undefined = undefined;

(async () => {



    await webOnnx.init();
    await webOnnx.predict(Array.from({ length: 478 }, () => [0, 0, 0] as PixelCoord));

    setUpDomElementVars();


    startGazeDetectionButton.addEventListener("click", async function () {

        startGazeDetectionButton.disabled = true;

        try {
            await Fullscreen(document.documentElement);

            subject =  new Subject();
            await new Session(subject).Run();
        } catch (e) {
            // @ts-ignore
            window.alert(e.toString());
        }

        startGazeDetectionButton.disabled = false;

    });

})();

const pollForAPIHandle =window.setInterval(() => {

    (async () => {

        const ready = await apiAvailable();
        if (ready)
            indicatorApiAvailableEl.classList.add('active');
        else {
            indicatorApiAvailableEl.classList.remove('active');
            if (subject) await subject.StopGazeDetection();
        }

        startGazeDetectionButton.disabled = !ready || (subject !== undefined && subject.GazeDetectionActive);

    } )();

}, 1000);

