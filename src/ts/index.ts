import {AddAlgorithms} from "./util/ArrayPlus";
// For diagnostics, export Luis response

import {Subject} from "./Subject";
import {Session} from "./Session";

import {GazeElement} from "./GazeDetector";

import {TabNavigator} from "./nav";
import {apiAvailable} from "./apiService";
import {enterFullscreenVideo} from "./util/util";
import {ScopeElement} from "./ScopeElement";

AddAlgorithms(Array.prototype);

customElements.define('gaze-element', GazeElement);
customElements.define('scope-element', ScopeElement);
// status fields and start button in UI

let statusDiv: HTMLDivElement;
let startGameButton!: HTMLButtonElement;
let indicatorApiAvailableEl: HTMLDivElement;


function setUpDomElementVars() {
    startGameButton = <HTMLButtonElement>document.querySelector("#startGameButton");

    statusDiv = <HTMLDivElement>document.querySelector("#statusDiv");

    indicatorApiAvailableEl = <HTMLDivElement>document.querySelector('.api-avail-indicator');



}



TabNavigator.switchToPage('page-face');


let subject: Subject | undefined = undefined;

(async () => {

    const doc = (<any>document);
    doc.fullscreenElement = doc.fullscreenElement || doc.mozFullscreenElement || doc.msFullscreenElement || doc.webkitFullscreenDocument;

    doc.exitFullscreen = doc.exitFullscreen || doc.mozExitFullscreen || doc.msExitFullscreen || doc.webkitExitFullscreen;

    setUpDomElementVars();


    startGameButton.addEventListener("click", async function () {

        startGameButton.disabled = true;

        try {
            await enterFullscreenVideo(document.documentElement);

            subject =  new Subject();
            await new Session(subject).Run();
        } catch (e) {
            // @ts-ignore
            window.alert(e.toString());
        }

        startGameButton.disabled = false;

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

        startGameButton.disabled = !ready || (subject !== undefined && subject.GazeDetectionActive);

    } )();

}, 1000);

