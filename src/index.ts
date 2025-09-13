// For diagnostics, export Luis response

import {AppController} from "./AppController";
import {TabNavigator} from "./util/nav";
import {apiAvailable} from "./apiService";
import {Fullscreen} from "./util/util";
import {getDomRefs, DomRefs} from "./domRefs";

class App {
    private appController: AppController | undefined;
    private dom: DomRefs;
    private pollHandle: number;

    constructor() {
        this.dom = getDomRefs();
        this.appController = undefined;
        TabNavigator.switchToPage('page-face');

        this.dom.startGazeDetectionButton.addEventListener("click", async () => {
            this.dom.startGazeDetectionButton.disabled = true;
            try {
                await Fullscreen(document.documentElement);
                this.appController = new AppController();
                await this.appController.Run();
            } catch (e) {
                // @ts-ignore
                window.alert(e.toString());
            }
            this.dom.startGazeDetectionButton.disabled = false;
        });

        this.pollHandle = window.setInterval(async () => {
            const ready = await apiAvailable();
            if (ready)
                this.dom.indicatorApiAvailableEl.classList.add('active');
            else {
                this.dom.indicatorApiAvailableEl.classList.remove('active');
                if (this.appController) await this.appController.StopGazeDetection();
            }

            this.dom.startGazeDetectionButton.disabled =
                !ready || (this.appController !== undefined && this.appController.GazeDetectionActive);
        }, 1000);
    }
}

function init() {
    new App();
}

document.addEventListener('DOMContentLoaded', init);

