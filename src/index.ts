import { GazeElement } from "./GazeElement";
import { Subject } from "./Subject";
import { Session } from "./Session";

import "./util/index";
import { TabNavigator } from "./util/nav";
import { apiAvailable } from "./apiService";
import { Fullscreen } from "./util/util";
import { webOnnx } from "./runtime/WebOnnxAdapter";
import { PixelCoord } from "./util/Coords";
import { ui } from "./UI";

let subject: Subject | undefined;

function initUI() {
    customElements.define('gaze-element', GazeElement);

    TabNavigator.switchToPage('page-face');

    ui.startGazeDetectionButton.addEventListener("click", async () => {
        ui.startGazeDetectionButton.disabled = true;
        try {
            await Fullscreen(document.documentElement);
            subject = new Subject();
            await new Session(subject).Run();
        } catch (e) {
            // @ts-ignore
            window.alert((e as Error).toString());
        }
        ui.startGazeDetectionButton.disabled = false;
    });
}

function monitorApiStatus() {
    window.setInterval(async () => {
        const ready = await apiAvailable();
        if (ready) {
            ui.apiIndicator.classList.add('active');
        } else {
            ui.apiIndicator.classList.remove('active');
            if (subject) await subject.StopGazeDetection();
        }

        ui.startGazeDetectionButton.disabled = !ready || (subject !== undefined && subject.GazeDetectionActive);
    }, 1000);
}

export async function bootstrap() {
    await webOnnx.init();
    await webOnnx.predict(Array.from({ length: 478 }, () => [0, 0, 0] as PixelCoord));
    initUI();
    monitorApiStatus();
}

window.addEventListener('DOMContentLoaded', () => {
    bootstrap();
});

