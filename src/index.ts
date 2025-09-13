import { GazeElement } from "./GazeElement";
import { GazeSession } from "./controllers/GazeSession";

import "./util/index";
import { TabNavigator } from "./util/nav";
import { apiAvailable } from "./apiService";
import { Fullscreen } from "./util/util";
import { webOnnx } from "./runtime/WebOnnxAdapter";
import { PixelCoord } from "./util/Coords";
import { ui } from "./UI";

let session: GazeSession | undefined;

function initUI() {
    customElements.define('gaze-element', GazeElement);

    TabNavigator.switchToPage('page-face');

    ui.startGazeDetectionButton.addEventListener("click", async () => {
        ui.startGazeDetectionButton.disabled = true;
        try {
            await Fullscreen(document.documentElement);
            session = new GazeSession();
            await session.Run();
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
            if (session) await session.StopGazeDetection();
        }

        ui.startGazeDetectionButton.disabled = !ready || (session !== undefined && session.GazeDetectionActive);
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

