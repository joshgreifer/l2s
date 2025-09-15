/*
 * Application entry point. Sets up UI components, initializes the ONNX runtime,
 * and starts gaze detection when requested by the user.
 */
import './styles/main.css';
import { GazeElement } from "./GazeElement";
import { GazeSession } from "./controllers/GazeSession";
import { InputHandler } from "./services/InputHandler";
import { DataAcquisitionService } from "./services/DataAcquisitionService";

import "./util/index";
import { TabNavigator } from "./util/nav";
import { apiAvailable, getSavedGazeModel } from "./apiService";
import { Fullscreen } from "./util/util";
import { webOnnx } from "./runtime/WebOnnxAdapter";
import { trainableOnnx } from "./runtime/TrainableOnnx";
import { PixelCoord } from "./util/Coords";
import { ui } from "./UI";

let session: GazeSession | undefined;
let inputHandler: InputHandler | undefined;
let dataAcquisition: DataAcquisitionService | undefined;

function initUI() {
    customElements.define('gaze-element', GazeElement);

    TabNavigator.switchToPage('page-face');

    ui.startGazeDetectionButton.addEventListener("click", async () => {
        ui.startGazeDetectionButton.disabled = true;
        try {
            await Fullscreen(document.documentElement);
            session = new GazeSession();
            await session.Run();
            if (session.gazeDetector)
                dataAcquisition = new DataAcquisitionService(session.gazeDetector);
            inputHandler = new InputHandler(session, dataAcquisition!);
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
            dataAcquisition?.stop();
            inputHandler?.dispose();
        }

        ui.startGazeDetectionButton.disabled = !ready || (session !== undefined && session.GazeDetectionActive);
    }, 1000);
}

export async function bootstrap() {
    await Promise.all([
        webOnnx.init(getSavedGazeModel() ?? undefined),
        trainableOnnx.init(),
    ]);
    await webOnnx.predict([Array.from({ length: 478 }, () => [0, 0, 0] as PixelCoord)]);
    initUI();
    monitorApiStatus();
}

window.addEventListener('DOMContentLoaded', () => {
    bootstrap();
});

