/*
 * Handles global keyboard shortcuts to control data capture, training,
 * and model saving during a gaze session.
 */
import { GazeSession } from "../controllers/GazeSession";
import { DataAcquisitionService } from "./DataAcquisitionService";
import { notifications } from "./NotificationService";

export class InputHandler {
  constructor(
    private session: GazeSession,
    private dataAcquisition: DataAcquisitionService
  ) {
    window.addEventListener("keyup", this.handleKeyboard);
  }

  private handleKeyboard = async (evt: KeyboardEvent) => {
    if (evt.key === "s") {
      notifications.notify("Saving model.");
      const success = await this.session.SaveGazeDetectorModel();
      notifications.notify(success ? "Saved model." : "Failed to save model.");
    } else if (evt.key === " ") {
      if (this.dataAcquisition.isActive) await this.dataAcquisition.stop();
      else await this.dataAcquisition.start();
      notifications.notify(
        this.dataAcquisition.isActive
          ? "Data acquisition started."
          : "Data acquisition stopped."
      );
    } else if (evt.key === "c") {
      if (this.session.isTrainingActive) {
        await this.session.StopTraining();
        notifications.notify("Calibration stopped.");
      } else {
        const started = this.session.StartTraining();
        notifications.notify(
          started
            ? "Calibration started."
            : "Not enough data to start training."
        );
      }
    }
  };

  public dispose() {
    window.removeEventListener("keyup", this.handleKeyboard);
  }
}
