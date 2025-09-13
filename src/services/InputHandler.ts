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
      if (this.session.isTrainingActive) await this.session.StopTraining();
      else await this.session.StartTraining();
      notifications.notify(
        this.session.isTrainingActive
          ? "Calibration started."
          : "Calibration stopped."
      );
    }
  };

  public dispose() {
    window.removeEventListener("keyup", this.handleKeyboard);
  }
}
