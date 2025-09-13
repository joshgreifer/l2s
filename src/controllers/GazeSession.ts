import EventEmitter from "eventemitter3";

import { GazeDetector } from "../GazeDetector";
import { save_gaze_model } from "../apiService";
import { ui } from "../UI";
import { notifications } from "../services/NotificationService";
import { Trainer, IGazeTrainer } from "../training/Trainer";

export class GazeSession extends EventEmitter {
  private isGazeDetectionActive = false;
  private trainer?: IGazeTrainer;

  public get GazeDetectionActive(): boolean {
    return this.isGazeDetectionActive;
  }

  public get isDataAcquisitionActive(): boolean {
    return this.gazeDetector !== undefined && this.gazeDetector.TargetPos !== undefined;
  }

  public get isTrainingActive(): boolean {
    return this.trainer?.isTraining ?? false;
  }

  private targetTimeMs = 5000;
  public set TargetTimeMs(t: number) {
    this.targetTimeMs = t;
  }

  public gazeDetector?: GazeDetector;

  private handleKeyboard = async (evt: KeyboardEvent) => {
    if (evt.key === "s") {
      notifications.notify("Saving model.");
      const success = await this.SaveGazeDetectorModel();
      notifications.notify(success ? "Saved model." : "Failed to save model.");
    } else if (evt.key === " ") {
      if (this.isDataAcquisitionActive) await this.StopDataAcquisition();
      else await this.StartDataAcquisition();
      notifications.notify(this.isDataAcquisitionActive ? "Data acquisition started." : "Data acquisition stopped.");
    } else if (evt.key === "c") {
      if (this.isTrainingActive) await this.StopTraining();
      else await this.StartTraining();
      notifications.notify(this.isTrainingActive ? "Calibration started." : "Calibration stopped.");
    }
  };

  public async Run() {
    window.addEventListener("keyup", this.handleKeyboard);
    await this.StartGazeDetection();
  }

  public async StopGazeDetection() {
    if (this.trainer) {
      await this.trainer.stopTraining();
    }
    if (this.gazeDetector) {
      this.gazeDetector.Trainer = undefined;
      await this.gazeDetector.term();
      this.gazeDetector = undefined;
    }
    this.trainer = undefined;
    this.isGazeDetectionActive = false;
  }

  public async StartDataAcquisition() {
    const this_ = this;
    const x_positions: number[] = [
      0,
      0,
      0,
      screen.width / 2 - 50,
      screen.width / 2 - 50,
      screen.width / 2 - 50,
      screen.width / 2,
      screen.width / 2 + 50,
      screen.width,
      screen.width,
      screen.width,
    ];
    const y_positions: number[] = [0, 0, 0, 0 / 2 - 50, screen.height / 2, screen.height / 2 + 50, screen.height, screen.height, screen.height];

    const center_x = screen.width / 2;
    const center_y = screen.height / 2 - 250;

    for (const offset of [-200, -100, 0, 0, 0, 100, 200]) {
      x_positions.push(center_x + offset);
      y_positions.push(center_y + offset);
    }
    const jitter = [-1, -2, -3, -4, 4, 3, 2, 1, 0];

    if (this.gazeDetector) this.gazeDetector.TargetPos = { x: 0, y: 0 };

    const new_pos = () => {
      if (this.gazeDetector && this_.isGazeDetectionActive && this.gazeDetector.TargetPos) {
        this.gazeDetector.TargetPos = {
          x: x_positions.randomElement() + jitter.randomElement(),
          y: y_positions.randomElement() + jitter.randomElement(),
        };
        setTimeout(new_pos, this_.targetTimeMs);
      }
    };
    new_pos();
  }

  public async StopDataAcquisition() {
    if (this.gazeDetector && this.isGazeDetectionActive) this.gazeDetector.TargetPos = undefined;
  }

  public async SaveGazeDetectorModel(): Promise<boolean> {
    return await save_gaze_model();
  }

  public async StartTraining() {
    if (this.trainer && !this.trainer.isTraining) this.trainer.startTraining();
  }

  public async StopTraining() {
    if (this.trainer && this.trainer.isTraining) await this.trainer.stopTraining();
  }

  public async StartGazeDetection() {
    if (this.isGazeDetectionActive) return;
    this.isGazeDetectionActive = true;
    if (!this.gazeDetector) this.gazeDetector = new GazeDetector(ui.vidCap, ui.landmarkSelector);
    if (!this.trainer) this.trainer = new Trainer();
    this.gazeDetector.Trainer = this.trainer;

    const vidcap_overlay = ui.vidCapOverlay;

    this.gazeDetector.on("ProcessedFrame", () => {
      // @ts-ignore
      vidcap_overlay.innerText = `${this.gazeDetector!.FrameRate.toFixed(0)} FPS`;
    });

    await this.gazeDetector.init();
  }
}
