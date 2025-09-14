import EventEmitter from "eventemitter3";

import { GazeDetector } from "../GazeDetector";
import { save_gaze_model } from "../apiService";
import { ui } from "../UI";
import { Trainer, IGazeTrainer } from "../training/Trainer";

export class GazeSession extends EventEmitter {
  private isGazeDetectionActive = false;
  private trainer?: IGazeTrainer;
  private epoch = 0;

  public get GazeDetectionActive(): boolean {
    return this.isGazeDetectionActive;
  }

  public get isTrainingActive(): boolean {
    return this.trainer?.isTraining ?? false;
  }

  public gazeDetector?: GazeDetector;

  public async Run() {
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


  public async SaveGazeDetectorModel(): Promise<boolean> {
    try {
      return await save_gaze_model();
    } catch (err) {
      console.error("Error saving gaze model", err);
      return false;
    }
  }

  public async StartTraining() {
    if (this.trainer && !this.trainer.isTraining) {
      this.epoch = 0;
      this.trainer.startTraining();
    }
  }

  public async StopTraining() {
    if (this.trainer && this.trainer.isTraining) await this.trainer.stopTraining();
  }

  public async StartGazeDetection() {
    if (this.isGazeDetectionActive) return;
    this.isGazeDetectionActive = true;
    if (!this.gazeDetector) this.gazeDetector = new GazeDetector(ui.vidCap, ui.landmarkSelector);
    if (!this.trainer) {
      this.trainer = new Trainer();
      this.trainer.on("epoch", (n: number) => {
        this.epoch = n;
      });
    }
    this.gazeDetector.Trainer = this.trainer;

    const vidcap_overlay = ui.vidCapOverlay;

    this.gazeDetector.on("ProcessedFrame", () => {
      let text = `${this.gazeDetector!.FrameRate.toFixed(0)} FPS`;
      if (this.trainer?.isTraining) {
        text += ` Epoch ${this.epoch}`;
      }
      // @ts-ignore
      vidcap_overlay.innerText = text;
    });

    await this.gazeDetector.init();
  }
}
