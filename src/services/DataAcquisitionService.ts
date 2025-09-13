import { GazeDetector } from "../GazeDetector";

export class DataAcquisitionService {
  private targetTimeMs = 5000;
  private active = false;

  constructor(private gazeDetector: GazeDetector) {}

  public set TargetTimeMs(t: number) {
    this.targetTimeMs = t;
  }

  public get isActive(): boolean {
    return this.active && this.gazeDetector.TargetPos !== undefined;
  }

  public async start() {
    if (this.active) return;
    this.active = true;

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
    const y_positions: number[] = [
      0,
      0,
      0,
      0 / 2 - 50,
      screen.height / 2,
      screen.height / 2 + 50,
      screen.height,
      screen.height,
      screen.height,
    ];

    const center_x = screen.width / 2;
    const center_y = screen.height / 2 - 250;

    for (const offset of [-200, -100, 0, 0, 0, 100, 200]) {
      x_positions.push(center_x + offset);
      y_positions.push(center_y + offset);
    }
    const jitter = [-1, -2, -3, -4, 4, 3, 2, 1, 0];

    this.gazeDetector.TargetPos = { x: 0, y: 0 };

    const new_pos = () => {
      if (this.active && this.gazeDetector.TargetPos) {
        this.gazeDetector.TargetPos = {
          x: x_positions.randomElement() + jitter.randomElement(),
          y: y_positions.randomElement() + jitter.randomElement(),
        };
        setTimeout(new_pos, this.targetTimeMs);
      }
    };
    new_pos();
  }

  public async stop() {
    this.active = false;
    this.gazeDetector.TargetPos = undefined;
  }
}
