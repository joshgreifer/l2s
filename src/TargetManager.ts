import EventEmitter from "eventemitter3";
import {GazeElement} from "./GazeElement";
import {Coord} from "./util/Coords";

/**
 * Controls the on‑screen training target, emitting updates as it moves and
 * adjusting visual feedback based on trainer losses.
 */
export class TargetManager extends EventEmitter {
    private target_pos: Coord | undefined = undefined;
    private next_target_pos: Coord | undefined = undefined;
    private targetElement: GazeElement;

    constructor(targetElement: GazeElement) {
        super();
        this.targetElement = targetElement;
        this.targetElement.onReachedNewPosition(() => {
            this.target_pos = this.next_target_pos;
            this.emit('targetUpdated', this.target_pos);
        });
    }

    public set TargetPos(next_target_pos: Coord | undefined) {
        if (next_target_pos === undefined) {
            this.target_pos = undefined;
        }
        this.next_target_pos = this.targetElement.setPosition(next_target_pos);
    }

    public get TargetPos(): Coord | undefined {
        return this.next_target_pos;
    }

    public get CurrentTarget(): Coord | undefined {
        return this.target_pos;
    }

    public updateFeedback(features: {losses: {h_loss: number, v_loss: number}, data_index: number}): void {
        let rx = features.losses.h_loss * screen.width;
        let ry = features.losses.v_loss * screen.height;
        rx = rx.clamp(15, 100);
        ry = ry.clamp(15, 100);
        this.targetElement.setRadius(rx, ry);
        this.targetElement.setCaption(`<div>${features.data_index}</div>`);
    }
}

