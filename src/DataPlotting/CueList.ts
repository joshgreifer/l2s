import {iDataConnection} from "../DataConnection";
import {Scope} from "./Scope";

class Cue {
 constructor(public start: number, public end: number, public Item: any) {
 }

}

export class CueList {
    private list_: Cue[] = [];

    constructor(private scope_: Scope) {}
    get Items() : Cue[] { return this.list_; }

    Range(start: number, duration: number): Cue[] {
        const end = start + duration;
        const cues: Cue[] = [];
        for (const cue of this.list_)
            if (cue.start >= start && cue.start < end )
                cues.push(cue);

        return cues;
    }

    add(item: any) {
        if (this.scope_.Connection) {
            const start = this.scope_.Connection.CurrentTimeSecs;
            this.list_.push(new Cue(start, start, item));
        }
    }

}
