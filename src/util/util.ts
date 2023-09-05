import EventEmitter from "eventemitter3";


export class EventEmitterWithBellsOn extends EventEmitter {
    constructor() {
        super();
    }
    waitForEventWithTimeout<T>(evtName: string, timeout_ms: number) : Promise<T | undefined> {
        const emitter = this;
        return new Promise<T | undefined>((resolve, reject) => {
            const tid = setTimeout( () => {
                emitter.off(evtName, cb);
                reject({ result: undefined, timeOut: true })
            }, timeout_ms);

            const cb = (result: T | undefined) => {
                clearTimeout(tid);
                resolve(result)
            };
            emitter.once(evtName, cb);

        });
    }
}



export async function Fullscreen(el:any, on_off: boolean = true ) {
    const doc = (<any>document);
    doc.fullscreenElement = doc.fullscreenElement || doc.mozFullscreenElement || doc.msFullscreenElement || doc.webkitFullscreenDocument;

    doc.exitFullscreen = doc.exitFullscreen || doc.mozExitFullscreen || doc.msExitFullscreen || doc.webkitExitFullscreen;
    if (on_off) {
        const requestFullscreen =
            el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen || el.requestFullscreen;

        if (el && requestFullscreen)
            await requestFullscreen.call(el);
    } else {
        if (document.fullscreenElement)  {
            await document.exitFullscreen();
        }
    }
}
