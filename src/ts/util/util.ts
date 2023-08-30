import EventEmitter from "eventemitter3";


// export function GetRandomElementFrom<EL>( a: EL[]): EL
// {
//     const idx = Math.floor(Math.random() * a.length);
//     return a[idx];
// }

export function escapeXML(xml: string) {
    xml = xml.replace(/&/g, '&amp');
    xml = xml.replace(/</g, '&lt');
    xml = xml.replace(/>/g, '&gt');
    return xml;
}

export function stripXML(xml: string) {

    xml = xml.replace(/\n/g, '');    // no newlines
    xml = xml.replace(/<!--.*?-->/g, '');  // comment tags and everything within them
    xml = xml.replace(/<[^>]*?>/g, ' ');    // any SSML
    xml = xml.replace(/\ +/g, ' ');    // multiple spaces with single space
    xml = xml.trim();  // no leading or trailing space
    return xml;
}

export interface eventResult<T> {
    result : T | undefined;
    timeOut: boolean;
}

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

export async function exitFullscreenVideo(el:any) {
    if (document.fullscreenElement)  {
        await document.exitFullscreen();
    }
}

export async function enterFullscreenVideo(el:any) {
    const requestFullscreen =
        el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen || el.requestFullscreen;

    if (el && requestFullscreen)
        await requestFullscreen.call(el);
}

