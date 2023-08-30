import EventEmitter from "eventemitter3";

/**
Error which is raised when a Promise returned by {@link waitOrTimeout} is rejected.
 */
export class TimeoutError extends Error {
    constructor(message: any) {
        super(message);
        this.name = 'TimeoutError';
    }
}
export class EventEmitterWithTimeout extends EventEmitter {

    /**
     *
     */
    static readonly WAIT_INFINITE = -1;

    constructor() {
        super();
    }

    /**
    Wait for an event to occur within a timeout period.
     @param eventName The name of the event to wait for.
     @param timeout_ms Number of milliseconds to wait. Use {@link WAIT_INFINITE}  for infinite wait
     @return  Promise<T> A Promise which will be resolved with the event's value if the event occurs within the wait period, or rejected with a {@link TimeoutError} if the timeout period expires before the event is fired.

     */
    waitOrTimeout<T>(eventName: string, timeout_ms: number) : Promise<T> {
        const self = this;
        // self.off(eventName);
        return new Promise<T>((resolve, reject) => {
            let waiting = true;
            const tid = timeout_ms === EventEmitterWithTimeout.WAIT_INFINITE ? 0 : setTimeout( () => {
                waiting = false;
                self.off(eventName, cb);
                reject(new TimeoutError(`waitOrTimeout timed out after ${timeout_ms}ms`));
            }, timeout_ms);

            const cb = (t: T) => {
                if (tid !== 0)
                    clearTimeout(tid);
                if (waiting)
                    resolve(t);
            };
            self.once(eventName, cb);

        });
    }


}