declare global {
    interface Number {

        /**
         * Clamp a number to [min, max]
         * @param min
         * @param max
         */
        clamp(min: Number, max: Number): number

    }
}
export function AddProps(obj: Object = Number.prototype) {
Object.defineProperty(obj, 'clamp', {
        value:
            function (min: Number, max: Number): number {
                return this > max ? max : this < min ? min : this;
            }

    })
}
