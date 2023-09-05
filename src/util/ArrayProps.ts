declare global {
  interface Array<T> {

      /**
       * Return a random element from the array.
       */
    randomElement(): T;

       /**
       * XOR an array of numbers with another array of numbers.
        * @param k an array of number
       */
    xor(k: ArrayLike<number>) : Array<T>;

      /**
       * Shuffle an array in-place.
       */
    shuffle() : Array<T>;

      /**
       * Partition an array according to a partition function.
       * The partition function is of the form (elem: T, index?: number, array?: Array<T>) => string | number
       * and returns a key value for the item.
       * Returns an object whose properties are the partitioned keys.  The value of these properties
       * is an array of items belonging to the partition.
       * @example
            `const lookup: { [key: string]: string; } =
       {H: 'Hearts', D: 'Diamonds', S: 'Spades', C: 'Clubs' };
            ['4H', 'KD', '3S', 'AS', '9H'].partition( (e: string) => lookup[e[1]]);`

        { Hearts: ["4H", "9H"], Diamonds: ["KD"], Spades: ["3S", "AS"] }

       * @param filter A PartitionFunc
       */
    partition<T>(filter: PartitionFunc<T>) : Object;
  }
}


export type Partition<T> = {
    [Key in string | number]: Array<T>;
};
export type FilterFunc<T> = (item: T, index?: number, array?: Array<T>) => boolean
export type PartitionFunc<T> = (item: T, index?: number, array?: Array<T>) => string | number

export function AddProps(obj: Object = Array.prototype) {

    if (!('length' in obj))
        throw `Can't add array-props to ${typeof obj} because it has no 'length' property`;

    // @ts-ignore
    if (typeof obj[Symbol.iterator] === 'undefined')
        throw `Can't add array-props to ${typeof obj}  because it's not iterable`;

    Object.defineProperty(obj, 'randomElement', {
        value: function () {
            return this.length ? this[Math.floor(Math.random() * this.length)] : undefined;
        }
    });


    Object.defineProperty(obj, 'xor', {
        value:
            function (k: ArrayLike<number>) {
                const nk = k.length;

                for (let i = 0; i < this.length; ++i)
                    this[i] ^=  k[i % nk];
                return this;
            }
    })


    Object.defineProperty(obj, 'shuffle', {
        value: function () {
            // Knuth shuffle https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
            for (let i = this.length; i > 0;) {
                const j = Math.floor(Math.random() * i);
                --i;
                const tmp = this[i];
                this[i] = this[j];
                this[j] = tmp;
            }
            return this;
        }
    });

    Object.defineProperty(obj, 'partition', {
        value:
            function <T>(filter: PartitionFunc<T>) {
                const partitions: Partition<T> = {};
                let index = 0;
                for (const item of this) {
                    const k = filter(item, index++, this);
                    if (partitions[k] === undefined)
                        partitions[k] = [];
                    partitions[k].push(item);
                }
                return partitions;
            }
    })
}
