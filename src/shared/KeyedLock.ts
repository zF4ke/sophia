/** Serializes operations on one resource while unrelated resources remain available. */
export class KeyedLock {
    private readonly pending = new Map<string, Promise<unknown>>();
    async run<T>(key: string, operation: () => Promise<T>): Promise<T> {
        const previous = this.pending.get(key) ?? Promise.resolve();
        const work = previous.catch(() => {}).then(operation);
        this.pending.set(key, work);
        try { return await work; } finally { if (this.pending.get(key) === work) this.pending.delete(key); }
    }
}
