interface Waiting {
    owner: string; background: boolean; signal?: AbortSignal;
    run: () => void; cancel: () => void;
}

/** Capacity limits simultaneous requests, never a task's total work. */
export class ModelRequestQueue {
    private active = 0;
    private readonly owners = new Set<string>();
    private readonly waiting: Waiting[] = [];
    constructor(private readonly capacity: () => number) {}
    run<T>(owner: string, background: boolean, signal: AbortSignal | undefined, work: () => Promise<T>): Promise<T> {
        if (signal?.aborted) return Promise.reject(new Error("Model request cancelled before dispatch."));
        return new Promise<T>((resolve, reject) => {
            const item: Waiting = { owner, background, signal, cancel: () => {
                const index = this.waiting.indexOf(item);
                if (index < 0) return;
                this.waiting.splice(index, 1);
                signal?.removeEventListener("abort", item.cancel);
                reject(new Error("Model request cancelled before dispatch."));
                this.drain();
            }, run: () => {
                signal?.removeEventListener("abort", item.cancel);
                this.active++;
                this.owners.add(owner);
                void Promise.resolve().then(work).then(resolve, reject).finally(() => {
                    this.active--;
                    this.owners.delete(owner);
                    this.drain();
                });
            } };
            this.waiting.push(item);
            signal?.addEventListener("abort", item.cancel, { once: true });
            this.drain();
        });
    }
    private drain(): void {
        const capacity = this.capacity();
        while (this.active < capacity) {
            const eligible = (item: Waiting) => !this.owners.has(item.owner);
            let index = this.waiting.findIndex(item => !item.background && eligible(item));
            if (index < 0) index = this.waiting.findIndex(eligible);
            if (index < 0) return;
            this.waiting.splice(index, 1)[0].run();
        }
    }
}
