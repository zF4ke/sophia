export class ActiveRequestTracker {
    public static isIdle(): boolean { return this.activeCount === 0; }
    private static activeCount = 0;
    private static readonly idleWaiters = new Set<() => void>();

    public static begin(): () => void {
        this.activeCount += 1;
        let released = false;
        return () => {
            if (released) return;
            released = true;
            this.activeCount = Math.max(0, this.activeCount - 1);
            if (this.activeCount !== 0) return;
            for (const resolve of this.idleWaiters) resolve();
            this.idleWaiters.clear();
        };
    }

    public static async waitForIdle(timeoutMs: number): Promise<boolean> {
        if (this.activeCount === 0) return true;

        let timer: ReturnType<typeof setTimeout> | undefined;
        let resolveIdle: (() => void) | undefined;
        const idle = new Promise<true>((resolve) => {
            resolveIdle = () => resolve(true);
            this.idleWaiters.add(resolveIdle);
        });
        const timeout = new Promise<false>((resolve) => {
            timer = setTimeout(() => resolve(false), Math.max(1, timeoutMs));
            timer.unref?.();
        });

        try {
            return await Promise.race([idle, timeout]);
        } finally {
            if (timer) clearTimeout(timer);
            if (resolveIdle) this.idleWaiters.delete(resolveIdle);
        }
    }
}
