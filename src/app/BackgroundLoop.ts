/** One serial polling loop whose in-flight callback cannot restart it after stop. */
export class BackgroundLoop {
    private timer: ReturnType<typeof setTimeout> | undefined;
    private generation = 0;
    private active = false;
    private inFlight = false;
    start(run: () => Promise<void>, interval: () => number, report: (error: unknown) => void): void {
        if (this.active) return;
        this.active = true;
        const generation = ++this.generation;
        const schedule = () => {
            if (!this.active || generation !== this.generation) return;
            this.timer = setTimeout(async () => {
                this.timer = undefined;
                if (this.inFlight) { schedule(); return; }
                this.inFlight = true;
                try { await run(); } catch (error) { report(error); }
                finally { this.inFlight = false; }
                schedule();
            }, Math.max(1, interval()));
            this.timer.unref?.();
        };
        schedule();
    }
    stop(): void {
        this.active = false;
        this.generation++;
        if (this.timer) clearTimeout(this.timer);
        this.timer = undefined;
    }
}
