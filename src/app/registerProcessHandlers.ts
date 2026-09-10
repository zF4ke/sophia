type ShutdownSignal = "SIGINT" | "SIGTERM" | "SIGUSR2";

export function registerProcessHandlers(shutdown: () => Promise<void>): void {
    let shutdownPromise: Promise<void> | null = null;

    const stop = (signal: ShutdownSignal) => {
        if (shutdownPromise) return;
        shutdownPromise = shutdown()
            .catch((error) => {
                console.error(`[shutdown] Cleanup failed after ${signal}.`, error);
            })
            .finally(() => {
                if (signal === "SIGUSR2") {
                    process.kill(process.pid, signal);
                    return;
                }
                process.exit(0);
            });
    };

    process.on("uncaughtException", (error) => {
        console.error("[process] Uncaught exception.", error);
    });
    process.on("unhandledRejection", (reason) => {
        console.error("[process] Unhandled rejection.", reason);
    });
    process.once("SIGINT", () => stop("SIGINT"));
    process.once("SIGTERM", () => stop("SIGTERM"));
    process.once("SIGUSR2", () => stop("SIGUSR2"));
}
