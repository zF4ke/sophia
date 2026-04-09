export function registerProcessHandlers(): void {
    process.on("uncaughtException", (error) => {
        console.error(error.stack);
    });
}
