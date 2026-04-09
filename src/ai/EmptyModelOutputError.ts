export class EmptyModelOutputError extends Error {
    public constructor(public readonly traceLabel: string) {
        super(`Model returned empty output for ${traceLabel}.`);
        this.name = "EmptyModelOutputError";
    }
}
