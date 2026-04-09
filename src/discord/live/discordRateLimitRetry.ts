const MAX_RETRIES = 3;
const DEFAULT_WAIT_MS = 7_000;

function getRetryDelayMs(error: unknown): number | null {
    const message =
        error instanceof Error ? error.message : typeof error === "string" ? error : "";
    if (!message) {
        return null;
    }

    const secondsMatch = message.match(/retry after\s+([\d.]+)\s+seconds?/i);
    if (secondsMatch) {
        return Math.ceil(Number(secondsMatch[1]) * 1000);
    }

    const millisecondsMatch = message.match(/retry after\s+(\d+)\s*ms/i);
    if (millisecondsMatch) {
        return Number(millisecondsMatch[1]);
    }

    return /rate limited/i.test(message) ? DEFAULT_WAIT_MS : null;
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withDiscordRateLimitRetry<T>(
    operation: () => Promise<T>
): Promise<T> {
    let attempt = 0;
    let lastError: unknown;

    while (attempt <= MAX_RETRIES) {
        try {
            return await operation();
        } catch (error) {
            lastError = error;
            const delayMs = getRetryDelayMs(error);
            if (delayMs === null || attempt === MAX_RETRIES) {
                throw error;
            }

            await sleep(delayMs);
            attempt += 1;
        }
    }

    throw lastError;
}
