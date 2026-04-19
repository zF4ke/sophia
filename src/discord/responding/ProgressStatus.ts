import type { Message, TextBasedChannel } from "discord.js";

const THROTTLE_MS = 1_500;

/**
 * In-channel progress status for long turns.
 *
 * Sends a subtle `-# ⏳ …` message that gets edited in place as
 * work progresses. `finalize()` deletes the message best-effort.
 * Tolerates user-deleted status messages by recreating once.
 */
export interface ProgressStatus {
    /** Update the status line. Throttled to 1 edit / 1500ms. */
    notify(summary: string): Promise<void>;
    /** Delete the status message (best-effort). */
    finalize(): Promise<void>;
}

export class ProgressStatusService {
    public static startForChannel(channel: TextBasedChannel): ProgressStatus {
        let statusMessage: Message | null = null;
        let lastEditAt = 0;
        let pendingSummary: string | null = null;
        let pendingRawSummary: string | null = null;
        let recreateAttempted = false;
        let lastObservedSummary: string | null = null;
        let repeatedCount = 0;

        const formatSummary = (summary: string, count: number): string =>
            count > 1 ? `${summary} (x${count})` : summary;

        const sendOrEdit = async (summary: string): Promise<void> => {
            const formatted = `-# ⏳ ${summary}`;
            try {
                if (!statusMessage) {
                    if (!("send" in channel)) return;
                    statusMessage = await channel.send(formatted);
                    lastEditAt = Date.now();
                    recreateAttempted = false;
                    return;
                }
                await statusMessage.edit(formatted);
                lastEditAt = Date.now();
            } catch {
                // Message was deleted by someone — recreate once.
                if (!recreateAttempted) {
                    recreateAttempted = true;
                    statusMessage = null;
                    try {
                        if ("send" in channel) {
                            statusMessage = await channel.send(formatted);
                            lastEditAt = Date.now();
                        }
                    } catch {
                        // Give up silently.
                    }
                }
            }
        };

        return {
            async notify(summary: string) {
                if (summary === lastObservedSummary) {
                    repeatedCount += 1;
                } else {
                    lastObservedSummary = summary;
                    repeatedCount = 1;
                }
                const renderedSummary = formatSummary(summary, repeatedCount);
                const now = Date.now();
                if (now - lastEditAt < THROTTLE_MS) {
                    // Coalesce: keep latest summary, but don't edit yet.
                    pendingSummary = renderedSummary;
                    pendingRawSummary = summary;
                    return;
                }
                const toSend = pendingSummary && pendingRawSummary !== summary
                    ? pendingSummary
                    : renderedSummary;
                pendingSummary = null;
                pendingRawSummary = null;
                await sendOrEdit(toSend);
            },

            async finalize() {
                // Flush any remaining pending summary before deleting.
                if (pendingSummary && statusMessage) {
                    try {
                        await statusMessage.edit(`-# ⏳ ${pendingSummary}`);
                    } catch {
                        // Ignore — message may already be gone.
                    }
                    pendingSummary = null;
                    pendingRawSummary = null;
                }
                if (statusMessage) {
                    try {
                        await statusMessage.delete();
                    } catch {
                        // Best-effort.
                    }
                    statusMessage = null;
                }
            },
        };
    }
}
