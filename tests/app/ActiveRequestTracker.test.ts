import { describe, expect, it } from "vitest";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";

describe("ActiveRequestTracker", () => {
    it("waits until every active request has released", async () => {
        const releaseFirst = ActiveRequestTracker.begin();
        const releaseSecond = ActiveRequestTracker.begin();
        let settled = false;
        const waiting = ActiveRequestTracker.waitForIdle(1_000).then((result) => {
            settled = true;
            return result;
        });

        releaseFirst();
        await Promise.resolve();
        expect(settled).toBe(false);

        releaseSecond();
        expect(await waiting).toBe(true);
    });

    it("makes release idempotent", async () => {
        const release = ActiveRequestTracker.begin();
        release();
        release();

        expect(await ActiveRequestTracker.waitForIdle(10)).toBe(true);
    });
});
