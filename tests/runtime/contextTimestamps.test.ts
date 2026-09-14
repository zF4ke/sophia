import { expect, it } from "vitest";
import { formatChannelContext, formatEvidenceTime } from "@/runtime/planning";

it("preserves dates across old and recent channel context instead of implying both are current", () => {
    const text = formatChannelContext([
        { authorName: "Member", content: "Old message", createdTimestamp: Date.parse("2026-04-24T12:07:00Z") },
        { authorName: "Member", content: "New message", createdTimestamp: Date.parse("2026-09-09T20:00:00Z") },
    ]);
    expect(text).toContain("[2026-04-24T12:07:00.000Z] Member: Old message");
    expect(text).toContain("[2026-09-09T20:00:00.000Z] Member: New message");
    expect(formatEvidenceTime(undefined)).toBe("timestamp unavailable");
    expect(formatEvidenceTime(NaN)).toBe("timestamp unavailable");
});
