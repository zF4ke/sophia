import { describe, expect, it, vi, beforeEach } from "vitest";
import { shouldCompact, resolveCompactionProfile, compactMessages } from "@/runtime/compaction";
import { shouldCompactInput } from "@/runtime/inputCompaction";
import { SettingsService } from "@/app/SettingsService";
import { ModelGateway } from "@/ai/ModelGateway";
import type { ToolChatMessage } from "@/ai/ModelGateway";

// Mock SettingsService and ModelGateway.
vi.mock("@/app/SettingsService", () => ({
    SettingsService: {
        load: vi.fn(),
    },
}));

vi.mock("@/app/modelProfiles", () => ({
    readModelProfiles: () => ({
        defaultProfile: "test-cheap",
        profiles: {
            "test-cheap": {
                chatModel: "test/cheap-model",
                embeddingModel: "openai/text-embedding-3-small",
                temperature: 0.3,
                maxOutputTokens: 4096,
                contextWindow: 100000,
            },
        },
    }),
}));

vi.mock("@/ai/ModelGateway", () => ({
    ModelGateway: {
        generateText: vi.fn(),
    },
}));

function mockSettings(overrides: Record<string, unknown> = {}) {
    (SettingsService.load as ReturnType<typeof vi.fn>).mockReturnValue({
        compaction: {
            summarizerModel: "test-cheap",
            triggerFraction: 0.85,
            ...overrides,
        },
        runtime: {},
    });
}

function makeMessages(count: number): ToolChatMessage[] {
    const msgs: ToolChatMessage[] = [
        { role: "system", content: "System prompt" },
        { role: "user", content: "User question" },
    ];
    for (let i = 0; i < count; i++) {
        msgs.push({ role: "assistant", content: `Response ${i}` });
        msgs.push({ role: "user", content: `Follow-up ${i}` });
    }
    return msgs;
}

describe("shouldCompact", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("returns true when tokens >= triggerFraction * contextWindow", () => {
        mockSettings({ triggerFraction: 0.85 });
        expect(shouldCompact(8500, 10000)).toBe(true);
        expect(shouldCompact(9000, 10000)).toBe(true);
    });

    it("returns false when tokens < triggerFraction * contextWindow", () => {
        mockSettings({ triggerFraction: 0.85 });
        expect(shouldCompact(8400, 10000)).toBe(false);
        expect(shouldCompact(5000, 10000)).toBe(false);
    });

    it("respects custom triggerFraction", () => {
        mockSettings({ triggerFraction: 0.5 });
        expect(shouldCompact(5000, 10000)).toBe(true);
        expect(shouldCompact(4999, 10000)).toBe(false);
    });
});

describe("resolveCompactionProfile", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("returns the configured profile", () => {
        mockSettings();
        const profile = resolveCompactionProfile();
        expect(profile).not.toBeNull();
        expect(profile!.chatModel).toBe("test/cheap-model");
    });

    it("falls back to default when configured profile is missing", () => {
        mockSettings({ summarizerModel: "nonexistent-model" });
        const profile = resolveCompactionProfile();
        // Falls back to default profile (test-cheap).
        expect(profile).not.toBeNull();
        expect(profile!.chatModel).toBe("test/cheap-model");
    });
});

describe("compactMessages", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockSettings();
    });

    it("skips compaction when not needed (under trigger)", () => {
        mockSettings({ triggerFraction: 0.85 });
        const msgs = makeMessages(10);
        const result = compactMessages({
            messages: msgs,
            promptTokens: 1000,
            contextWindow: 10000,
        });
        return result.then((r) => {
            expect(r.compacted).toBe(false);
            expect(r.removedCount).toBe(0);
        });
    });

    it("skips compaction when not enough messages", () => {
        const msgs: ToolChatMessage[] = [
            { role: "system", content: "sys" },
            { role: "user", content: "q" },
        ];
        return compactMessages({
            messages: msgs,
            promptTokens: 9000,
            contextWindow: 10000,
        }).then((r) => {
            expect(r.compacted).toBe(false);
        });
    });

    it("compacts messages and preserves head + tail", async () => {
        (ModelGateway.generateText as ReturnType<typeof vi.fn>).mockResolvedValue(
            "Summary of the conversation so far.",
        );

        const msgs = makeMessages(10); // 2 + 20 = 22 messages
        const originalLength = msgs.length;

        const result = await compactMessages({
            messages: msgs,
            promptTokens: 9000,
            contextWindow: 10000,
        });

        expect(result.compacted).toBe(true);
        expect(result.removedCount).toBeGreaterThan(0);
        // Head preserved: [0]=system, [1]=user.
        expect(msgs[0].role).toBe("system");
        expect(msgs[0].content).toBe("System prompt");
        expect(msgs[1].role).toBe("user");
        expect(msgs[1].content).toBe("User question");
        // Summary inserted at index 2.
        expect(msgs[2].role).toBe("system");
        expect(msgs[2].content).toContain("<compaction_summary>");
        // Total messages should be much less.
        expect(msgs.length).toBeLessThan(originalLength);
    });

    it("handles ModelGateway failure gracefully", async () => {
        (ModelGateway.generateText as ReturnType<typeof vi.fn>).mockRejectedValue(
            new Error("API error"),
        );

        const msgs = makeMessages(10);
        const originalLength = msgs.length;

        const result = await compactMessages({
            messages: msgs,
            promptTokens: 9000,
            contextWindow: 10000,
        });

        expect(result.compacted).toBe(false);
        expect(msgs.length).toBe(originalLength); // Unchanged.
    });
});

describe("shouldCompactInput", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("returns false when both fraction and absoluteMax are zero/missing", () => {
        (SettingsService.load as ReturnType<typeof vi.fn>).mockReturnValue({
            compaction: { inputTriggerFraction: 0, inputMaxTokens: 0 },
        });
        expect(shouldCompactInput(50000, 100000)).toBe(false);
    });

    it("triggers on ratio-based ceiling for small context windows", () => {
        (SettingsService.load as ReturnType<typeof vi.fn>).mockReturnValue({
            compaction: { inputTriggerFraction: 0.4, inputMaxTokens: 0 },
        });
        expect(shouldCompactInput(40000, 100000)).toBe(true);
        expect(shouldCompactInput(39999, 100000)).toBe(false);
    });

    it("triggers on absolute ceiling even when ratio would not fire", () => {
        // 1M context window: ratio triggers at 400K, but absoluteMax triggers at 12K
        (SettingsService.load as ReturnType<typeof vi.fn>).mockReturnValue({
            compaction: { inputTriggerFraction: 0.4, inputMaxTokens: 12000 },
        });
        // 15K tokens — ratio says no (15K < 400K), but absolute says yes (15K >= 12K)
        expect(shouldCompactInput(15000, 1000000)).toBe(true);
        // 11K tokens — neither fires
        expect(shouldCompactInput(11000, 1000000)).toBe(false);
    });

    it("uses whichever ceiling fires first", () => {
        // Small context window: ratio triggers at 4K, absolute at 12K
        // Ratio should fire first here
        (SettingsService.load as ReturnType<typeof vi.fn>).mockReturnValue({
            compaction: { inputTriggerFraction: 0.4, inputMaxTokens: 12000 },
        });
        expect(shouldCompactInput(4000, 10000)).toBe(true);
        expect(shouldCompactInput(3999, 10000)).toBe(false);
    });

    it("returns false for invalid contextWindow", () => {
        (SettingsService.load as ReturnType<typeof vi.fn>).mockReturnValue({
            compaction: { inputTriggerFraction: 0.4, inputMaxTokens: 12000 },
        });
        expect(shouldCompactInput(15000, 0)).toBe(false);
        expect(shouldCompactInput(15000, -1)).toBe(false);
    });
});
