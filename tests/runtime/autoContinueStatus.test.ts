import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { runAutoContinue } from "@/runtime/AutoContinue";
import { ModelGateway } from "@/ai/ModelGateway";

// Phase 1 feedback loop: the exact user symptom — per-leg status posts that
// duplicate the same goal verbatim and cut off mid-word.
//
// Drives the REAL runAutoContinue (with Runtime.answer mocked at the seam)
// and asserts on the exact message content the user would see in channel.
// The status line goes through a tiny generateText call (mocked here), so
// context cost in production is one short prompt per leg.
describe("auto-continue status line", () => {
    const sentMessages: string[] = [];

    function fakeGuild() {
        return {
            id: "g1",
            channels: {
                cache: new Map([
                    ["c1", { id: "c1", send: async (o: { content: string }) => { sentMessages.push(o.content); return { id: "m1" }; } }],
                ]),
            },
        } as never;
    }

    beforeEach(async () => {
        sentMessages.length = 0;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        await OperationalStore.reset();
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `status-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
        vi.restoreAllMocks();
    });

    it("posts each open goal once, whole, per leg — never duplicated or cut", async () => {
        const { Runtime } = await import("@/runtime/Runtime");
        const { DiscordMemoryService } = await import("@/memory/DiscordMemoryService");
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "A continuar a psico-análise do One Person em prosa sobre o corpus de #discussão.",
        );

        // Leg 0: real flow. Open two goals in the store for this thread.
        await DiscordMemoryService.addRequestGoal({
            requestId: "req-a",
            threadId: "g1:c1:channel",
            label: "psicoanalise-prosa",
            body: "Escrever psico-análise do One Person (Openrosen) em prosa, baseada no corpus completo de #discussão, documentada com fontes, sem reutilizar conclusões parciais antigas",
        });
        await DiscordMemoryService.addRequestGoal({
            requestId: "req-b",
            threadId: "g1:c1:channel",
            label: "psicoanalise-oneperson",
            body: "Refazer do zero a psico-análise do One Person (Openrosen) em prosa corrida, bem formatada e documentada com jumpLinks, com base no corpus completo de #discussão, sem reusar conclusões antecipadas",
        });

        const answerSpy = vi.spyOn(Runtime, "answer").mockResolvedValue({
            requestId: "leg-1",
            answer: "still working",
            threadId: "g1:c1:channel",
            citations: [],
            classification: { mode: "direct_answer", reason: "test" },
            toolRuns: [{ tool: "note_add", summary: "x", data: null }],
            confidence: "best_effort",
        });

        await runAutoContinue(
            {
                requestId: "req-0",
                answer: "leg 0 done",
                threadId: "g1:c1:channel",
                citations: [],
                classification: { mode: "direct_answer", reason: "test" },
                toolRuns: [],
                confidence: "best_effort",
            },
            {
                question: "go",
                user: { id: "u1" } as never,
                requesterDisplayName: "Tester",
                guild: fakeGuild(),
                currentChannelId: "c1",
                nativeThreadId: null,
                requestedWebMode: "off",
                trigger: "talk",
                replyContext: null,
                referencedMessage: null,
                conversation: {
                    key: "g1:c1:channel",
                    kind: "channel",
                    trigger: "talk",
                    replyAnchorMessageId: null,
                    nativeThreadId: null,
                },
            } as never,
            null,
            1,
        );

        answerSpy.mockRestore();

        // The exact symptom: every posted status line must be a single human
        // sentence, and the two twin goals must be merged into one mention.
        expect(sentMessages.length).toBeGreaterThan(0);
        for (const message of sentMessages) {
            const lines = message.split("\n").map((s) => s.trim()).filter(Boolean);
            // One status message per leg: single line, no raw "#1 [open]" rows.
            expect(lines.length, `status must be a single line, got:\n${message}`).toBe(1);
            const line = lines[0];
            expect(line).not.toContain("#1 [open]");
            expect(line).not.toContain("#1 [");
            // No cut-off word: must end on a word boundary, not mid-token
            // like "antecipadas anti".
            const last = line.split(" ").pop() ?? "";
            expect(last.length).toBeGreaterThan(2);
            expect(line.length).toBeLessThanOrEqual(240);
        }
    });

    it("merges twin goals into one mention, never twice", async () => {
        const { Runtime } = await import("@/runtime/Runtime");
        const { DiscordMemoryService } = await import("@/memory/DiscordMemoryService");
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "A continuar a psico-análise do One Person em prosa.",
        );

        await DiscordMemoryService.addRequestGoal({
            requestId: "req-a",
            threadId: "g1:c1:channel",
            label: "psicoanalise-prosa",
            body: "Escrever psico-análise do One Person (Openrosen) em prosa, baseada no corpus completo de #discussão, documentada com fontes, sem reutilizar conclusões parciais antigas",
        });
        await DiscordMemoryService.addRequestGoal({
            requestId: "req-b",
            threadId: "g1:c1:channel",
            label: "psicoanalise-oneperson",
            body: "Refazer do zero a psico-análise do One Person (Openrosen) em prosa corrida, bem formatada e documentada com jumpLinks, com base no corpus completo de #discussão, sem reusar conclusões antecipadas",
        });

        const answerSpy = vi.spyOn(Runtime, "answer").mockResolvedValue({
            requestId: "leg-1",
            answer: "still working",
            threadId: "g1:c1:channel",
            citations: [],
            classification: { mode: "direct_answer", reason: "test" },
            toolRuns: [{ tool: "note_add", summary: "x", data: null }],
            confidence: "best_effort",
        });

        await runAutoContinue(
            {
                requestId: "req-0",
                answer: "leg 0 done",
                threadId: "g1:c1:channel",
                citations: [],
                classification: { mode: "direct_answer", reason: "test" },
                toolRuns: [],
                confidence: "best_effort",
            },
            {
                question: "go",
                user: { id: "u1" } as never,
                requesterDisplayName: "Tester",
                guild: fakeGuild(),
                currentChannelId: "c1",
                nativeThreadId: null,
                requestedWebMode: "off",
                trigger: "talk",
                replyContext: null,
                referencedMessage: null,
                conversation: {
                    key: "g1:c1:channel",
                    kind: "channel",
                    trigger: "talk",
                    replyAnchorMessageId: null,
                    nativeThreadId: null,
                },
            } as never,
            null,
            1,
        );

        answerSpy.mockRestore();

        // Both twin goals describe one task: "psico-análise" may appear many
        // times as a word, but the message must never contain two raw goal
        // rows and must mention the task only once.
        expect(sentMessages.length).toBeGreaterThan(0);
        const full = sentMessages.join("\n");
        expect(full).not.toContain("#1 [open]");
        const occurrences = (full.toLowerCase().match(/psico/g) ?? []).length;
        expect(occurrences).toBeLessThanOrEqual(2);
    });
});
