import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import { detectStallPromise, type StallDetectionInput } from "@/runtime/stallGuard";
import type { ToolInvocationRecord } from "@/runtime/contracts";
import type { DiscordToolEvidenceRole } from "@/shared/discordTools";
import type { ToolEffect } from "@/tools/types";

function makeRecord(overrides: Partial<ToolInvocationRecord> = {}): ToolInvocationRecord {
    return {
        tool: "retrieve_messages",
        arguments: {},
        summary: "ok",
        learned: "ok",
        confidenceImproved: false,
        output: { tool: "retrieve_messages", summary: "ok", data: null, errorMessage: null },
        durationMs: 100,
        ...overrides,
    } as ToolInvocationRecord;
}

const effectMap: Record<string, ToolEffect> = {
    retrieve_messages: "read",
    search_messages: "read",
    clear_messages: "destructive",
    create_channel: "write",
};
const getToolEffect = (name: string) => effectMap[name];

function makeInput(overrides: Partial<StallDetectionInput> = {}): StallDetectionInput {
    return {
        answer: "Vou verificar isso agora",
        toolHistoryThisTurn: [],
        evidenceRoles: new Set<DiscordToolEvidenceRole>(),
        getToolEffect,
        ...overrides,
    };
}

/**
 * Mock the AI classifier. When `stall` is true the classifier says the
 * answer is a promise; when false it says the answer is real content.
 */
function mockClassifier(stall: boolean) {
    vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({ stall });
}

describe("stallGuard", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
    });

    describe("detectStallPromise", () => {
        it("detects a promise answer with no productive work", async () => {
            mockClassifier(true);
            const result = await detectStallPromise(makeInput({
                answer: "Vou dar uma olhada nos canais!",
            }));
            expect(result.stalled).toBe(true);
            expect(result.matchedPhrase).toBe("ai_classified_promise");
        });

        it("detects an English promise with no productive work", async () => {
            mockClassifier(true);
            const result = await detectStallPromise(makeInput({
                answer: "Let me check that for you!",
            }));
            expect(result.stalled).toBe(true);
        });

        it("does not stall when classifier says it's real content", async () => {
            mockClassifier(false);
            const result = await detectStallPromise(makeInput({
                answer: "O João disse isso ontem no #geral.",
            }));
            expect(result.stalled).toBe(false);
        });

        it("does not stall when answer is empty (no classifier call)", async () => {
            const result = await detectStallPromise(makeInput({ answer: "" }));
            expect(result.stalled).toBe(false);
            expect(ModelGateway.generateJson).not.toHaveBeenCalled;
        });

        it("redeems when a write tool succeeded (skips classifier)", async () => {
            const result = await detectStallPromise(makeInput({
                answer: "Vou criar o canal agora",
                toolHistoryThisTurn: [
                    makeRecord({ tool: "create_channel", blocked: false, output: { tool: "create_channel", summary: "Created", data: {}, errorMessage: null } }),
                ],
            }));
            expect(result.stalled).toBe(false);
        });

        it("redeems when a destructive tool succeeded (skips classifier)", async () => {
            const result = await detectStallPromise(makeInput({
                answer: "Vou limpar as mensagens",
                toolHistoryThisTurn: [
                    makeRecord({ tool: "clear_messages", blocked: false, output: { tool: "clear_messages", summary: "Cleared", data: {}, errorMessage: null } }),
                ],
            }));
            expect(result.stalled).toBe(false);
        });

        it("does not redeem when write tool was blocked", async () => {
            mockClassifier(true);
            const result = await detectStallPromise(makeInput({
                answer: "Vou criar o canal",
                toolHistoryThisTurn: [
                    makeRecord({ tool: "create_channel", blocked: true }),
                ],
            }));
            expect(result.stalled).toBe(true);
        });

        it("does not redeem when tool had an error", async () => {
            mockClassifier(true);
            const result = await detectStallPromise(makeInput({
                answer: "Let me search for that",
                toolHistoryThisTurn: [
                    makeRecord({
                        tool: "retrieve_messages",
                        blocked: false,
                        output: { tool: "retrieve_messages", summary: "Failed", data: null, errorMessage: "Error" },
                    }),
                ],
            }));
            expect(result.stalled).toBe(true);
        });

        it("redeems when productive evidence role is present (skips classifier)", async () => {
            const result = await detectStallPromise(makeInput({
                answer: "Vou procurar isso",
                evidenceRoles: new Set<DiscordToolEvidenceRole>(["message_evidence"]),
            }));
            expect(result.stalled).toBe(false);
        });

        it("does not redeem for discovery_only evidence role", async () => {
            mockClassifier(true);
            const result = await detectStallPromise(makeInput({
                answer: "Vou verificar",
                evidenceRoles: new Set<DiscordToolEvidenceRole>(["discovery_only"]),
            }));
            expect(result.stalled).toBe(true);
        });

        it("detects stall when longTaskGranted with prep tools but no evidence", async () => {
            mockClassifier(false); // Not a promise, but B2 path triggers
            const result = await detectStallPromise(makeInput({
                answer: "A análise do canal está pronta.",
                longTaskGranted: true,
                toolHistoryThisTurn: [
                    makeRecord({ tool: "retrieve_messages", output: { tool: "retrieve_messages", summary: "ok", data: null, errorMessage: null } }),
                ],
            }));
            expect(result.stalled).toBe(true);
            expect(result.matchedPhrase).toBe("long_task_without_evidence");
        });

        it("does not stall longTaskGranted when no tools were called", async () => {
            mockClassifier(false);
            const result = await detectStallPromise(makeInput({
                answer: "Tudo pronto.",
                longTaskGranted: true,
                toolHistoryThisTurn: [],
            }));
            expect(result.stalled).toBe(false);
        });

        it("does not stall longTaskGranted when productive evidence exists and classifier says no promise", async () => {
            mockClassifier(false);
            const result = await detectStallPromise(makeInput({
                answer: "Encontrei 200 mensagens relevantes.",
                longTaskGranted: true,
                evidenceRoles: new Set<DiscordToolEvidenceRole>(["message_evidence"]),
            }));
            expect(result.stalled).toBe(false);
        });

        it("stalls longTaskGranted when classifier detects promise even with evidence", async () => {
            mockClassifier(true);
            const result = await detectStallPromise(makeInput({
                answer: "Ainda estou no meio da coleta de dados. Assim que terminar te aviso.",
                longTaskGranted: true,
                evidenceRoles: new Set<DiscordToolEvidenceRole>(["message_evidence"]),
            }));
            expect(result.stalled).toBe(true);
        });

        it("does not stall longTaskGranted when write tool succeeded", async () => {
            const result = await detectStallPromise(makeInput({
                answer: "Canal criado com sucesso.",
                longTaskGranted: true,
                toolHistoryThisTurn: [
                    makeRecord({ tool: "create_channel", blocked: false, output: { tool: "create_channel", summary: "Created", data: {}, errorMessage: null } }),
                ],
            }));
            expect(result.stalled).toBe(false);
        });

        it("fails open when classifier throws an error", async () => {
            vi.spyOn(ModelGateway, "generateJson").mockRejectedValue(new Error("network error"));
            const result = await detectStallPromise(makeInput({
                answer: "I'll check that for you!",
            }));
            expect(result.stalled).toBe(false);
        });
    });
});
