import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import {
    fallbackEvidenceDecision,
    fallbackStepDecision,
    guessPlan,
    planNextStep,
    planWithModel,
} from "@/runtime/planning";
import type { TurnInput } from "@/runtime/contracts";

function createInput(overrides: Partial<TurnInput> = {}): TurnInput {
    return {
        question: "test",
        user: { id: "u-requester" } as any,
        requesterDisplayName: "Requester",
        guild: { id: "g1" } as any,
        currentChannelId: "c1",
        nativeThreadId: null,
        requestedWebMode: "auto",
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
        ...overrides,
    };
}

describe("runtime planning", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("uses a generic conversation fallback when no guild is available", () => {
        const plan = guessPlan(
            createInput({
                guild: null,
                question: "oi sophia",
            })
        );

        expect(plan.mode).toBe("conversation");
        expect(plan.candidateCapabilities).toEqual([]);
    });

    it("keeps short reply follow-ups conversational in the generic fallback path", () => {
        const plan = guessPlan(
            createInput({
                question: "and that?",
                replyContext: {
                    messageId: "m1",
                    authorId: "u2",
                    authorName: "alice",
                    authorDisplayName: "Alice",
                    content: "I think the roadmap should slip by a week.",
                    jumpLink: null,
                },
            })
        );

        expect(plan.mode).toBe("conversation");
        expect(plan.candidateCapabilities).toEqual([]);
    });

    it("uses the generic current-guild research fallback when guild context is available", () => {
        const plan = guessPlan(createInput({ question: "what is happening here?" }));

        expect(plan.mode).toBe("research");
        expect(plan.candidateCapabilities).toEqual([
            "resolve_member_identity",
            "resolve_channel_targets",
            "retrieve_messages",
            "list_guild_structure",
        ]);
    });

    it("normalizes model planner output to valid capabilities only", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            mode: "research",
            reason: "Need current guild tools.",
            goal: "Answer the question.",
            successCriteria: "Use the relevant tools.",
            candidateCapabilities: [
                "resolve_member_identity",
                "not_a_real_tool",
                "retrieve_messages",
            ],
            confidence: "best_effort",
        } as any);

        const plan = await planWithModel(createInput({ question: "Quem sou eu?" }));

        expect(plan.mode).toBe("research");
        expect(plan.candidateCapabilities).toEqual([
            "resolve_member_identity",
            "retrieve_messages",
        ]);
    });

    it("forces exact member mentions through research even if the planner tries to stay conversational", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            mode: "conversation",
            reason: "Handle it directly.",
            goal: "Answer directly.",
            successCriteria: "Reply casually.",
            candidateCapabilities: [],
            confidence: "confident",
        } as any);

        const plan = await planWithModel(
            createInput({ question: "Quem é <@123456789012345678>?" })
        );

        expect(plan.mode).toBe("research");
        expect(plan.candidateCapabilities).toEqual([
            "resolve_member_identity",
            "retrieve_messages",
        ]);
    });

    it("forces exact channel mentions through research even if the planner returns no tools", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            mode: "research",
            reason: "Need a tool.",
            goal: "Answer directly.",
            successCriteria: "Use tools.",
            candidateCapabilities: [],
            confidence: "best_effort",
        } as any);

        const plan = await planWithModel(
            createInput({ question: "o que aconteceu em <#123456789012345678>?" })
        );

        expect(plan.mode).toBe("research");
        expect(plan.candidateCapabilities).toEqual([
            "resolve_member_identity",
            "resolve_channel_targets",
            "retrieve_messages",
            "list_guild_structure",
        ]);
    });

    it("forces named channel references through scoped channel research", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            mode: "conversation",
            reason: "Handle it directly.",
            goal: "Answer directly.",
            successCriteria: "Reply casually.",
            candidateCapabilities: [],
            confidence: "confident",
        } as any);

        const plan = await planWithModel(
            createInput({ question: "o que tem nesse canal who-riddle?" })
        );

        expect(plan.mode).toBe("research");
        expect(plan.candidateCapabilities).toEqual([
            "resolve_channel_targets",
            "retrieve_messages",
            "list_guild_structure",
        ]);
    });

    it("falls back to the generic current-guild plan when the planner model fails", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockRejectedValue(new Error("boom"));

        const plan = await planWithModel(createInput({ question: "Quem sou eu?" }));

        expect(plan.mode).toBe("research");
        expect(plan.candidateCapabilities).toEqual([
            "resolve_member_identity",
            "resolve_channel_targets",
            "retrieve_messages",
            "list_guild_structure",
        ]);
    });

    it("generic step fallback resolves exact member ids before broader retrieval", () => {
        const step = fallbackStepDecision({
            question: "<@123456789012345678> who is this?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            candidateCapabilities: [
                "resolve_member_identity",
                "resolve_channel_targets",
                "retrieve_messages",
            ],
            toolHistory: [],
        });

        expect(step).toEqual({
            nextCapability: "resolve_member_identity",
            arguments: { query: "123456789012345678" },
            reason: "Resolve the exact member reference before broader retrieval.",
            learnedExpectation:
                "Return the best current-guild identity match or historical fallback.",
        });
    });

    it("generic step fallback resolves exact channel ids before broader retrieval", () => {
        const step = fallbackStepDecision({
            question: "look in <#123456789012345678>",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            candidateCapabilities: [
                "resolve_channel_targets",
                "retrieve_messages",
                "list_guild_structure",
            ],
            toolHistory: [],
        });

        expect(step).toEqual({
            nextCapability: "resolve_channel_targets",
            arguments: { targetText: "123456789012345678" },
            reason: "Resolve the exact channel or category reference before broader retrieval.",
            learnedExpectation:
                "Return exact message-channel ids for the current guild target.",
        });
    });

    it("generic step fallback resolves named channel references before retrieval", () => {
        const step = fallbackStepDecision({
            question: "o que tem nesse canal who-riddle?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            candidateCapabilities: [
                "resolve_channel_targets",
                "retrieve_messages",
                "list_guild_structure",
            ],
            toolHistory: [],
        });

        expect(step).toEqual({
            nextCapability: "resolve_channel_targets",
            arguments: { targetText: "who-riddle" },
            reason: "Resolve the exact channel or category reference before broader retrieval.",
            learnedExpectation:
                "Return exact message-channel ids for the current guild target.",
        });
    });

    it("generic step fallback reuses named channel references from the replied message", () => {
        const step = fallbackStepDecision({
            question: "tenta de novo",
            actorId: "u-requester",
            replyContext: {
                messageId: "m1",
                authorId: "u-sophia",
                authorName: "Sophia",
                authorDisplayName: "Sophia",
                content: "Pois é, F4zke, o #atlas continua sendo um mistério por aqui!",
                jumpLink: null,
            },
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            candidateCapabilities: [
                "resolve_channel_targets",
                "retrieve_messages",
                "list_guild_structure",
            ],
            toolHistory: [],
        });

        expect(step).toEqual({
            nextCapability: "resolve_channel_targets",
            arguments: { targetText: "atlas" },
            reason: "Resolve the exact channel or category reference before broader retrieval.",
            learnedExpectation:
                "Return exact message-channel ids for the current guild target.",
        });
    });

    it("generic step fallback uses retrieve_messages before broader discovery when no exact references exist", () => {
        const step = fallbackStepDecision({
            question: "what is going on in the server?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            candidateCapabilities: [
                "resolve_member_identity",
                "resolve_channel_targets",
                "retrieve_messages",
                "list_guild_structure",
            ],
            toolHistory: [],
        });

        expect(step.nextCapability).toBe("resolve_channel_targets");
        expect(step.arguments).toMatchObject({
            targetText: "what is going on in the server?",
        });
    });

    it("falls back to the generic step ladder when the next-step model returns an invalid capability", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            nextCapability: "unknown_tool",
            arguments: {},
            reason: "bad",
            learnedExpectation: "bad",
        } as any);

        const step = await planNextStep({
            question: "what is going on in the server?",
            goal: "Answer the question.",
            successCriteria: "Use current guild tools.",
            confidence: "best_effort",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            candidateCapabilities: [
                "resolve_member_identity",
                "resolve_channel_targets",
                "retrieve_messages",
                "list_guild_structure",
            ],
            toolHistory: [],
        });

        expect(step.nextCapability).toBe("resolve_channel_targets");
    });

    it("prefers list_guild_structure after a category target is already resolved", () => {
        const step = fallbackStepDecision({
            question: "que serviços estão disponíveis nesse servidor?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: {
                query: "serviços",
                resolvedIds: ["c-bots", "c-logs"],
                entries: [],
                exactIdMatch: false,
                confidence: "high",
            },
            activeResolvedChannelIds: ["c-bots", "c-logs"],
            candidateCapabilities: [
                "resolve_channel_targets",
                "list_guild_structure",
                "retrieve_messages",
            ],
            toolHistory: [],
        });

        expect(step.nextCapability).toBe("list_guild_structure");
        expect(step.arguments).toMatchObject({
            targetText: "serviços",
        });
    });

    it("prefers scoped retrieval after structure has already been inspected", () => {
        const step = fallbackStepDecision({
            question: "que serviços estão disponíveis nesse servidor?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: {
                query: "serviços",
                resolvedIds: ["c-bots", "c-logs"],
                entries: [],
                exactIdMatch: false,
                confidence: "high",
            },
            activeResolvedChannelIds: ["c-bots", "c-logs"],
            candidateCapabilities: [
                "resolve_channel_targets",
                "list_guild_structure",
                "retrieve_messages",
            ],
            toolHistory: [
                {
                    tool: "list_guild_structure",
                    arguments: { targetText: "serviços" },
                    summary: "Resolved structure",
                    learned: "Serviços includes #bots and #logs",
                    confidenceImproved: true,
                    output: {
                        tool: "list_guild_structure",
                        summary: "Resolved structure",
                        data: {
                            query: "serviços",
                            entries: [],
                            focusedEntries: [],
                            focusedResolvedIds: ["c-bots", "c-logs"],
                        },
                    },
                    durationMs: 5,
                },
            ],
        });

        expect(step.nextCapability).toBe("retrieve_messages");
        expect(step.arguments).toMatchObject({
            query: "que serviços estão disponíveis nesse servidor?",
            channelIds: "c-bots,c-logs",
        });
    });

    it("normalizes model-selected retrieve_messages into channel resolution for named channel questions", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            nextCapability: "retrieve_messages",
            arguments: {},
            reason: "Read the messages directly.",
            learnedExpectation: "Find the relevant channel messages.",
        } as any);

        const step = await planNextStep({
            question: "o que tem nesse canal who-riddle?",
            goal: "Explain what is in who-riddle.",
            successCriteria: "Use grounded channel evidence.",
            confidence: "best_effort",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            candidateCapabilities: [
                "resolve_channel_targets",
                "retrieve_messages",
                "list_guild_structure",
            ],
            toolHistory: [],
        });

        expect(step).toEqual({
            nextCapability: "resolve_channel_targets",
            arguments: { targetText: "who-riddle" },
            reason: "Resolve the referenced channel or category before unscoped retrieval.",
            learnedExpectation:
                "Return exact message-channel ids for the named current-guild target.",
        });
    });

    it("treats direct requester resolution as sufficient evidence", () => {
        const sufficient = fallbackEvidenceDecision({
            question: "quem sou eu",
            actorId: "u-requester",
            candidateCapabilities: ["resolve_member_identity"],
            evidence: [],
            toolHistory: [
                {
                    tool: "resolve_member_identity",
                    arguments: { query: "u-requester" },
                    summary: "Resolved Requester",
                    learned: "Found requester profile",
                    confidenceImproved: true,
                    output: {
                        tool: "resolve_member_identity",
                        summary: "Resolved Requester",
                        data: {
                            resolvedId: "u-requester",
                            displayName: "Requester",
                            username: "req",
                            isCurrentGuildMember: true,
                        },
                    },
                    durationMs: 5,
                },
            ],
        });

        expect(sufficient).toEqual({
            sufficient: true,
            confidence: "confident",
            reason: "The requesting member was resolved directly.",
        });
    });

    it("keeps retrieval-based evidence insufficient when retrieval produced no messages", () => {
        const decision = fallbackEvidenceDecision({
            question: "what did alice say?",
            actorId: "u-requester",
            candidateCapabilities: ["retrieve_messages"],
            evidence: [],
            toolHistory: [
                {
                    tool: "retrieve_messages",
                    arguments: { query: "what did alice say?" },
                    summary: "No message evidence",
                    learned: "Nothing useful yet",
                    confidenceImproved: false,
                    output: {
                        tool: "retrieve_messages",
                        summary: "No message evidence",
                        data: {},
                    },
                    durationMs: 5,
                    retrievalSummary: {
                        cacheHit: false,
                        liveEscalated: true,
                        searchedChannelIds: [],
                        fetchedChannelIds: [],
                        cacheEnriched: false,
                        evidenceSufficient: false,
                        strongResultCount: 0,
                        weakResultCount: 0,
                        sourceOrigin: "live_refresh",
                    },
                },
            ],
        });

        expect(decision).toEqual({
            sufficient: false,
            confidence: "insufficient",
            reason: "Message retrieval did not produce usable message evidence yet.",
        });
    });
});
