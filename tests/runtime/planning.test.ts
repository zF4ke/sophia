import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import {
    fallbackEvidenceDecision,
    fallbackStepDecision,
    guessPlan,
    planNextStep,
    planWithModel,
    summarizeEvidence,
} from "@/runtime/planning";
import type { TurnInput, TurnIntent } from "@/runtime/contracts";

const NULL_INTENT: TurnIntent = {
    continuation: false,
    retrievalMode: null,
    beforeTimestamp: null,
    afterTimestamp: null,
    source: {
        continuation: "deterministic",
        retrievalMode: "none",
        timeBounds: "none",
    },
};

function intentWith(overrides: Partial<TurnIntent> = {}): TurnIntent {
    return { ...NULL_INTENT, ...overrides };
}

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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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

    it("generic step fallback resolves named member references before broader retrieval", () => {
        const step = fallbackStepDecision({
            question:
                "9 de fevereiro o openrosen mandou um link do youtube no canal de comandos, ele referiu de quem era?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
            candidateCapabilities: [
                "resolve_member_identity",
                "resolve_channel_targets",
                "retrieve_messages",
            ],
            toolHistory: [],
        });

        expect(step).toEqual({
            nextCapability: "resolve_member_identity",
            arguments: { query: "openrosen" },
            reason: "Resolve the exact member reference before broader retrieval.",
            learnedExpectation:
                "Return the best current-guild identity match or historical fallback.",
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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
            channelIds: ["c-bots", "c-logs"],
        });
    });

    it("skips structure inspection for forensic date/member lookups and retrieves immediately", () => {
        const step = fallbackStepDecision({
            question:
                "9 de fevereiro o openrosen mandou um link no canal comandos. ele referiu de quem era?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: {
                query: "openrosen",
                resolvedId: "u-open",
                displayName: "Openrosen",
                username: "oneperson",
                globalName: null,
                nickname: null,
                isBot: false,
                isCurrentGuildMember: true,
                source: "live_search",
                confidence: "high",
                roles: [],
            },
            activeChannelTarget: {
                query: "comandos",
                resolvedIds: ["c-comandos"],
                entries: [],
                exactIdMatch: false,
                confidence: "high",
            },
            activeResolvedChannelIds: ["c-comandos"],
            activeRetrievalSession: null,
            turnIntent: intentWith({
                beforeTimestamp: Date.parse("2026-02-10T00:00:00Z"),
                afterTimestamp: Date.parse("2026-02-09T00:00:00Z"),
            }),
            candidateCapabilities: [
                "resolve_member_identity",
                "list_guild_structure",
                "retrieve_messages",
            ],
            toolHistory: [
                {
                    tool: "resolve_member_identity",
                    arguments: { query: "openrosen" },
                    summary: "resolved",
                    learned: "resolved",
                    confidenceImproved: true,
                    output: { tool: "resolve_member_identity", summary: "resolved", data: {} },
                    durationMs: 1,
                },
            ],
        });

        expect(step.nextCapability).toBe("retrieve_messages");
        expect(step.arguments).toMatchObject({
            channelIds: ["c-comandos"],
            authorId: "u-open",
            afterTimestamp: Date.parse("2026-02-09T00:00:00Z"),
            beforeTimestamp: Date.parse("2026-02-10T00:00:00Z"),
        });
    });

    it("continues the active retrieval session with stable until-yesterday bounds", () => {
        const step = fallbackStepDecision({
            question: "continue",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: {
                query: "atlas",
                resolvedIds: ["c-atlas"],
                entries: [],
                exactIdMatch: false,
                confidence: "high",
            },
            activeResolvedChannelIds: ["c-atlas"],
            turnIntent: intentWith({ continuation: true }),
            activeRetrievalSession: {
                mode: "history",
                channelIds: ["c-atlas"],
                authorId: null,
                beforeTimestamp: 1_700_000_000_000,
                afterTimestamp: null,
                historyCursorByChannel: { "c-atlas": "101" },
                semanticCursor: null,
                seenMessageIds: ["105", "104"],
                accumulatedUniqueCount: 2,
                exhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                continuationAvailable: true,
            },
            candidateCapabilities: ["retrieve_messages"],
            toolHistory: [],
        });

        expect(step).toEqual({
            nextCapability: "retrieve_messages",
            arguments: {
                query: "continue",
                mode: "history",
                limit: 8,
                channelIds: ["c-atlas"],
                beforeTimestamp: 1_700_000_000_000,
                cursor: {
                    history: { "c-atlas": "101" },
                },
                excludedMessageIds: ["105", "104"],
            },
            reason: "Continue the active scoped history read without restarting from the beginning.",
            learnedExpectation: "Return the next non-duplicate page from the active retrieval session.",
        });
    });

    it("does not auto-apply cursor or exclusions on a fresh follow-up without continuation intent", () => {
        const step = fallbackStepDecision({
            question: "ele comentou de quem era?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: {
                query: "openrosen",
                resolvedId: "u-open",
                displayName: "Openrosen",
                username: "oneperson",
                globalName: null,
                nickname: null,
                isBot: false,
                isCurrentGuildMember: true,
                source: "live_search",
                confidence: "high",
                roles: [],
            },
            activeChannelTarget: {
                query: "comandos",
                resolvedIds: ["c-comandos"],
                entries: [],
                exactIdMatch: false,
                confidence: "high",
            },
            activeResolvedChannelIds: ["c-comandos"],
            turnIntent: intentWith({ continuation: false }),
            activeRetrievalSession: {
                mode: "history",
                channelIds: ["c-comandos"],
                authorId: "u-open",
                beforeTimestamp: Date.parse("2026-02-10T00:00:00Z"),
                afterTimestamp: Date.parse("2026-02-09T00:00:00Z"),
                historyCursorByChannel: { "c-comandos": "100" },
                semanticCursor: null,
                seenMessageIds: ["104", "103", "102"],
                accumulatedUniqueCount: 3,
                exhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                continuationAvailable: true,
            },
            candidateCapabilities: ["retrieve_messages"],
            toolHistory: [],
        });

        expect(step.nextCapability).toBe("retrieve_messages");
        expect(step.arguments).toMatchObject({
            query: "ele comentou de quem era?",
            channelIds: ["c-comandos"],
            authorId: "u-open",
        });
        expect(step.arguments).not.toHaveProperty("cursor");
        expect(step.arguments).not.toHaveProperty("excludedMessageIds");
    });

    it("escalates to get_member_profile when duplicate display names remain ambiguous", () => {
        const step = fallbackStepDecision({
            question: "qual glonos e o verdadeiro?",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: {
                query: "Glonos",
                resolvedId: "u-subjectless",
                displayName: "Glonos",
                username: "subjectless",
                globalName: null,
                nickname: null,
                isBot: false,
                isCurrentGuildMember: true,
                source: "live_search",
                confidence: "high",
                roles: [],
            },
            activeChannelTarget: null,
            activeResolvedChannelIds: [],
            turnIntent: NULL_INTENT,
            activeRetrievalSession: null,
            candidateCapabilities: ["resolve_member_identity", "list_members"],
            toolHistory: [
                {
                    tool: "resolve_member_identity",
                    arguments: { query: "Glonos" },
                    summary: "Glonos (@subjectless) resolved from the current guild.",
                    learned: "Glonos (@subjectless)",
                    confidenceImproved: true,
                    output: {
                        tool: "resolve_member_identity",
                        summary: "resolved",
                        data: {
                            query: "Glonos",
                            resolvedId: "u-subjectless",
                            displayName: "Glonos",
                            username: "subjectless",
                            isCurrentGuildMember: true,
                        },
                    },
                    durationMs: 5,
                },
                {
                    tool: "list_members",
                    arguments: { filters: "Glonos" },
                    summary: "2 members listed in join order.",
                    learned: "Glonos (@subjectless) | Glonos (@glonos)",
                    confidenceImproved: false,
                    output: {
                        tool: "list_members",
                        summary: "2 members listed in join order.",
                        data: {
                            members: [
                                {
                                    id: "u-subjectless",
                                    username: "subjectless",
                                    displayName: "Glonos",
                                },
                                {
                                    id: "u-glonos",
                                    username: "glonos",
                                    displayName: "Glonos",
                                },
                            ],
                        },
                    },
                    durationMs: 5,
                },
            ],
        });

        expect(step).toEqual({
            nextCapability: "get_member_profile",
            arguments: { nameOrId: "subjectless" },
            reason:
                "Multiple current-guild members share the same visible name; fetch profile details for each to disambiguate safely.",
            learnedExpectation:
                "Return a distinguishing profile for the ambiguous member so profiles can be compared.",
        });
    });

    it("resets the active retrieval session when the user changes scope mid-session", () => {
        const step = fallbackStepDecision({
            question: "continue in <#123456789012345678>",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: {
                query: "atlas",
                resolvedIds: ["c-atlas"],
                entries: [],
                exactIdMatch: false,
                confidence: "high",
            },
            activeResolvedChannelIds: ["c-atlas"],
            turnIntent: NULL_INTENT,
            activeRetrievalSession: {
                mode: "history",
                channelIds: ["c-atlas"],
                authorId: null,
                beforeTimestamp: null,
                afterTimestamp: null,
                historyCursorByChannel: { "c-atlas": "101" },
                semanticCursor: null,
                seenMessageIds: ["105", "104"],
                accumulatedUniqueCount: 2,
                exhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                continuationAvailable: true,
            },
            candidateCapabilities: ["resolve_channel_targets", "retrieve_messages", "list_guild_structure"],
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
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
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

    it("parses ISO-like timestamp strings from model step arguments into numeric bounds", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            nextCapability: "retrieve_messages",
            arguments: {
                beforeTimestamp: "2025-02-10T00:00:00Z",
                afterTimestamp: "2025-02-09T00:00:00Z",
            },
            reason: "Read bounded history.",
            learnedExpectation: "Find relevant rows within the requested window.",
        } as any);

        const step = await planNextStep({
            question: "o que aconteceu no canal comandos?",
            goal: "Find bounded message history.",
            successCriteria: "Use grounded evidence in the requested date window.",
            confidence: "best_effort",
            actorId: "u-requester",
            replyContext: null,
            activeMemberTarget: null,
            activeChannelTarget: {
                query: "comandos",
                resolvedIds: ["c-comandos"],
                entries: [],
                exactIdMatch: false,
                confidence: "high",
            },
            activeResolvedChannelIds: ["c-comandos"],
            activeRetrievalSession: null,
            turnIntent: NULL_INTENT,
            candidateCapabilities: ["retrieve_messages"],
            toolHistory: [],
        });

        expect(step.nextCapability).toBe("retrieve_messages");
        expect(step.arguments).toMatchObject({
            channelIds: ["c-comandos"],
            beforeTimestamp: Date.parse("2025-02-10T00:00:00Z"),
            afterTimestamp: Date.parse("2025-02-09T00:00:00Z"),
        });
    });

    it("includes ISO timestamps in summarized evidence when available", () => {
        const summary = summarizeEvidence({
            evidence: [
                {
                    tool: "retrieve_messages",
                    summary: "",
                    content: "example",
                    evidenceRole: "history_evidence",
                    strength: "strong",
                    sourceOrigin: "cache",
                    createdTimestamp: Date.parse("2025-02-09T10:11:12Z"),
                    channelName: "comandos",
                    authorName: "Openrosen",
                },
            ],
        });

        expect(summary).toContain("@2025-02-09T10:11:12.000Z");
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
                        mode: "history",
                        cacheHit: false,
                        liveEscalated: true,
                        searchedChannelIds: [],
                        fetchedChannelIds: [],
                        cacheEnriched: false,
                        evidenceSufficient: false,
                        strongResultCount: 0,
                        weakResultCount: 0,
                        historyMessageCount: 0,
                        semanticMatchCount: 0,
                    accumulatedUniqueCount: 0,
                    sourceOrigin: "live_refresh",
                    continuationAvailable: false,
                    historyContinuationAvailable: false,
                    historyCursorByChannel: {},
                    semanticContinuationAvailable: false,
                    semanticCursor: null,
                        exhaustedChannelIds: [],
                        historyExhausted: false,
                        semanticExhausted: false,
                        beforeTimestamp: null,
                        afterTimestamp: null,
                        activeChannelIds: [],
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
