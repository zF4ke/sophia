import { describe, expect, it } from "vitest";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";

function collectDisplayTextLength(value: unknown): number {
    if (Array.isArray(value)) {
        return value.reduce((total, item) => total + collectDisplayTextLength(item), 0);
    }
    if (!value || typeof value !== "object") {
        return 0;
    }

    return Object.entries(value as Record<string, unknown>).reduce((total, [key, entry]) => {
        if (key === "content" && typeof entry === "string") {
            return total + entry.length;
        }
        return total + collectDisplayTextLength(entry);
    }, 0);
}

describe("renderDebugTrace", () => {
    it("renders rich request, conversation, retrieval, and timeline sections", () => {
        const json = renderDebugTrace({
            questionPreview: "Quem decidiu isso no canal de produto?",
            status: "running",
            stage: "Running retrieve_messages",
            requesterLabel: "F4zke",
            trigger: "mention",
            classificationMode: "discord_grounded",
            runtimeMode: "research",
            selectedCapabilities: ["retrieve_messages", "get_member_profile"],
            toolCallCount: 2,
            groundingSummary: {
                messageEvidenceCount: 4,
                liveEvidenceCount: 0,
                sufficient: true,
            },
            retrievalSummary: {
                mode: "mixed",
                cacheHit: true,
                liveEscalated: false,
                searchedChannelIds: ["c1", "c2"],
                fetchedChannelIds: [],
                cacheEnriched: false,
                evidenceSufficient: true,
                strongResultCount: 2,
                weakResultCount: 1,
                historyMessageCount: 3,
                semanticMatchCount: 1,
                accumulatedUniqueCount: 4,
                sourceOrigin: "cache",
                continuationAvailable: true,
                historyContinuationAvailable: true,
                historyCursorByChannel: { c1: "101", c2: "202" },
                semanticContinuationAvailable: false,
                semanticCursor: null,
                exhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                beforeTimestamp: null,
                afterTimestamp: null,
                activeChannelIds: ["c1", "c2"],
            },
            groundedAnswerMode: "best_effort",
            stopReason: "evidence_sufficient",
            stopDetail: "Scoped message evidence is available.",
            checkpointThreadId: "g1:m-root:reply-chain",
            conversationContext: {
                threadId: "g1:m-root:reply-chain",
                kind: "reply_chain",
                replyAnchorMessageId: "m-root",
                replyContext: {
                    messageId: "m-root",
                    authorId: "u-alice",
                    authorName: "alice",
                    authorDisplayName: "Alice",
                    content: "I think the product team should wait for more feedback.",
                    jumpLink: "https://discord.com/channels/g1/c1/m-root",
                },
            },
            webStatus: "enabled",
            contextPreview: null,
            recentEvents: ["Running retrieve_messages", "Runtime mode: research"],
            timeline: [
                {
                    label: "tool_result",
                    detail: "retrieve_messages: found strong evidence",
                    tone: "success",
                    timestamp: Date.now(),
                },
            ],
            collapsedSections: {
                request: false,
                conversation: false,
                retrieval: false,
                context: false,
                timeline: false,
            },
            startedAt: Date.now() - 1000,
            failureMessage: null,
        }).map((container) => container.toJSON());

        const rendered = JSON.stringify(json);
        expect(rendered).toContain("Sophia Debug · Request");
        expect(rendered).toContain("Sophia Debug · Conversation");
        expect(rendered).toContain("Sophia Debug · Retrieval");
        expect(rendered).toContain("Sophia Debug · Timeline");
        expect(rendered).toContain("F4zke");
        expect(rendered).toContain("mention");
        expect(rendered).toContain("discord_grounded");
        expect(rendered).toContain("reply_chain");
        expect(rendered).toContain("m-root");
        expect(rendered).toContain("Alice");
        expect(rendered).toContain("retrieve_messages, get_member_profile");
        expect(rendered).toContain("Retrieval Origin");
        expect(rendered).toContain("Stop Reason");
        expect(rendered).toContain("History Cursor");
        expect(rendered).toContain("Web");
        expect(rendered).toContain("Channels Searched");
        expect(rendered).toContain("<#c1>");
        expect(rendered).toContain("Timeline");
        expect(rendered).toContain("Expand All");
        expect(rendered).toContain("Collapse All");
    });

    it("renders a failure section", () => {
        const json = renderDebugTrace({
            questionPreview: "teste",
            status: "failed",
            stage: "Failed",
            requesterLabel: "F4zke",
            trigger: "talk",
            classificationMode: null,
            runtimeMode: null,
            selectedCapabilities: [],
            toolCallCount: 0,
            groundingSummary: {
                messageEvidenceCount: 0,
                liveEvidenceCount: 0,
                sufficient: false,
            },
            retrievalSummary: null,
            groundedAnswerMode: null,
            stopReason: null,
            stopDetail: null,
            checkpointThreadId: null,
            conversationContext: {
                threadId: null,
                kind: null,
                replyAnchorMessageId: null,
                replyContext: null,
            },
            webStatus: null,
            contextPreview: null,
            recentEvents: ["Error: timeout"],
            timeline: [
                {
                    label: "error",
                    detail: "SQLITE_ERROR: timeout",
                    tone: "error",
                    timestamp: Date.now(),
                },
            ],
            collapsedSections: {
                request: false,
                conversation: false,
                retrieval: false,
                context: false,
                timeline: false,
            },
            startedAt: Date.now() - 1000,
            failureMessage: "SQLITE_ERROR: timeout",
        }).map((container) => container.toJSON());

        const rendered = JSON.stringify(json);
        expect(rendered).toContain("Failure");
        expect(rendered).toContain("SQLITE_ERROR: timeout");
    });

    it("renders a context preview panel when context is available", () => {
        const json = renderDebugTrace({
            questionPreview: "do que o One Person esta falando?",
            status: "completed",
            stage: "Completed",
            requesterLabel: "F4zke",
            trigger: "mention",
            classificationMode: "discord_grounded",
            runtimeMode: "research",
            selectedCapabilities: ["retrieve_messages"],
            toolCallCount: 1,
            groundingSummary: {
                messageEvidenceCount: 2,
                liveEvidenceCount: 0,
                sufficient: true,
            },
            retrievalSummary: null,
            groundedAnswerMode: "best_effort",
            stopReason: "evidence_sufficient",
            stopDetail: "Scoped message evidence is available.",
            checkpointThreadId: "g1:c1:channel",
            conversationContext: {
                threadId: "g1:c1:channel",
                kind: "channel",
                replyAnchorMessageId: null,
                replyContext: null,
            },
            webStatus: "enabled",
            contextPreview: {
                recentChannelMessages: [
                    "One Person: Ha uma nameplate que eu queria comprar",
                    "F4zke: imagina o tanto de chatgpt pros que vc podia comprar",
                    "One Person: E o pior e que ela uma das poucas que voce nao pode comprar via orb",
                ],
                evidencePreview: [
                    "[retrieve_messages] #reflexoes · One Person: Ha uma nameplate que eu queria comprar",
                ],
                recentTurns: [
                    "Q: oi | A: Oi! Estou por aqui.",
                ],
            },
            recentEvents: ["Completed"],
            timeline: [
                {
                    label: "Completed",
                    detail: "evidence_sufficient",
                    tone: "success",
                    timestamp: Date.now(),
                },
            ],
            collapsedSections: {
                request: false,
                conversation: false,
                retrieval: false,
                context: false,
                timeline: false,
            },
            startedAt: Date.now() - 2000,
            failureMessage: null,
        }).map((container) => container.toJSON());

        const rendered = JSON.stringify(json);
        expect(rendered).toContain("Sophia Debug · Context");
        expect(rendered).toContain("Channel Messages:");
        expect(rendered).toContain("One Person:");
        expect(rendered).toContain("Evidence:");
        expect(rendered).toContain("Prior Turns:");
        expect(rendered).toContain("#reflexoes");
    });

    it("omits context panel when contextPreview is null", () => {
        const json = renderDebugTrace({
            questionPreview: "oi",
            status: "completed",
            stage: "Completed",
            requesterLabel: "F4zke",
            trigger: "mention",
            classificationMode: "direct_answer",
            runtimeMode: "conversation",
            selectedCapabilities: [],
            toolCallCount: 0,
            groundingSummary: null,
            retrievalSummary: null,
            groundedAnswerMode: "confident",
            stopReason: "direct_answer",
            stopDetail: "Answered directly without entering the research loop.",
            checkpointThreadId: "g1:c1:channel",
            conversationContext: {
                threadId: "g1:c1:channel",
                kind: "channel",
                replyAnchorMessageId: null,
                replyContext: null,
            },
            webStatus: null,
            contextPreview: null,
            recentEvents: [],
            timeline: [],
            collapsedSections: {
                request: false,
                conversation: false,
                retrieval: false,
                context: false,
                timeline: false,
            },
            startedAt: Date.now() - 500,
            failureMessage: null,
        }).map((container) => container.toJSON());

        const rendered = JSON.stringify(json);
        expect(rendered).not.toContain("Sophia Debug · Context");
    });

    it("shows stop condition prominently and can collapse the timeline section", () => {
        const json = renderDebugTrace({
            questionPreview: "teste",
            status: "completed",
            stage: "Completed",
            requesterLabel: "F4zke",
            trigger: "reply",
            classificationMode: "discord_grounded",
            runtimeMode: "research",
            selectedCapabilities: ["resolve_channel_targets", "retrieve_messages"],
            toolCallCount: 2,
            groundingSummary: {
                messageEvidenceCount: 0,
                liveEvidenceCount: 3,
                sufficient: false,
            },
            retrievalSummary: {
                mode: "history",
                cacheHit: false,
                liveEscalated: true,
                searchedChannelIds: ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
                fetchedChannelIds: ["c1"],
                cacheEnriched: true,
                evidenceSufficient: false,
                strongResultCount: 0,
                weakResultCount: 0,
                historyMessageCount: 2,
                semanticMatchCount: 0,
                accumulatedUniqueCount: 2,
                sourceOrigin: "live_refresh",
                continuationAvailable: false,
                historyContinuationAvailable: false,
                historyCursorByChannel: { c1: "101" },
                semanticContinuationAvailable: false,
                semanticCursor: null,
                exhaustedChannelIds: ["c1"],
                historyExhausted: false,
                semanticExhausted: true,
                beforeTimestamp: null,
                afterTimestamp: null,
                activeChannelIds: ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
            },
            groundedAnswerMode: "best_effort",
            stopReason: "budget_exhausted",
            stopDetail: "Reached the latency budget.",
            checkpointThreadId: "g1:c1:channel",
            conversationContext: {
                threadId: "g1:c1:channel",
                kind: "reply_chain",
                replyAnchorMessageId: "m1",
                replyContext: null,
            },
            webStatus: "enabled",
            contextPreview: null,
            recentEvents: [],
            timeline: [
                {
                    label: "stop",
                    detail: "Reached the latency budget.",
                    tone: "warning",
                    timestamp: Date.now(),
                },
            ],
            collapsedSections: {
                request: false,
                conversation: false,
                retrieval: false,
                context: false,
                timeline: true,
            },
            startedAt: Date.now() - 500,
            failureMessage: null,
        }).map((container) => container.toJSON());

        const rendered = JSON.stringify(json);
        expect(rendered).toContain("Stop Condition");
        expect(rendered).toContain("budget_exhausted");
        expect(rendered).toContain("Stop Detail");
        expect(rendered).toContain("Reached the latency budget.");
        expect(rendered).not.toContain("Sophia Debug · Timeline");
        expect(rendered).toContain("+1 more");
    });

    it("keeps long-session debug rendering under Discord display text limits", () => {
        const json = renderDebugTrace({
            questionPreview: "q".repeat(300),
            status: "completed",
            stage: "Completed",
            requesterLabel: "F4zke",
            trigger: "reply",
            classificationMode: "discord_grounded",
            runtimeMode: "research",
            selectedCapabilities: [
                "resolve_channel_targets",
                "list_guild_structure",
                "retrieve_messages",
            ],
            toolCallCount: 3,
            groundingSummary: {
                messageEvidenceCount: 12,
                liveEvidenceCount: 6,
                sufficient: false,
            },
            retrievalSummary: {
                mode: "mixed",
                cacheHit: true,
                liveEscalated: true,
                searchedChannelIds: Array.from({ length: 20 }, (_, index) => `c${index + 1}`),
                fetchedChannelIds: Array.from({ length: 12 }, (_, index) => `f${index + 1}`),
                cacheEnriched: true,
                evidenceSufficient: false,
                strongResultCount: 8,
                weakResultCount: 5,
                historyMessageCount: 10,
                semanticMatchCount: 7,
                accumulatedUniqueCount: 40,
                sourceOrigin: "cache_after_refresh",
                continuationAvailable: true,
                historyContinuationAvailable: true,
                historyCursorByChannel: Object.fromEntries(
                    Array.from({ length: 10 }, (_, index) => [`c${index + 1}`, `${100 + index}`])
                ),
                semanticContinuationAvailable: true,
                semanticCursor: {
                    lastScore: 1.4,
                    lastCreatedTimestamp: 1_700_000_000_000,
                    lastMessageId: "999",
                },
                exhaustedChannelIds: Array.from({ length: 8 }, (_, index) => `x${index + 1}`),
                historyExhausted: false,
                semanticExhausted: false,
                beforeTimestamp: 1_700_000_000_000,
                afterTimestamp: 1_699_000_000_000,
                activeChannelIds: Array.from({ length: 12 }, (_, index) => `a${index + 1}`),
            },
            groundedAnswerMode: "best_effort",
            stopReason: "budget_exhausted",
            stopDetail: "Reached the latency budget while continuation was still available.",
            checkpointThreadId: "g1:c1:channel",
            conversationContext: {
                threadId: "g1:c1:channel",
                kind: "reply_chain",
                replyAnchorMessageId: "m1",
                replyContext: {
                    messageId: "m1",
                    authorId: "u1",
                    authorName: "alice",
                    authorDisplayName: "Alice",
                    content: "x".repeat(600),
                    jumpLink: null,
                },
            },
            webStatus: "enabled",
            contextPreview: {
                recentChannelMessages: Array.from({ length: 10 }, (_, index) => `msg ${index} ${"x".repeat(140)}`),
                evidencePreview: Array.from({ length: 10 }, (_, index) => `[retrieve_messages] #c${index} · user: ${"x".repeat(140)}`),
                recentTurns: Array.from({ length: 5 }, (_, index) => `Q${index}: ${"x".repeat(200)}`),
            },
            recentEvents: [],
            timeline: Array.from({ length: 30 }, (_, index) => ({
                label: `step-${index}`,
                detail: `detail ${index} ${"x".repeat(200)}`,
                tone: "info" as const,
                timestamp: Date.now() - index * 1000,
            })),
            collapsedSections: {
                request: false,
                conversation: false,
                retrieval: false,
                context: false,
                timeline: false,
            },
            startedAt: Date.now() - 10_000,
            failureMessage: null,
        }).map((container) => container.toJSON());

        expect(collectDisplayTextLength(json)).toBeLessThanOrEqual(4000);
    });
});
