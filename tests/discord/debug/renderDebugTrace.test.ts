import { describe, expect, it } from "vitest";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";

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
                cacheHit: true,
                liveEscalated: false,
                searchedChannelIds: ["c1", "c2"],
                fetchedChannelIds: [],
                cacheEnriched: false,
                evidenceSufficient: true,
                strongResultCount: 2,
                weakResultCount: 1,
                sourceOrigin: "cache",
            },
            groundedAnswerMode: "best_effort",
            stopReason: "evidence_sufficient",
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
        expect(rendered).toContain("Web");
        expect(rendered).toContain("Channels Searched");
        expect(rendered).toContain("<#c1>");
        expect(rendered).toContain("Timeline");
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
                    "[retrieve_messages] One Person: Ha uma nameplate que eu queria comprar",
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
            startedAt: Date.now() - 2000,
            failureMessage: null,
        }).map((container) => container.toJSON());

        const rendered = JSON.stringify(json);
        expect(rendered).toContain("Sophia Debug · Context");
        expect(rendered).toContain("Channel Messages:");
        expect(rendered).toContain("One Person:");
        expect(rendered).toContain("Evidence:");
        expect(rendered).toContain("Prior Turns:");
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
            startedAt: Date.now() - 500,
            failureMessage: null,
        }).map((container) => container.toJSON());

        const rendered = JSON.stringify(json);
        expect(rendered).not.toContain("Sophia Debug · Context");
    });
});
