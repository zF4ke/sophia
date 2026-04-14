import type { ActiveRetrievalSession, EvidenceItem, RetrievalSummary } from "@/runtime/contracts";
import type { DiscordToolResult, RetrievalMode } from "@/shared/appTypes";
import type { RetrievalPayload, ToolStrategy } from "./types";

function toRetrievalSummary(payload: RetrievalPayload): RetrievalSummary {
    return {
        mode: (payload.mode || "history") as RetrievalSummary["mode"],
        cacheHit: Boolean(payload.cacheHit),
        liveEscalated: Boolean(payload.liveEscalated),
        searchedChannelIds: (payload.searchedChannelIds || []).map(String),
        fetchedChannelIds: (payload.fetchedChannelIds || []).map(String),
        cacheEnriched: Boolean(payload.cacheEnriched),
        evidenceSufficient: Boolean(payload.evidenceSufficient),
        strongResultCount: Number(payload.strongResultCount || 0),
        weakResultCount: Number(payload.weakResultCount || 0),
        historyMessageCount: Number(payload.historyMessageCount || 0),
        semanticMatchCount: Number(payload.semanticMatchCount || 0),
        accumulatedUniqueCount: Number(payload.accumulatedUniqueCount || 0),
        sourceOrigin: (payload.sourceOrigin || "none") as RetrievalSummary["sourceOrigin"],
        continuationAvailable: Boolean(payload.continuation?.continuationAvailable),
        historyContinuationAvailable:
            payload.continuation?.history?.continuationAvailable == null
                ? Boolean(payload.continuation?.continuationAvailable)
                : Boolean(payload.continuation?.history?.continuationAvailable),
        historyCursorByChannel:
            payload.continuation?.history?.perChannelOldestMessageId ||
            payload.continuation?.perChannelOldestMessageId ||
            {},
        semanticContinuationAvailable: Boolean(payload.continuation?.semantic?.continuationAvailable),
        semanticCursor: payload.continuation?.semantic?.cursor || null,
        exhaustedChannelIds: (payload.exhaustion?.exhaustedChannelIds || []).map(String),
        historyExhausted: Boolean(payload.exhaustion?.historyExhausted),
        semanticExhausted: Boolean(payload.exhaustion?.semanticExhausted),
        beforeTimestamp:
            payload.beforeTimestamp == null ? null : Number(payload.beforeTimestamp),
        afterTimestamp:
            payload.afterTimestamp == null ? null : Number(payload.afterTimestamp),
        activeChannelIds: (payload.targetChannelIds || []).map(String),
    };
}

function toRetrievalSession(payload: RetrievalPayload & Record<string, unknown>): ActiveRetrievalSession | null {
    const targetChannelIds = Array.isArray(payload.targetChannelIds)
        ? payload.targetChannelIds.map(String)
        : [];

    if (!targetChannelIds.length) {
        return null;
    }

    return {
        mode: (payload.mode || "history") as RetrievalMode,
        channelIds: targetChannelIds,
        authorId: payload.targetAuthorId == null ? null : String(payload.targetAuthorId),
        beforeTimestamp: payload.beforeTimestamp == null ? null : Number(payload.beforeTimestamp),
        afterTimestamp: payload.afterTimestamp == null ? null : Number(payload.afterTimestamp),
        historyCursorByChannel:
            payload.continuation?.history?.perChannelOldestMessageId ||
            payload.continuation?.perChannelOldestMessageId ||
            {},
        semanticCursor: payload.continuation?.semantic?.cursor || null,
        seenMessageIds: (payload.combinedResults || [])
            .map((row) => (row && typeof row === "object" && "messageId" in row ? String((row as Record<string, unknown>).messageId) : null))
            .filter((value): value is string => Boolean(value)),
        accumulatedUniqueCount: Number(payload.accumulatedUniqueCount || 0),
        exhaustedChannelIds: (payload.exhaustion?.exhaustedChannelIds || []).map(String),
        historyExhausted: Boolean(payload.exhaustion?.historyExhausted),
        semanticExhausted: Boolean(payload.exhaustion?.semanticExhausted),
        continuationAvailable: Boolean(payload.continuation?.continuationAvailable),
    };
}

export const retrieveMessagesStrategy: ToolStrategy = {
    id: "retrieve_messages",

    extractEvidence(run: DiscordToolResult): EvidenceItem[] {
        if (!run.data || typeof run.data !== "object") {
            return [];
        }

        const payload = run.data as RetrievalPayload;
        const historyRows = payload.historyMessages || [];
        const semanticRows = payload.semanticMatches || [];
        const sourceOrigin = (payload.sourceOrigin || "none") as RetrievalSummary["sourceOrigin"];

        const historyEvidence = historyRows.map((item) => {
            const body = String(item.content || "");
            return {
                tool: "retrieve_messages" as const,
                summary: run.summary,
                content: body,
                evidenceRole: "history_evidence" as const,
                strength: "strong" as const,
                sourceOrigin,
                messageId: item.messageId == null ? null : String(item.messageId),
                authorId: item.authorId == null ? null : String(item.authorId),
                authorName: item.authorName == null ? null : String(item.authorName),
                authorUsername: item.authorUsername == null ? null : String(item.authorUsername),
                authorNickname: item.authorNickname == null ? null : String(item.authorNickname),
                channelId: item.channelId == null ? null : String(item.channelId),
                channelName: item.channelName == null ? null : String(item.channelName),
                jumpLink: item.jumpLink == null ? null : String(item.jumpLink),
                createdTimestamp: item.createdTimestamp == null ? null : Number(item.createdTimestamp),
            };
        });

        const semanticEvidence = semanticRows.map((item) => {
            const body = String(item.content || "");
            const lexicalScore = Number(item.lexicalScore || 0);
            const strength: EvidenceItem["strength"] =
                lexicalScore >= 2 || (lexicalScore >= 1 && body.length >= 80)
                    ? "strong"
                    : "weak";

            return {
                tool: "retrieve_messages" as const,
                summary: run.summary,
                content: body,
                evidenceRole: "semantic_evidence" as const,
                strength,
                sourceOrigin,
                messageId: item.messageId == null ? null : String(item.messageId),
                authorId: item.authorId == null ? null : String(item.authorId),
                authorName: item.authorName == null ? null : String(item.authorName),
                authorUsername: item.authorUsername == null ? null : String(item.authorUsername),
                authorNickname: item.authorNickname == null ? null : String(item.authorNickname),
                channelId: item.channelId == null ? null : String(item.channelId),
                channelName: item.channelName == null ? null : String(item.channelName),
                jumpLink: item.jumpLink == null ? null : String(item.jumpLink),
                createdTimestamp: item.createdTimestamp == null ? null : Number(item.createdTimestamp),
            };
        });

        return [...historyEvidence, ...semanticEvidence];
    },

    extractRetrievalSummary(run: DiscordToolResult): RetrievalSummary | null {
        if (!run.data || typeof run.data !== "object") {
            return null;
        }
        return toRetrievalSummary(run.data as RetrievalPayload);
    },

    extractRetrievalSession(run: DiscordToolResult): ActiveRetrievalSession | null {
        if (!run.data || typeof run.data !== "object") {
            return null;
        }
        return toRetrievalSession(run.data as RetrievalPayload & Record<string, unknown>);
    },
};
