import { ContainerBuilder, TextDisplayBuilder } from "discord.js";
import type { DebugTraceState } from "@/discord/debug/types";

function formatDuration(startedAt: number): string {
    const elapsedMs = Math.max(0, Date.now() - startedAt);
    const elapsedSeconds = Math.max(1, Math.round(elapsedMs / 1000));
    return `${elapsedSeconds}s`;
}

function formatStatus(status: DebugTraceState["status"]): string {
    if (status === "completed") return "Completed";
    if (status === "failed") return "Failed";
    return "Running";
}

function formatLabelValue(label: string, value: string): string {
    return `**${label}:** ${value}`;
}

function formatChannelList(channelIds: string[]): string {
    if (!channelIds.length) {
        return "none";
    }

    return channelIds.map((channelId) => `<#${channelId}>`).join(", ");
}

function trimPreview(text: string | null | undefined, maxLength = 220): string {
    const compact = String(text || "").replace(/\s+/g, " ").trim();
    if (!compact) {
        return "none";
    }

    return compact.length > maxLength ? `${compact.slice(0, maxLength - 3)}...` : compact;
}

function buildSection(title: string, lines: string[], accentColor: number): ContainerBuilder {
    return new ContainerBuilder()
        .setAccentColor(accentColor)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(title),
            new TextDisplayBuilder().setContent(lines.join("\n"))
        );
}

export function renderDebugTrace(state: DebugTraceState): ContainerBuilder[] {
    const accentColor =
        state.status === "failed"
            ? 0xed4245
            : state.status === "completed"
              ? 0x57f287
              : 0x9aa7ff;

    const overviewLines = [
        formatLabelValue("Question", state.questionPreview),
        formatLabelValue("Requester", state.requesterLabel || "unknown"),
        formatLabelValue("Trigger", state.trigger || "unknown"),
        formatLabelValue("Status", formatStatus(state.status)),
        formatLabelValue("Stage", state.stage),
        formatLabelValue("Classification", state.classificationMode ?? "pending"),
        formatLabelValue("Runtime Mode", state.runtimeMode ?? "pending"),
        formatLabelValue("Final Confidence", state.groundedAnswerMode ?? "pending"),
        formatLabelValue("Elapsed", formatDuration(state.startedAt)),
    ];

    const conversationLines = [
        formatLabelValue("Conversation Key", state.checkpointThreadId ?? "not set"),
        formatLabelValue("Conversation Kind", state.conversationContext.kind ?? "pending"),
        formatLabelValue(
            "Reply Anchor",
            state.conversationContext.replyAnchorMessageId || "channel fallback"
        ),
        formatLabelValue(
            "Reply Target",
            state.conversationContext.replyContext
                ? `${state.conversationContext.replyContext.authorDisplayName} (${state.conversationContext.replyContext.authorId})`
                : "none"
        ),
        formatLabelValue(
            "Reply Excerpt",
            trimPreview(state.conversationContext.replyContext?.content, 180)
        ),
    ];

    const retrieval = state.retrievalSummary;
    const retrievalLines = [
        formatLabelValue(
            "Capabilities",
            state.selectedCapabilities.length ? state.selectedCapabilities.join(", ") : "none"
        ),
        formatLabelValue("Tool Calls", String(state.toolCallCount)),
        formatLabelValue("Web", state.webStatus ?? "off"),
        formatLabelValue("Stop Reason", state.stopReason ?? "pending"),
        formatLabelValue(
            "Message Evidence",
            state.groundingSummary
                ? `${state.groundingSummary.messageEvidenceCount}`
                : "0"
        ),
        formatLabelValue(
            "Live Metadata Evidence",
            state.groundingSummary ? `${state.groundingSummary.liveEvidenceCount}` : "0"
        ),
        formatLabelValue(
            "Evidence Sufficient",
            state.groundingSummary
                ? state.groundingSummary.sufficient
                    ? "yes"
                    : "no"
                : "not judged"
        ),
        formatLabelValue("Retrieval Origin", retrieval?.sourceOrigin || "none"),
        formatLabelValue("Cache Hit", retrieval ? (retrieval.cacheHit ? "yes" : "no") : "n/a"),
        formatLabelValue(
            "Live Refresh",
            retrieval ? (retrieval.liveEscalated ? "yes" : "no") : "n/a"
        ),
        formatLabelValue(
            "Cache Enriched",
            retrieval ? (retrieval.cacheEnriched ? "yes" : "no") : "n/a"
        ),
        formatLabelValue(
            "Strong / Weak Results",
            retrieval ? `${retrieval.strongResultCount} / ${retrieval.weakResultCount}` : "0 / 0"
        ),
        formatLabelValue(
            "Channels Searched",
            retrieval ? formatChannelList(retrieval.searchedChannelIds) : "none"
        ),
        formatLabelValue(
            "Channels Fetched",
            retrieval ? formatChannelList(retrieval.fetchedChannelIds) : "none"
        ),
    ];

    const timelineEntries = state.timeline.length
        ? state.timeline
              .slice(0, 10)
              .map((entry) => `- [${entry.label}] ${entry.detail}`)
        : ["- Waiting for runtime activity."];

    const containers = [
        buildSection("## Sophia Debug · Request", overviewLines, accentColor),
        buildSection("## Sophia Debug · Conversation", conversationLines, accentColor),
        buildSection("## Sophia Debug · Retrieval", retrievalLines, accentColor),
    ];

    if (state.contextPreview) {
        const contextLines: string[] = [];
        if (state.contextPreview.recentChannelMessages.length) {
            contextLines.push("**Channel Messages:**");
            contextLines.push(
                ...state.contextPreview.recentChannelMessages
                    .slice(0, 6)
                    .map((msg) => `- ${trimPreview(msg, 120)}`)
            );
        }
        if (state.contextPreview.evidencePreview.length) {
            contextLines.push("**Evidence:**");
            contextLines.push(
                ...state.contextPreview.evidencePreview
                    .slice(0, 4)
                    .map((item) => `- ${trimPreview(item, 120)}`)
            );
        }
        if (state.contextPreview.recentTurns.length) {
            contextLines.push("**Prior Turns:**");
            contextLines.push(
                ...state.contextPreview.recentTurns
                    .slice(0, 3)
                    .map((turn) => `- ${trimPreview(turn, 120)}`)
            );
        }
        if (contextLines.length) {
            containers.push(
                buildSection("## Sophia Debug · Context", contextLines, accentColor)
            );
        }
    }

    containers.push(
        buildSection("## Sophia Debug · Timeline", timelineEntries, accentColor),
    );

    if (state.failureMessage) {
        containers.push(
            buildSection(
                "## Sophia Debug · Failure",
                [formatLabelValue("Error", state.failureMessage)],
                0xed4245
            )
        );
    }

    return containers;
}
