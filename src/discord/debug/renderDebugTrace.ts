import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    TextDisplayBuilder,
} from "discord.js";
import type { DebugSectionKey, DebugTraceState } from "@/discord/debug/types";

const TIMELINE_VISIBLE_ROWS = 25;
const MAX_LINE_LENGTH = 120;
const REQUEST_SECTION_BUDGET = 480;
const CONVERSATION_SECTION_BUDGET = 520;
const RETRIEVAL_SECTION_BUDGET = 900;
const CONTEXT_SECTION_BUDGET = 700;
const TIMELINE_SECTION_BUDGET = 1500;
const FAILURE_SECTION_BUDGET = 240;
const TRACE_TOGGLE_PREFIX = "debug:trace:toggle:";
const TRACE_EXPAND_ALL_ID = "debug:trace:expand_all";
const TRACE_COLLAPSE_ALL_ID = "debug:trace:collapse_all";

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

    const visible = channelIds.slice(0, 6).map((channelId) => `<#${channelId}>`);
    const hiddenCount = Math.max(0, channelIds.length - visible.length);
    return hiddenCount > 0
        ? `${visible.join(", ")}, +${hiddenCount} more`
        : visible.join(", ");
}

function trimPreview(text: string | null | undefined, maxLength = 220): string {
    const compact = String(text || "").replace(/\s+/g, " ").trim();
    if (!compact) {
        return "none";
    }

    return compact.length > maxLength ? `${compact.slice(0, maxLength - 3)}...` : compact;
}

function fitSectionLines(lines: string[], maxChars: number): string[] {
    const fitted: string[] = [];
    let used = 0;

    for (const line of lines) {
        const trimmed = trimPreview(line, MAX_LINE_LENGTH);
        const nextLength = trimmed.length + (fitted.length ? 1 : 0);
        if (used + nextLength > maxChars) {
            fitted.push("...truncated");
            break;
        }
        fitted.push(trimmed);
        used += nextLength;
    }

    return fitted.length ? fitted : ["none"];
}

function buildSection(
    title: string,
    lines: string[],
    accentColor: number,
    maxChars: number
): ContainerBuilder {
    return new ContainerBuilder()
        .setAccentColor(accentColor)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(title),
            new TextDisplayBuilder().setContent(fitSectionLines(lines, maxChars).join("\n"))
        );
}

function buildTimelineLines(state: DebugTraceState): string[] {
    return state.timeline.length
        ? state.timeline
              .slice(0, TIMELINE_VISIBLE_ROWS)
              .map((entry) => `- [${entry.label}] ${trimPreview(entry.detail, 96)}`)
        : ["- Waiting for runtime activity."];
}

function buildToggleRows(
    state: DebugTraceState
): Array<ActionRowBuilder<ButtonBuilder>> {
    const sectionLabels: Array<{ key: DebugSectionKey; label: string }> = [
        { key: "request", label: "Request" },
        { key: "conversation", label: "Conversation" },
        { key: "retrieval", label: "Retrieval" },
        { key: "context", label: "Context" },
        { key: "timeline", label: "Timeline" },
    ];

    const sectionButtons = sectionLabels.map(({ key, label }) =>
        new ButtonBuilder()
            .setCustomId(`${TRACE_TOGGLE_PREFIX}${key}`)
            .setLabel(label)
            .setStyle(
                state.collapsedSections[key] ? ButtonStyle.Secondary : ButtonStyle.Primary
            )
    );

    return [
        new ActionRowBuilder<ButtonBuilder>().addComponents(...sectionButtons),
        new ActionRowBuilder<ButtonBuilder>().addComponents(
            new ButtonBuilder()
                .setCustomId(TRACE_EXPAND_ALL_ID)
                .setLabel("Expand All")
                .setStyle(ButtonStyle.Success),
            new ButtonBuilder()
                .setCustomId(TRACE_COLLAPSE_ALL_ID)
                .setLabel("Collapse All")
                .setStyle(ButtonStyle.Secondary)
        ),
    ];
}

export function parseDebugTraceSectionToggle(customId: string): DebugSectionKey | null {
    if (!customId.startsWith(TRACE_TOGGLE_PREFIX)) {
        return null;
    }

    const section = customId.slice(TRACE_TOGGLE_PREFIX.length);
    return section === "request" ||
        section === "conversation" ||
        section === "retrieval" ||
        section === "context" ||
        section === "timeline"
        ? section
        : null;
}

export function isDebugTraceExpandAll(customId: string): boolean {
    return customId === TRACE_EXPAND_ALL_ID;
}

export function isDebugTraceCollapseAll(customId: string): boolean {
    return customId === TRACE_COLLAPSE_ALL_ID;
}

export function renderDebugTrace(
    state: DebugTraceState
): Array<ContainerBuilder | ActionRowBuilder<ButtonBuilder>> {
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
        formatLabelValue("Retrieval Mode", retrieval?.mode || "history"),
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
            "History / Semantic",
            retrieval ? `${retrieval.historyMessageCount} / ${retrieval.semanticMatchCount}` : "0 / 0"
        ),
        formatLabelValue(
            "Time Scope",
            retrieval
                ? `after=${retrieval.afterTimestamp ?? "none"} before=${retrieval.beforeTimestamp ?? "none"}`
                : "none"
        ),
        formatLabelValue(
            "Continuation Available",
            retrieval ? (retrieval.continuationAvailable ? "yes" : "no") : "n/a"
        ),
        formatLabelValue(
            "Exhausted Channels",
            retrieval ? formatChannelList(retrieval.exhaustedChannelIds) : "none"
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

    const timelineEntries = buildTimelineLines(state);

    const containers: Array<ContainerBuilder | ActionRowBuilder<ButtonBuilder>> = [
        ...buildToggleRows(state),
    ];

    overviewLines.splice(4, 0, formatLabelValue("Stop Condition", state.stopReason ?? "pending"));
    overviewLines.splice(
        5,
        0,
        formatLabelValue("Stop Detail", trimPreview(state.stopDetail, 180))
    );

    if (!state.collapsedSections.request) {
        containers.push(
            buildSection("## Sophia Debug · Request", overviewLines, accentColor, REQUEST_SECTION_BUDGET)
        );
    }
    if (!state.collapsedSections.conversation) {
        containers.push(
            buildSection("## Sophia Debug · Conversation", conversationLines, accentColor, CONVERSATION_SECTION_BUDGET)
        );
    }
    if (!state.collapsedSections.retrieval) {
        containers.push(
            buildSection("## Sophia Debug · Retrieval", retrievalLines, accentColor, RETRIEVAL_SECTION_BUDGET)
        );
    }

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
        if (contextLines.length && !state.collapsedSections.context) {
            containers.push(
                buildSection("## Sophia Debug · Context", contextLines, accentColor, CONTEXT_SECTION_BUDGET)
            );
        }
    }

    if (!state.collapsedSections.timeline) {
        containers.push(
            buildSection("## Sophia Debug · Timeline", timelineEntries, accentColor, TIMELINE_SECTION_BUDGET),
        );
    }

    if (state.failureMessage) {
        containers.push(
            buildSection(
                "## Sophia Debug · Failure",
                [formatLabelValue("Error", state.failureMessage)],
                0xed4245,
                FAILURE_SECTION_BUDGET
            )
        );
    }

    return containers;
}
