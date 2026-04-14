import {
    ContainerBuilder,
    TextDisplayBuilder,
} from "discord.js";
import type { DebugTraceState } from "@/discord/debug/types";

const TIMELINE_VISIBLE_ROWS = 20;

function formatDuration(startedAt: number): string {
    const elapsedMs = Math.max(0, Date.now() - startedAt);
    const elapsedSeconds = Math.max(1, Math.round(elapsedMs / 1000));
    return `${elapsedSeconds}s`;
}

function statusEmoji(status: DebugTraceState["status"]): string {
    if (status === "completed") return "✅";
    if (status === "failed") return "❌";
    return "⏳";
}

function toneEmoji(tone: "info" | "success" | "warning" | "error"): string {
    if (tone === "success") return "✅";
    if (tone === "warning") return "⚠️";
    if (tone === "error") return "❌";
    return "▸";
}

function trimText(text: string | null | undefined, maxLen = 200): string {
    const compact = String(text || "").replace(/\s+/g, " ").trim();
    if (!compact) return "—";
    return compact.length > maxLen ? `${compact.slice(0, maxLen - 3)}...` : compact;
}

export function renderDebugTrace(
    state: DebugTraceState
): ContainerBuilder[] {
    const accentColor =
        state.status === "failed"
            ? 0xed4245
            : state.status === "completed"
              ? 0x57f287
              : 0x9aa7ff;

    const overviewLines = [
        `${statusEmoji(state.status)} **${state.status.toUpperCase()}** — ${state.stage} — ${formatDuration(state.startedAt)}`,
        `> **Question:** ${state.questionPreview}`,
        `> **Requester:** ${state.requesterLabel || "unknown"} · **Trigger:** ${state.trigger || "unknown"}`,
        `> **Mode:** ${state.classificationMode ?? "pending"} · **Runtime:** ${state.runtimeMode ?? "pending"}`,
        `> **Tools:** ${state.selectedCapabilities.length ? state.selectedCapabilities.join(", ") : "none"} (${state.toolCallCount} calls)`,
        `> **Evidence:** ${state.evidenceCount} items · **Confidence:** ${state.confidence ?? "pending"}`,
    ];

    if (state.cumulativePromptTokens > 0) {
        const tokenInfo = `${state.cumulativePromptTokens.toLocaleString()} prompt + ${state.cumulativeCompletionTokens.toLocaleString()} completion`;
        const contextInfo = state.contextUsagePercent != null ? ` · **Context:** ${state.contextUsagePercent.toFixed(1)}%` : "";
        overviewLines.push(`> **Tokens:** ${tokenInfo}${contextInfo}`);
    }

    if (state.stopReason) {
        overviewLines.push(`> **Stop:** ${state.stopReason}${state.stopDetail ? ` — ${trimText(state.stopDetail, 100)}` : ""}`);
    }

    if (state.conversationThreadId) {
        overviewLines.push(`> **Thread:** \`${state.conversationThreadId}\``);
    }

    if (state.failureMessage) {
        overviewLines.push(`\n⛔ **Error:** ${trimText(state.failureMessage, 200)}`);
    }

    const containers: ContainerBuilder[] = [];

    containers.push(
        new ContainerBuilder()
            .setAccentColor(accentColor)
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent("## Sophia Debug Trace"),
                new TextDisplayBuilder().setContent(overviewLines.join("\n"))
            )
    );

    if (state.timeline.length) {
        const visible = state.timeline.slice(-TIMELINE_VISIBLE_ROWS);
        const hiddenCount = state.timeline.length - visible.length;
        const timelineLines = visible.map((entry) => {
            const relMs = Math.max(0, entry.timestamp - state.startedAt);
            const relS = (relMs / 1000).toFixed(1);
            return `${toneEmoji(entry.tone)} \`+${relS}s\` ${trimText(entry.detail, 120)}`;
        });

        if (hiddenCount > 0) {
            timelineLines.unshift(`… ${hiddenCount} earlier events`);
        }

        containers.push(
            new ContainerBuilder()
                .setAccentColor(0x5865f2)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("### Timeline"),
                    new TextDisplayBuilder().setContent(timelineLines.join("\n"))
                )
        );
    }

    return containers;
}
