import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        channel_id: { type: "string", description: "Channel ID where the poll lives. Omit to use the current channel." },
        message_id: { type: "string", description: "Message ID of the poll (from the create_poll result). Omit to use the most recent poll in the channel." },
    },
    required: [],
} as const;

interface RawPollAnswer {
    id?: number | string;
    text?: string | null;
    voteCount?: number;
    emoji?: { name?: string | null } | null;
}

interface RawPoll {
    question?: { text?: string | null };
    answers?: { values?: () => IterableIterator<RawPollAnswer> } | RawPollAnswer[];
    allowMultiselect?: boolean;
    expiresTimestamp?: number | null;
    fetch?: () => Promise<unknown>;
}

function answersOf(poll: RawPoll): RawPollAnswer[] {
    const answers = poll.answers;
    if (Array.isArray(answers)) return answers;
    if (answers && typeof answers.values === "function") return [...answers.values()];
    return [];
}

export const getPollResultsTool: ToolDefinition = {
    name: T.get_poll_results,

    catalog: {
        effect: "read",
        description: "Read the current vote counts of a Discord poll (per-answer totals, percentages, expiry).",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Read the live vote counts of a poll. Pass the poll's message_id (from the create_poll result); omit message_id to use the most recent poll in the channel, and omit channel_id to use the current channel. Returns per-answer vote counts with percentages, total votes, and expiry. Vote totals are visible even when individual voters are anonymous.",
        parameters,
    },

    capability: {
        description: "Read poll vote counts.",
        inputSchema: z.object({
            channel_id: z.string().optional(),
            message_id: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: ["the target message contains a poll"],
        postconditions: ["returns per-answer vote counts"],
        async run(context, args) {
            if (!context.guild) {
                return { tool: T.get_poll_results, summary: "No guild context.", data: null, errorMessage: "No guild context." };
            }
            const targetId = typeof args.channel_id === "string" && args.channel_id.trim() ? args.channel_id.trim() : context.currentChannelId;
            if (!targetId) {
                return { tool: T.get_poll_results, summary: "No channel.", data: null, errorMessage: "No channel_id given and no current channel." };
            }
            const channel = context.guild.channels.cache.get(targetId) as
                | { messages?: { fetch: (query: unknown) => Promise<unknown> } }
                | undefined;
            if (!channel?.messages) {
                return { tool: T.get_poll_results, summary: "Channel not found.", data: null, errorMessage: "Channel not found or not text-based." };
            }

            // Locate the poll message: explicit id, or the newest poll in the channel.
            let pollMessage: { id: string; poll?: RawPoll | null } | null = null;
            try {
                if (typeof args.message_id === "string" && args.message_id.trim()) {
                    pollMessage = await channel.messages.fetch(args.message_id.trim()) as never;
                } else {
                    const recent = await channel.messages.fetch({ limit: 50 });
                    const list = Array.isArray(recent)
                        ? recent
                        : [...(recent as { values: () => IterableIterator<{ id: string; poll?: RawPoll | null }> }).values()];
                    pollMessage = list.find((m) => m && m.poll) ?? null;
                }
            } catch (error) {
                const detail = error instanceof Error ? error.message : String(error);
                return { tool: T.get_poll_results, summary: "Could not fetch the poll message.", data: null, errorMessage: `Message fetch failed: ${detail}` };
            }
            if (!pollMessage) {
                return { tool: T.get_poll_results, summary: "No poll found in the recent channel history.", data: null, errorMessage: "No poll found in the last 50 messages." };
            }
            if (!pollMessage.poll) {
                return { tool: T.get_poll_results, summary: "That message is not a poll.", data: { messageId: pollMessage.id }, errorMessage: "The message has no poll attached." };
            }

            // Refresh counts: cached messages may carry stale answer counts.
            const poll = pollMessage.poll;
            if (typeof poll.fetch === "function") {
                try {
                    const refreshed = await poll.fetch();
                    if (refreshed && typeof refreshed === "object") Object.assign(poll, refreshed as object);
                } catch {
                    // Stale counts are better than no counts.
                }
            }

            const question = poll.question?.text || "Poll";
            const rows = answersOf(poll).map((answer) => ({
                id: answer.id ?? null,
                text: answer.text ?? "",
                emoji: answer.emoji?.name ?? null,
                votes: Number(answer.voteCount ?? 0),
            }));
            const totalVotes = rows.reduce((sum, row) => sum + row.votes, 0);
            const withPercentages = rows.map((row) => ({
                ...row,
                percentage: totalVotes > 0 ? Math.round((row.votes / totalVotes) * 100) : 0,
            }));
            const lines = withPercentages.map((row) => `${row.text}: ${row.votes} vote(s) (${row.percentage}%)`);
            const leader = totalVotes > 0
                ? [...withPercentages].sort((a, b) => b.votes - a.votes)[0]
                : null;
            const expiresAt = poll.expiresTimestamp ? new Date(poll.expiresTimestamp).toISOString() : null;

            return {
                tool: T.get_poll_results,
                summary: `Poll "${question}" (${totalVotes} vote(s)${poll.allowMultiselect ? ", multi-select" : ""}${expiresAt ? `, ends ${expiresAt}` : ""}):\n${lines.join("\n")}${leader ? `\nLeading: "${leader.text}"` : ""}`,
                data: {
                    messageId: pollMessage.id,
                    channelId: targetId,
                    channelMention: `<#${targetId}>`,
                    question,
                    totalVotes,
                    allowMultiselect: Boolean(poll.allowMultiselect),
                    expiresAt,
                    answers: withPercentages,
                },
            };
        },
    },

    strategy: { extractEvidence() { return []; } },
    display: { icon: "🗳️", labelPt: "Ver votos da poll" },
};

export type PollArguments = ToolArguments;
