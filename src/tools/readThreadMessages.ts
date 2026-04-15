import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        thread_id: {
            type: "string",
            description:
                "The thread ID to read messages from (from list_threads).",
        },
        limit: {
            type: "number",
            description:
                "Number of messages to fetch (1-100). Default: 50.",
        },
    },
    required: ["thread_id"],
} as const;

export const readThreadMessagesTool: ToolDefinition = {
    name: T.read_thread_messages,

    catalog: {
        effect: "read",
        description: "Read recent messages from a Discord thread.",
        evidenceRole: "message_evidence",
    },

    schema: {
        description:
            "Read recent messages from a Discord thread. Returns messages with author info, timestamps, and content. For deeper search, use retrieve_messages with the thread ID as a channelId instead.",
        parameters,
    },

    capability: {
        description: "Read recent messages from a Discord thread.",
        inputSchema: z.object({
            thread_id: z.string().describe("Thread ID to read from."),
            limit: z
                .number()
                .optional()
                .describe("Number of messages (1-100). Default: 50."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["returns messages from the thread"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.read_thread_messages,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const thread = context.guild.channels.cache.get(
                String(args.thread_id),
            );
            if (!thread || !thread.isThread()) {
                return {
                    tool: T.read_thread_messages,
                    summary: "Thread not found.",
                    data: null,
                    errorMessage: "Thread not found.",
                };
            }
            const limit = Math.min(
                Math.max(Number(args.limit) || 50, 1),
                100,
            );
            const fetched = await thread.messages.fetch({ limit });
            const messages = [...fetched.values()]
                .sort((a, b) => a.createdTimestamp - b.createdTimestamp)
                .map((m) => ({
                    messageId: m.id,
                    authorId: m.author.id,
                    authorName:
                        m.member?.displayName ??
                        m.author.globalName ??
                        m.author.username,
                    content: m.content || "",
                    embeds: m.embeds
                        .map((e) => ({
                            title: e.title ?? null,
                            description: e.description ?? null,
                            url: e.url ?? null,
                        }))
                        .filter((e) => e.title || e.description),
                    createdTimestamp: m.createdTimestamp,
                    jumpLink: m.url,
                }));
            const threadMention = `<#${thread.id}>`;
            return {
                tool: T.read_thread_messages,
                summary: messages.length
                    ? `${messages.length} message(s) from thread ${threadMention}.`
                    : `No messages in thread ${threadMention}.`,
                data: {
                    threadId: thread.id,
                    threadMention,
                    threadName: thread.name,
                    messages,
                    messageCount: messages.length,
                },
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];

            const data = run.data as Record<string, unknown>;
            const messages = Array.isArray(data.messages)
                ? data.messages
                : [];
            return messages
                .slice(0, 20)
                .map((m: Record<string, unknown>) => ({
                    tool: T.read_thread_messages,
                    summary: `${String(m.authorName || "?")} in thread: ${String(m.content || "").slice(0, 120)}`,
                    content: String(m.content || ""),
                    authorId: String(m.authorId || ""),
                    authorName: String(m.authorName || ""),
                    channelId: String(data.threadId || ""),
                    channelName: String(data.threadName || ""),
                    jumpLink: String(m.jumpLink || ""),
                    createdTimestamp: Number(m.createdTimestamp || 0),
                    evidenceRole: "message_evidence" as const,
                    strength:
                        String(m.content || "").trim().length >= 80
                            ? ("strong" as const)
                            : ("weak" as const),
                    sourceOrigin: "none" as const,
                }));
        },
    },

    display: { icon: "💬", labelPt: "Ler thread" },
};
