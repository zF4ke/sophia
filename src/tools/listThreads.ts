import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { DiscordToolResult } from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        channel_id: {
            type: "string",
            description: "The parent channel ID to list threads from.",
        },
        include_archived: {
            type: "boolean",
            description:
                "Whether to include archived threads. Default: true.",
        },
    },
    required: ["channel_id"],
} as const;

export const listThreadsTool: ToolDefinition = {
    name: T.list_threads,

    catalog: {
        effect: "read",
        description:
            "List active and archived threads in a Discord channel.",
        evidenceRole: "live_evidence",
    },

    schema: {
        description:
            "List active and recently archived threads in a Discord channel. Returns thread IDs, names, message counts, and creation info. Use the returned thread IDs with retrieve_messages or read_thread_messages to read thread content.",
        parameters,
    },

    capability: {
        description:
            "List active and recently archived threads in a Discord channel.",
        inputSchema: z.object({
            channel_id: z.string().describe("Parent channel ID."),
            include_archived: z
                .boolean()
                .optional()
                .describe("Include archived threads. Default: true."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["returns thread list with IDs and metadata"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.list_threads,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const channel = context.guild.channels.cache.get(
                String(args.channel_id),
            );
            if (!channel || !("threads" in channel)) {
                return {
                    tool: T.list_threads,
                    summary:
                        "Channel not found or does not support threads.",
                    data: null,
                    errorMessage:
                        "Channel not found or does not support threads.",
                };
            }
            const threadChannel =
                channel as import("discord.js").TextChannel;
            const activeThreads =
                await threadChannel.threads.fetchActive();
            const includeArchived = args.include_archived !== false;
            let archivedThreadList: Array<{
                id: string;
                name: string;
                archived: boolean;
                messageCount: number | null;
                createdAt: string | null;
                ownerId: string | null;
                locked: boolean;
            }> = [];
            if (includeArchived) {
                const archivedThreads = await threadChannel.threads
                    .fetchArchived({ fetchAll: true })
                    .catch(() => null);
                if (archivedThreads) {
                    archivedThreadList = archivedThreads.threads.map(
                        (t) => ({
                            id: t.id,
                            name: t.name,
                            archived: true,
                            messageCount: t.messageCount ?? null,
                            createdAt:
                                t.createdAt?.toISOString() ?? null,
                            ownerId: t.ownerId ?? null,
                            locked: t.locked ?? false,
                        }),
                    );
                }
            }
            const allThreads = [
                ...activeThreads.threads.map((t) => ({
                    id: t.id,
                    name: t.name,
                    archived: false,
                    messageCount: t.messageCount ?? null,
                    createdAt: t.createdAt?.toISOString() ?? null,
                    ownerId: t.ownerId ?? null,
                    locked: t.locked ?? false,
                })),
                ...archivedThreadList,
            ];
            return {
                tool: T.list_threads,
                summary: allThreads.length
                    ? `${allThreads.length} thread(s) in <#${channel.id}>.`
                    : `No threads found in <#${channel.id}>.`,
                data: {
                    channelId: channel.id,
                    channelMention: `<#${channel.id}>`,
                    threads: allThreads,
                    threadCount: allThreads.length,
                },
            };
        },
    },

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];

            const data = run.data as Record<string, unknown>;
            const threads = Array.isArray(data.threads)
                ? data.threads
                : [];
            return threads
                .slice(0, 10)
                .map((t: Record<string, unknown>) => ({
                    tool: T.list_threads,
                    summary: `Thread: ${String(t.name || "?")} (${String(t.messageCount || 0)} messages)`,
                    content: `Thread ${String(t.name || "?")} in <#${String(data.channelId || "?")}>: ${String(t.messageCount || 0)} messages, ${t.archived ? "archived" : "active"}`,
                    evidenceRole: "live_evidence" as const,
                    strength: "metadata" as const,
                    sourceOrigin: "none" as const,
                }));
        },
    },

    display: { icon: "🧵", labelPt: "Listar threads" },
};
