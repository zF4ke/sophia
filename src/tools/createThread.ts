import { z } from "zod";
import { ThreadAutoArchiveDuration } from "discord.js";
import { T } from "@/shared/discordTools";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        channel_id: {
            type: "string",
            description: "The parent channel ID to create the thread in.",
        },
        name: {
            type: "string",
            description: "The name for the new thread.",
        },
        message: {
            type: "string",
            description: "Optional initial message to post in the thread.",
        },
        auto_archive_duration: {
            type: "number",
            description:
                "Auto-archive duration in minutes. Must be one of: 60 (1h), 1440 (1d), 4320 (3d), 10080 (7d). Default: 1440.",
        },
    },
    required: ["channel_id", "name"],
} as const;

export const createThreadTool: ToolDefinition = {
    name: T.create_thread,

    catalog: {
        effect: "write",
        description:
            "Create a new thread in a channel. Write — requires admin approval.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Create a new thread in a channel. This is a WRITE action that requires admin approval. Use only when explicitly asked to create a thread.",
        parameters,
    },

    capability: {
        description:
            "Create a new thread in a channel. Write — requires admin approval.",
        inputSchema: z.object({
            channel_id: z.string().describe("Parent channel ID."),
            name: z.string().describe("Thread name."),
            message: z
                .string()
                .optional()
                .describe("Initial message content."),
            auto_archive_duration: z
                .number()
                .optional()
                .describe("Auto-archive in minutes."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [
            "guild context should exist",
            "channel must support threads",
        ],
        postconditions: ["creates a new thread in the channel"],
        async run(context, args) {
            if (!context.guild) {
                return {
                    tool: T.create_thread,
                    summary: "No guild context.",
                    data: null,
                    errorMessage: "No guild context.",
                };
            }
            const channel = context.guild.channels.cache.get(
                String(args.channel_id),
            );
            if (
                !channel ||
                !channel.isTextBased() ||
                !("threads" in channel)
            ) {
                return {
                    tool: T.create_thread,
                    summary:
                        "Channel not found or does not support threads.",
                    data: null,
                    errorMessage:
                        "Channel not found or does not support threads.",
                };
            }
            const validDurations = [60, 1440, 4320, 10080];
            const archiveDuration =
                typeof args.auto_archive_duration === "number" &&
                validDurations.includes(args.auto_archive_duration)
                    ? (args.auto_archive_duration as ThreadAutoArchiveDuration)
                    : ThreadAutoArchiveDuration.OneDay;

            const threadChannel =
                channel as import("discord.js").TextChannel;
            const thread = await threadChannel.threads.create({
                name: String(args.name),
                autoArchiveDuration: archiveDuration,
                reason: "Created by Sophia",
            });
            if (typeof args.message === "string" && args.message.trim()) {
                await thread.send(args.message.trim());
            }
            const threadMention = `<#${thread.id}>`;
            return {
                tool: T.create_thread,
                summary: `Created thread ${threadMention} in <#${channel.id}>.`,
                data: {
                    threadId: thread.id,
                    threadMention,
                    threadName: thread.name,
                    parentChannelId: channel.id,
                    parentChannelMention: `<#${channel.id}>`,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🧵", labelPt: "Criar thread" },

    describeApproval(args: ToolArguments) {
        return `Criar thread "${args.name ?? "?"}" em <#${args.channel_id ?? "desconhecido"}>`;
    },
};
