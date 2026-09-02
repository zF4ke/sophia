import { z } from "zod";
import { PollLayoutType } from "discord.js";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";

const params = {
    type: "object",
    properties: {
        channel_id: { type: "string", description: "Channel ID to send the poll to. If omitted, uses the current channel." },
        question: { type: "string", description: "Poll question (max 300 chars)." },
        answers: {
            type: "array",
            description: "Poll answers (2-10 strings, each max 55 chars).",
            items: { type: "string" },
        },
        duration_hours: { type: "number", description: "Duration in hours (1-768, default 24). 1h minimum for Discord polls." },
        allow_multiselect: { type: "boolean", description: "Allow multiple answers per voter (default false)." },
    },
    required: ["question", "answers"],
} as const;

export const createPollTool: ToolDefinition = {
    name: T.create_poll,
    catalog: {
        effect: "write",
        description: "Create a native Discord poll in a channel (2-10 answers, with duration).",
        evidenceRole: "discovery_only",
    },
    schema: {
        description: "Create a Discord native poll. Use when the user says 'faz uma poll', 'poll: ...' or wants a vote. Provide question and 2-10 answers. The poll renders as a real Discord poll, not emoji reactions.",
        parameters: params,
    },
    capability: {
        description: "Create Discord poll.",
        inputSchema: z.object({
            channel_id: z.string().optional(),
            question: z.string().min(5).max(300),
            answers: z.array(z.string().min(1).max(55)).min(2).max(10),
            duration_hours: z.number().min(1).max(768).optional(),
            allow_multiselect: z.boolean().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: ["isGuildMember"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["bot has SendMessages permission in target channel"],
        postconditions: ["poll message created with ID and channel mention"],
        async run(context, args) {
            const guild = context.guild;
            if (!guild) return { tool: T.create_poll, summary: "No guild context.", data: null, errorMessage: "Polls require a guild." };
            const targetId = String(args.channel_id || context.currentChannelId || "").trim();
            if (!targetId) return { tool: T.create_poll, summary: "No channel.", data: null, errorMessage: "Provide channel_id or use current channel." };
            const channel = guild.channels.cache.get(targetId) as never as { isTextBased: () => boolean; send: (o: unknown) => Promise<{ id: string; url: string }> } | undefined;
            const isText = channel && typeof (channel as { isTextBased?: () => boolean }).isTextBased === "function" ? (channel as { isTextBased: () => boolean }).isTextBased() : false;
            if (!channel || !isText) return { tool: T.create_poll, summary: `Channel ${targetId} not text-based.`, data: null, errorMessage: "Target channel is not text-based or not found." };
            const question = String(args.question).trim().slice(0, 300);
            const answers = (args.answers as string[]).map((a) => String(a).trim().slice(0, 55)).filter(Boolean);
            const duration = Math.max(1, Math.min(768, Number(args.duration_hours) || 24));
            const allowMultiselect = Boolean(args.allow_multiselect);
            try {
                const msg = await (channel as unknown as { send: (o: unknown) => Promise<{ id: string }> }).send({
                    poll: {
                        question: { text: question },
                        answers: answers.map((text) => ({ text, emoji: undefined })),
                        duration,
                        allowMultiselect,
                        layoutType: PollLayoutType.Default,
                    },
                } as never);
                const channelMention = `<#${targetId}>`;
                return {
                    tool: T.create_poll,
                    summary: `Created poll "${question}" in ${channelMention} (${answers.length} answers, ${duration}h).`,
                    data: { messageId: msg.id, channelId: targetId, channelMention, question, answers, durationHours: duration },
                };
            } catch (e) {
                const msg = e instanceof Error ? e.message : String(e);
                return { tool: T.create_poll, summary: `Failed to create poll: ${msg}`, data: null, errorMessage: msg };
            }
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "📊", labelPt: "Criar poll" },
    describeApproval: (args) => `Criar poll "${String(args.question).slice(0, 60)}" em <#${String(args.channel_id || "?")}>`,
};
