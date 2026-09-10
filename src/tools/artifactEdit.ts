import { z } from "zod";
import { T } from "@/shared/discordTools";
import { applyArtifactEdit } from "@/discord/artifacts/ArtifactSession";
import { artifactExpiryTimestamp, validateArtifactSpec } from "@/discord/artifacts/ArtifactBuilder";
import { ArtifactStore } from "@/discord/artifacts/ArtifactStore";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        message_id: {
            type: "string",
            description: "Message ID of the artifact card to edit, from the artifact_send result.",
        },
        title: { type: "string", description: "New card title. Omit to keep the current one." },
        summary: { type: "string", description: "New summary. Omit to keep the current one." },
        sections: {
            type: "array",
            description: "Replacement sections (1 to 50). Omit to keep the current ones. This replaces the whole list, not a partial patch.",
            items: {
                type: "object",
                properties: {
                    heading: { type: "string", description: "Optional section heading, max 80 chars." },
                    body: { type: "string", description: "Section body, markdown allowed, max 3000 chars." },
                    thumbnail_url: { type: "string", description: "Optional direct https image URL rendered top-right of this section like an embed thumbnail." },
                    accessory_button: {
                        type: "object",
                        description: "Optional button accessory (right side). Provide either thumbnail_url or accessory_button, not both.",
                        properties: {
                            label: { type: "string" },
                            url: { type: "string" },
                            customId: { type: "string" },
                            style: { type: "number" },
                            emoji: { type: "string" },
                        },
                        required: ["label"],
                    },
                },
                required: ["body"],
            },
        },
        gallery: {
            type: "array",
            description: "Replacement image grid below the content (max 10 direct https image URLs). Omit to keep the current one.",
            items: { type: "string" },
        },
        files: {
            type: "array",
            description: "Replacement file cards (max 10 https file URLs). Omit to keep the current ones.",
            items: { type: "string" },
        },
        accent_color: {
            type: "number",
            description: "New accent color as an integer (0 to 0xffffff). Omit to keep the current one.",
        },
        spoiler: {
            type: "boolean",
            description: "Gray out the whole card. Omit to keep the current value.",
        },
        navigation: {
            type: "object",
            description: "Replacement navigation config ({type:'select'} or {type:'pagination'}). Omit to keep the current one.",
            properties: {
                type: { type: "string", enum: ["select", "pagination"], description: "Navigation style." },
            },
            required: ["type"],
        },
        action_rows: {
            type: "array",
            description: "Replacement custom interactive rows (max 5). Each row is buttons or a select (string/user/role/mentionable/channel). Omit to keep the current ones.",
            items: {
                type: "object",
                properties: {
                    type: { type: "string", enum: ["buttons", "stringSelect", "userSelect", "roleSelect", "mentionableSelect", "channelSelect"] },
                    buttons: {
                        type: "array",
                        items: {
                            type: "object",
                            properties: {
                                label: { type: "string" },
                                style: { type: "number" },
                                customId: { type: "string" },
                                url: { type: "string" },
                                emoji: { type: "string" },
                                disabled: { type: "boolean" },
                            },
                            required: ["label"],
                        },
                    },
                    customId: { type: "string" },
                    placeholder: { type: "string" },
                    minValues: { type: "number" },
                    maxValues: { type: "number" },
                    channelTypes: { type: "array", items: { type: "number" } },
                    options: {
                        type: "array",
                        items: {
                            type: "object",
                            properties: {
                                label: { type: "string" },
                                value: { type: "string" },
                                description: { type: "string" },
                                emoji: { type: "string" },
                                default: { type: "boolean" },
                            },
                            required: ["label", "value"],
                        },
                    },
                },
                required: ["type"],
            },
        },
        game_state: {
            type: "object",
            description: "Replacement game state JSON (max ~4000 chars serialized). Persisted per card for minigames.",
            properties: {},
            required: [],
        },
        handlers: {
            type: "object",
            description: "Replacement sandboxed JS handlers keyed by customId. Omit to keep the current ones. Same VM contract as artifact_send.",
            properties: {},
            required: [],
        },
        link_buttons: {
            type: "array",
            description: "Replacement https link buttons (max 5). Omit to keep the current ones.",
            items: {
                type: "object",
                properties: {
                    label: { type: "string", description: "Button label, max 80 chars." },
                    url: { type: "string", description: "Absolute https URL the button opens." },
                    emoji: { type: "string", description: "Optional emoji shown before the label." },
                },
                required: ["label", "url"],
            },
        },
        ttl_days: {
            type: "number",
            description: "New retention in days, counted from now. 0 removes the expiry.",
        },
        rearm: {
            type: "boolean",
            description: "No content change: re-render the card's controls with fresh, correctly-namespaced custom ids. Use when the user reports buttons not responding (legacy cards, bot restarts, or disabled controls). Needs no other fields.",
        },
    },
    required: ["message_id"],
} as const;

const EDITABLE_FIELDS = ["title", "summary", "sections", "navigation", "link_buttons", "gallery", "files", "accent_color", "spoiler", "action_rows", "game_state", "handlers", "ttl_days"] as const;

export const artifactEditTool: ToolDefinition = {
    name: T.artifact_edit,

    catalog: {
        effect: "write",
        description: "Edit an artifact card you previously sent: update its title, summary, sections, navigation, buttons, or TTL in place.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Edit an artifact card in place. Pass the card's message_id plus only the fields you want to change; everything else is preserved. The merged spec is validated before the edit is applied.",
        parameters,
    },

    capability: {
        description: "Edit an artifact card.",
        inputSchema: z.object({
            message_id: z.string(),
            title: z.string().optional(),
            summary: z.string().optional(),
            // String forms are accepted and recovered by validateArtifactSpec.
            sections: z.union([z.array(z.object({
                heading: z.string().optional(),
                body: z.string(),
                thumbnail_url: z.string().optional(),
                accessory_button: z.object({
                    label: z.string(),
                    url: z.string().optional(),
                    customId: z.string().optional(),
                    style: z.number().optional(),
                    emoji: z.string().optional(),
                    disabled: z.boolean().optional(),
                }).optional(),
            })), z.string()]).optional(),
            gallery: z.union([z.array(z.string()), z.string()]).optional(),
            files: z.union([z.array(z.string()), z.string()]).optional(),
            accent_color: z.union([z.number(), z.string()]).optional(),
            spoiler: z.boolean().optional(),
            navigation: z.union([z.object({ type: z.enum(["select", "pagination"]) }), z.string()]).optional(),
            link_buttons: z.union([z.array(z.object({
                label: z.string(),
                url: z.string(),
                emoji: z.string().optional(),
            })), z.string()]).optional(),
            action_rows: z.union([z.array(z.any()), z.string()]).optional(),
            game_state: z.union([z.record(z.string(), z.unknown()), z.string()]).optional(),
            handlers: z.union([z.record(z.string(), z.string()), z.string()]).optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["admin"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["the card was sent by this bot and exists in the artifacts store"],
        postconditions: ["card re-rendered with the merged spec"],
        async run(context, args) {
            const provided = args as Record<string, unknown>;
            const rearm = Boolean(provided.rearm);
            const changed = EDITABLE_FIELDS.filter((field) => provided[field] !== undefined);
            if (!changed.length && !rearm) {
                const error = "Nothing to change: pass at least one of title, summary, sections, navigation, link_buttons, gallery, files, accent_color, spoiler, action_rows, game_state, handlers, ttl_days, or rearm.";
                return { tool: T.artifact_edit, summary: error, data: { error }, errorMessage: error };
            }

            const messageId = String(args.message_id);
            const stored = await ArtifactStore.get(messageId);
            if (!stored) {
                const error = `No artifact card found for message ${messageId}.`;
                return { tool: T.artifact_edit, summary: error, data: { messageId, error }, errorMessage: error };
            }
            if (!context.guild || (stored.guildId && stored.guildId !== context.guild.id)) {
                const error = "This card belongs to another server.";
                return { tool: T.artifact_edit, summary: error, data: { messageId, error }, errorMessage: error };
            }

            let currentSpec: Record<string, unknown>;
            try {
                currentSpec = JSON.parse(stored.specJson || "{}") as Record<string, unknown>;
            } catch {
                const error = "The stored card spec is unreadable; the card cannot be edited.";
                return { tool: T.artifact_edit, summary: error, data: { messageId, error }, errorMessage: error };
            }

            const merged: Record<string, unknown> = { ...currentSpec };
            for (const field of changed) {
                merged[field] = provided[field];
            }
            // Rearm-only calls revalidate the stored spec itself: it was valid
            // when accepted, and the goal is just fresh controls.
            const validated = rearm && !changed.length
                ? validateArtifactSpec(currentSpec)
                : validateArtifactSpec(merged);
            if (!validated.ok) {
                return { tool: T.artifact_edit, summary: "Invalid merged spec.", data: { messageId, error: validated.error }, errorMessage: validated.error };
            }

            const channel = context.guild.channels.cache.get(stored.channelId);
            if (!channel || !channel.isTextBased() || !("messages" in channel)) {
                const error = "The card's channel was not found or is not text-based.";
                return { tool: T.artifact_edit, summary: error, data: { messageId, error }, errorMessage: error };
            }
            const message = await (channel as { messages: { fetch(id: string): Promise<unknown> } }).messages.fetch(messageId).catch(() => null) as
                | { edit(payload: unknown): Promise<unknown>; author?: { id: string }; url: string }
                | null;
            if (!message) {
                const error = "The card message no longer exists in that channel.";
                return { tool: T.artifact_edit, summary: error, data: { messageId, error }, errorMessage: error };
            }
            const botId = context.guild.client.user?.id;
            if (botId && message.author?.id !== botId) {
                const error = "Only artifact cards sent by this bot can be edited.";
                return { tool: T.artifact_edit, summary: error, data: { messageId, error }, errorMessage: error };
            }

            const sent = await applyArtifactEdit(message as never, validated.value);

            const expiresAt = provided.ttl_days !== undefined
                ? artifactExpiryTimestamp(validated.value.ttlDays)
                : stored.expiresAt;
            await ArtifactStore.updateSpec(messageId, JSON.stringify(validated.value), expiresAt);

            const channelMention = `<#${sent.channelId}>`;
            return {
                tool: T.artifact_edit,
                summary: rearm
                    ? `Artifact "${validated.value.title}" re-armed in ${channelMention}: controls re-rendered fresh, view reset to section 1.`
                    : `Artifact "${validated.value.title}" updated in ${channelMention} (changed: ${changed.join(", ")}).`,
                data: {
                    messageId: sent.messageId,
                    channelId: sent.channelId,
                    channelMention,
                    messageUrl: sent.messageUrl,
                    rearmed: rearm,
                    changed: changed.length ? changed : ["rearm"],
                    sections: validated.value.sections.length,
                    navigation: (validated.value as { navigation?: unknown }).navigation ?? null,
                    ttlDays: (validated.value as { ttlDays?: number }).ttlDays ?? 0,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "✏️", labelPt: "Editar cartão" },

    describeApproval(args: ToolArguments) {
        const messageId = typeof args.message_id === "string" ? args.message_id : "cartão";
        return `Editar cartão (mensagem ${messageId})`;
    },
};
