import { z } from "zod";
import { captureDerivedSources } from "@/security/DerivedSources";
import { T } from "@/shared/discordTools";
import { sendInteractiveArtifact } from "@/discord/artifacts/ArtifactSession";
import { validateArtifactSpec } from "@/discord/artifacts/ArtifactBuilder";
import type { ToolArguments } from "@/runtime/contracts";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        channel_id: {
            type: "string",
            description: "Target channel ID. Omit to send to the current channel.",
        },
        title: {
            type: "string",
            description: "Card title, max 120 chars.",
        },
        summary: {
            type: "string",
            description: "Optional one-paragraph lead-in under the title, max 400 chars.",
        },
        sections: {
            type: "array",
            description: "1 to 50 content sections. With navigation, each section becomes a tab or page.",
            items: {
                type: "object",
                properties: {
                    heading: { type: "string", description: "Optional section heading, max 80 chars. Also used as the tab label." },
                    body: { type: "string", description: "Section body, markdown allowed, max 3000 chars." },
                    thumbnail_url: { type: "string", description: "Optional direct https image URL rendered top-right of this section like an embed thumbnail (corner image). Sources: avatar URL from get_member_profile, attachment URLs from retrieve_messages results, or any direct image link (.png/.jpg/.jpeg/.gif/.webp). A page containing the image is NOT an image." },
                    accessory_button: {
                        type: "object",
                        description: "Optional button shown as the section accessory (right side). Provide either thumbnail_url or accessory_button, not both. The button can be a Link (url) or an interactive custom button (customId).",
                        properties: {
                            label: { type: "string", description: "Button label, max 80 chars." },
                            url: { type: "string", description: "Absolute https URL for a Link button." },
                            customId: { type: "string", description: "Custom id for an interactive button (prefix action: or game: for minigames). Omit for Link buttons." },
                            style: { type: "number", description: "Button style: 1 Primary, 2 Secondary, 3 Success, 4 Danger, 5 Link. Default Primary for interactive." },
                            emoji: { type: "string", description: "Optional emoji." },
                        },
                        required: ["label"],
                    },
                },
                required: ["body"],
            },
        },
        gallery: {
            type: "array",
            description: "Optional image grid rendered below the content (max 10 direct https image URLs). Types: png, jpg, webp, gif (animates), avif.",
            items: { type: "string" },
        },
        files: {
            type: "array",
            description: "Optional file cards rendered below the gallery (max 10 https file URLs, e.g. Discord attachment links). Files are re-uploaded with the message.",
            items: { type: "string" },
        },
        accent_color: {
            type: "number",
            description: "Optional container accent color as an integer (0 to 0xffffff). Pick deliberately: red for danger dossiers, gold for awards, muted colors for reference cards.",
        },
        spoiler: {
            type: "boolean",
            description: "Optional: gray out the whole card (spoiler style). Use for genuinely spoilery or sensitive content.",
        },
        navigation: {
            type: "object",
            description: "Optional interactivity for 2+ sections. {type:'select'} adds a dropdown tab switcher; {type:'pagination'} adds prev/next buttons. Omit to render all sections at once.",
            properties: {
                type: { type: "string", enum: ["select", "pagination"], description: "Navigation style." },
            },
            required: ["type"],
        },
        link_buttons: {
            type: "array",
            description: "Optional https link buttons (max 5) shown under the content.",
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
        action_rows: {
            type: "array",
            description: "Optional fully custom interactive rows (max 5). Each row is either buttons (max 5) or a select menu (string/user/role/mentionable/channel). Buttons with url are Link buttons; with customId they are interactive (prefix action: for plain acks, game: for minigame state). Selects with customId starting game: also mutate gameState.",
            items: {
                type: "object",
                properties: {
                    type: { type: "string", enum: ["buttons", "stringSelect", "userSelect", "roleSelect", "mentionableSelect", "channelSelect"], description: "Row type." },
                    buttons: {
                        type: "array",
                        description: "Buttons for type=buttons (max 5).",
                        items: {
                            type: "object",
                            properties: {
                                label: { type: "string", description: "Button label, max 80 chars." },
                                style: { type: "number", description: "1 Primary, 2 Secondary, 3 Success, 4 Danger, 5 Link." },
                                customId: { type: "string", description: "Custom id for interactive buttons. Use prefix action: or game: for handler routing." },
                                url: { type: "string", description: "https URL for Link buttons." },
                                emoji: { type: "string" },
                                disabled: { type: "boolean" },
                            },
                            required: ["label"],
                        },
                    },
                    customId: { type: "string", description: "Custom id for select rows." },
                    placeholder: { type: "string" },
                    minValues: { type: "number" },
                    maxValues: { type: "number" },
                    channelTypes: { type: "array", items: { type: "number" } },
                    options: {
                        type: "array",
                        description: "Options for stringSelect (max 25).",
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
            description: "Optional initial JSON state for minigames and scripted buttons (max ~4000 chars serialized). Persisted per card. Available to handlers as `state`.",
            properties: {},
            required: [],
        },
        handlers: {
            type: "object",
            description: "Optional sandboxed JS handlers keyed by customId (max 10, each max 4000 chars). Executed on click in a bare VM (no require/process/network, 100ms timeout). Available inside: state (persisted object, mutate freely), user {id, username}, values (select choices), customId, cardId; helpers reply(text), send(channelId, text), setTitle/setSummary/setSection(n)/setAccent(color)/setSpoiler(bool), log(text). Example counter: \"state.count = (state.count||0)+1; reply(`Rodada ${state.count}`)\".",
            properties: {},
            required: [],
        },
        ttl_days: {
            type: "number",
            description: "Retention in days: the card is auto-deleted after this long. 0 (default) keeps it forever.",
        },
    },
    required: ["title", "sections"],
} as const;

export const artifactSendTool: ToolDefinition = {
    name: T.artifact_send,
    publicationTarget: (context, args) => String(args.channel_id || context.currentChannelId || ""),

    catalog: {
        effect: "write",
        description: "Send an interactive Components V2 card covering the full component surface: text, sections with thumbnails or button accessories, galleries, files, custom buttons and all select types, game state, and navigation. Built from retrieved evidence.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Render and send a rich Discord card (artifact) using the full Components V2 surface. Provide title + sections; optionally add thumbnails, gallery, files, accent_color, spoiler, navigation, link_buttons, action_rows (buttons and all select types for minigames), and game_state. The spec is validated before sending; validation errors come back to you to fix and retry.",
        parameters,
    },

    capability: {
        description: "Send an interactive artifact card (full Components V2).",
        inputSchema: z.object({
            channel_id: z.string().optional(),
            title: z.string(),
            summary: z.string().optional(),
            // Models sometimes serialize these to JSON strings; the string form
            // is accepted here and recovered by validateArtifactSpec.
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
            })).min(1), z.string()]),
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
            ttl_days: z.number().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "write",
        authRequirements: ["actor_grant"],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: ["guild context should exist"],
        postconditions: ["artifact card sent to the target channel"],
        async run(context, args) {
            const validated = validateArtifactSpec(args);
            if (!validated.ok) {
                return { tool: T.artifact_send, summary: `Invalid artifact spec: ${validated.error}`, data: { error: validated.error }, errorMessage: validated.error };
            }
            const spec = validated.value;

            if (!context.guild) {
                return { tool: T.artifact_send, summary: "No guild context.", data: null, errorMessage: "No guild context." };
            }
            const targetId = typeof args.channel_id === "string" && args.channel_id.trim() ? args.channel_id.trim() : context.currentChannelId;
            if (!targetId) {
                return { tool: T.artifact_send, summary: "No target channel.", data: null, errorMessage: "No channel_id given and no current channel." };
            }
            const channel = context.guild.channels.cache.get(targetId);
            if (!channel || !channel.isTextBased() || !("send" in channel)) {
                return { tool: T.artifact_send, summary: "Channel not found or not text-based.", data: null, errorMessage: "Channel not found or not text-based." };
            }

            const sent = await sendInteractiveArtifact(channel, spec, {
                guildId: context.guild.id,
                ownerId: context.actorId,
                sources: await captureDerivedSources(context),
                ttlDays: spec.ttlDays ?? 0,
            });

            const channelMention = `<#${sent.channelId}>`;
            return {
                tool: T.artifact_send,
                summary: `Artifact "${spec.title}" sent in ${channelMention} (${spec.sections.length} section(s)${spec.navigation ? `, ${spec.navigation.type} nav` : ""}${spec.ttlDays ? `, TTL ${spec.ttlDays}d` : ""}).`,
                errorMessage: sent.persistenceError,
                data: {
                    messageId: sent.messageId,
                    channelId: sent.channelId,
                    channelMention,
                    messageUrl: sent.messageUrl,
                    sections: spec.sections.length,
                    navigation: spec.navigation?.type ?? null,
                    ttlDays: spec.ttlDays ?? 0,
                },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "🗂️", labelPt: "Enviar cartão" },

    describeApproval(args: ToolArguments) {
        const title = typeof args.title === "string" ? args.title : "cartão";
        return `Enviar cartão "${title}" em <#${args.channel_id ?? "canal atual"}>`;
    },
};
