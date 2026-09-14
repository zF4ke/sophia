import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ChannelSelectMenuBuilder,
    ContainerBuilder,
    FileBuilder,
    MediaGalleryBuilder,
    MediaGalleryItemBuilder,
    MentionableSelectMenuBuilder,
    RoleSelectMenuBuilder,
    SectionBuilder,
    SeparatorBuilder,
    SeparatorSpacingSize,
    StringSelectMenuBuilder,
    TextDisplayBuilder,
    ThumbnailBuilder,
    UserSelectMenuBuilder,
} from "discord.js";
import { SCRIPT_LIMITS } from "./ArtifactScript";

export interface ArtifactSection {
    heading?: string;
    body: string;
    /** Direct https image URL rendered top-right of this section, like an embed thumbnail. Mutually exclusive with accessoryButton. */
    thumbnailUrl?: string;
    /** Button shown as the section accessory (right side). Provide either thumbnailUrl or accessoryButton, not both. */
    accessoryButton?: { label: string; url?: string; customId?: string; style?: number; emoji?: string; disabled?: boolean };
}

export type ArtifactActionRow =
    | { type: "buttons"; buttons: Array<{ label: string; style?: number; customId?: string; url?: string; emoji?: string; disabled?: boolean }> }
    | { type: "stringSelect"; customId?: string; placeholder?: string; minValues?: number; maxValues?: number; disabled?: boolean; options: Array<{ label: string; value: string; description?: string; emoji?: string; default?: boolean }> }
    | { type: "userSelect"; customId?: string; placeholder?: string; minValues?: number; maxValues?: number; disabled?: boolean }
    | { type: "roleSelect"; customId?: string; placeholder?: string; minValues?: number; maxValues?: number; disabled?: boolean }
    | { type: "mentionableSelect"; customId?: string; placeholder?: string; minValues?: number; maxValues?: number; disabled?: boolean }
    | { type: "channelSelect"; customId?: string; placeholder?: string; channelTypes?: number[]; minValues?: number; maxValues?: number; disabled?: boolean };

export interface ArtifactSpec {
    title: string;
    summary?: string;
    sections: ArtifactSection[];
    /** Image grid rendered below the content (direct https image URLs). */
    gallery?: string[];
    /** File cards rendered below the gallery. Files are re-uploaded with the message. */
    files?: string[];
    /** Container accent color, normalized to a hex integer by validation. */
    accentColor?: number;
    /** Gray out the whole card (spoiler style). */
    spoiler?: boolean;
    navigation?: { type: "select" | "pagination" };
    linkButtons?: Array<{ label: string; url: string; emoji?: string }>;
    /** Fully custom interactive rows: buttons, string/user/role/mentionable/channel selects. Rendered after navigation/link rows. */
    actionRows?: ArtifactActionRow[];
    /** Arbitrary JSON state for minigames. Persisted per card and available to interaction handlers. */
    gameState?: Record<string, unknown>;
    /**
     * Sandboxed JS handlers keyed by customId (e.g. "action:roll", "game:cell0").
     * Executed on click in the isolated container backend with state/reply/send helpers.
     */
    handlers?: Record<string, string>;
    ephemeral?: boolean;
    /** Retention hint in days. 0 or omitted means no expiry. */
    ttlDays?: number;
}

export const ARTIFACT_LIMITS = {
    maxTitleChars: 120,
    maxSummaryChars: 400,
    maxSections: 50,
    maxHeadingChars: 80,
    maxBodyChars: 3000,
    maxLinkButtons: 5,
    maxButtonLabelChars: 80,
    maxTotalChars: 3900,
    maxGalleryItems: 10,
    maxFiles: 10,
    maxActionRows: 10,
    maxButtonsPerRow: 5,
    maxSelectOptions: 25,
    maxMediaUrlChars: 2048,
    maxTtlDays: 365,
} as const;

const IMAGE_EXTENSION_PATTERN = /\.(png|jpe?g|gif|webp|avif)(\?|#|$)/i;
const IMAGE_HOST_PATTERN = /discordapp\.(?:com|net)|discord\.com\/assets|(?:i\.)?imgur\.com|i\.redd\.it|twimg\.com|ytimg\.com|googleusercontent\.com|tenor\.com|giphy\.com|unsplash\.com|spotifycdn\.com|discord\.com\/attachments/i;

/**
 * Discord Components V2 media needs a direct image link: a URL that resolves
 * to an image file, not a page containing one. Extensions are the reliable
 * static signal; known image CDNs cover extension-less Discord attachments.
 */
export function looksLikeImageUrl(url: string): boolean {
    return IMAGE_EXTENSION_PATTERN.test(url) || IMAGE_HOST_PATTERN.test(url);
}

function validateMediaUrl(url: string, label: string): string | null {
    if (!url.startsWith("https://")) return `${label} must be an https URL.`;
    if (url.length > ARTIFACT_LIMITS.maxMediaUrlChars) return `${label} exceeds ${ARTIFACT_LIMITS.maxMediaUrlChars} chars.`;
    if (!looksLikeImageUrl(url)) {
        return `${label} must be a direct image link (.png/.jpg/.jpeg/.gif/.webp or a known image CDN like cdn.discordapp.com) — a page containing the image is not an image.`;
    }
    return null;
}

function validateFileUrl(url: string, label: string): string | null {
    if (!url.startsWith("https://")) return `${label} must be an https URL.`;
    if (url.length > ARTIFACT_LIMITS.maxMediaUrlChars) return `${label} exceeds ${ARTIFACT_LIMITS.maxMediaUrlChars} chars.`;
    return null;
}

function validateButtonForSection(raw: Record<string, unknown>): string | null {
    const label = typeof raw.label === "string" ? raw.label.trim() : "";
    if (!label) return "Section accessoryButton needs a label.";
    if (label.length > ARTIFACT_LIMITS.maxButtonLabelChars) return `Section accessoryButton label exceeds ${ARTIFACT_LIMITS.maxButtonLabelChars} chars.`;
    const url = raw.url != null ? String(raw.url).trim() : "";
    const customId = raw.customId != null ? String(raw.customId).trim() : raw.custom_id != null ? String(raw.custom_id).trim() : "";
    if (!url && !customId) return "Section accessoryButton needs either a url (Link button) or a customId (interactive button).";
    if (url && !/^https:\/\/\S+$/.test(url)) return "Section accessoryButton url must be an absolute https URL.";
    if (raw.style != null) {
        const style = Number(raw.style);
        if (![1, 2, 3, 4, 5].includes(style)) return "Section accessoryButton style must be 1-5 (Primary/Secondary/Success/Danger/Link).";
    }
    return null;
}

function validateActionRow(entry: unknown): string | null {
    if (!entry || typeof entry !== "object") return "Each actionRow must be an object with a type.";
    const record = entry as Record<string, unknown>;
    const type = typeof record.type === "string" ? record.type : "";
    if (!["buttons", "stringSelect", "userSelect", "roleSelect", "mentionableSelect", "channelSelect"].includes(type)) {
        return `actionRow type must be one of buttons, stringSelect, userSelect, roleSelect, mentionableSelect, channelSelect. Got "${type}".`;
    }
    if (type === "buttons") {
        if (!Array.isArray(record.buttons)) return "buttons row needs a buttons array.";
        if (record.buttons.length === 0 || record.buttons.length > ARTIFACT_LIMITS.maxButtonsPerRow) return `buttons row needs 1-${ARTIFACT_LIMITS.maxButtonsPerRow} buttons.`;
        for (const button of record.buttons as unknown[]) {
            if (!button || typeof button !== "object") return "Each button must be an object.";
            const b = button as Record<string, unknown>;
            const label = typeof b.label === "string" ? b.label.trim() : "";
            if (!label) return "Each button needs a label.";
            if (label.length > ARTIFACT_LIMITS.maxButtonLabelChars) return `Button label exceeds ${ARTIFACT_LIMITS.maxButtonLabelChars} chars.`;
            const url = b.url != null ? String(b.url).trim() : "";
            const customId = b.customId != null ? String(b.customId).trim() : b.custom_id != null ? String(b.custom_id).trim() : "";
            if (!url && !customId) return `Button "${label}" needs either a url (Link) or a customId (interactive).`;
            if (url && !/^https:\/\/\S+$/.test(url)) return `Button "${label}" url must be an absolute https URL.`;
            if (customId && customId.length > 100) return `Button "${label}" customId exceeds 100 chars.`;
            if (b.style != null) {
                const style = Number(b.style);
                if (![1, 2, 3, 4, 5].includes(style)) return `Button "${label}" style must be 1-5.`;
            }
        }
    }
    if (type === "stringSelect") {
        if (!Array.isArray(record.options)) return "stringSelect needs an options array.";
        if (record.options.length === 0 || record.options.length > ARTIFACT_LIMITS.maxSelectOptions) return `stringSelect needs 1-${ARTIFACT_LIMITS.maxSelectOptions} options.`;
        for (const option of record.options as unknown[]) {
            const o = option as Record<string, unknown>;
            if (!o || typeof o.label !== "string" || !o.label.trim()) return "Each stringSelect option needs a label.";
            if (!o || typeof o.value !== "string" || !o.value.trim()) return "Each stringSelect option needs a value.";
            if (String(o.label).length > 100) return "Select option label exceeds 100 chars.";
            if (String(o.value).length > 100) return "Select option value exceeds 100 chars.";
        }
        const minValues = record.minValues != null ? Number(record.minValues) : record.min_values != null ? Number(record.min_values) : undefined;
        const maxValues = record.maxValues != null ? Number(record.maxValues) : record.max_values != null ? Number(record.max_values) : undefined;
        if (minValues != null && (!Number.isInteger(minValues) || minValues < 0 || minValues > 25)) return "stringSelect minValues must be 0-25.";
        if (maxValues != null && (!Number.isInteger(maxValues) || maxValues < 1 || maxValues > 25)) return "stringSelect maxValues must be 1-25.";
    }
    if (["userSelect", "roleSelect", "mentionableSelect", "channelSelect"].includes(type)) {
        const minValues = record.minValues != null ? Number(record.minValues) : record.min_values != null ? Number(record.min_values) : undefined;
        const maxValues = record.maxValues != null ? Number(record.maxValues) : record.max_values != null ? Number(record.max_values) : undefined;
        if (minValues != null && (!Number.isInteger(minValues) || minValues < 0 || minValues > 25)) return `${type} minValues must be 0-25.`;
        if (maxValues != null && (!Number.isInteger(maxValues) || maxValues < 1 || maxValues > 25)) return `${type} maxValues must be 1-25.`;
        if (type === "channelSelect" && record.channelTypes != null && record.channel_types != null) {
            // checked below
        }
        if (type === "channelSelect" && Array.isArray(record.channelTypes ?? record.channel_types)) {
            const types = (record.channelTypes ?? record.channel_types) as unknown[];
            if (types.some((value) => !Number.isInteger(Number(value)))) return "channelSelect channelTypes must be integers.";
        }
    }
    return null;
}

function normalizeAccentColor(value: unknown): { ok: true; value?: number } | { ok: false; error: string } {
    if (value == null) return { ok: true, value: undefined };
    if (typeof value === "number") {
        if (!Number.isInteger(value) || value < 0 || value > 0xffffff) {
            return { ok: false, error: "accent_color must be an integer between 0 and 0xffffff." };
        }
        return { ok: true, value };
    }
    if (typeof value === "string") {
        const match = /^#?([0-9a-fA-F]{6})$/.exec(value.trim());
        if (!match) return { ok: false, error: "accent_color must be a hex color like '#9aa7ff' or an integer." };
        return { ok: true, value: parseInt(match[1], 16) };
    }
    return { ok: false, error: "accent_color must be a hex color like '#9aa7ff' or an integer." };
}

/** Derive a safe attachment filename from a URL for File component uploads. */
export function attachmentFileNameFromUrl(url: string, index: number): string {
    let name = "file";
    try {
        const parsed = new URL(url);
        const base = decodeURIComponent(parsed.pathname.split("/").filter(Boolean).pop() ?? "");
        if (base) name = base;
    } catch {
        // Keep the fallback name.
    }
    const safe = name.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 180);
    return safe || `file-${index + 1}`;
}

/** Artifact fields the model sometimes serializes to JSON strings instead of passing as arrays/objects. */
const COERCIBLE_FIELDS = ["sections", "navigation", "link_buttons", "linkButtons", "gallery", "actionRows", "gameState", "handlers", "game_state"] as const;

function coerceJsonField(value: unknown): unknown {
    if (typeof value !== "string") return value;
    const trimmed = value.trim();
    if (!trimmed.startsWith("[") && !trimmed.startsWith("{")) return value;
    try {
        return JSON.parse(trimmed);
    } catch {
        return value;
    }
}

export function validateArtifactSpec(spec: unknown): { ok: true; value: ArtifactSpec } | { ok: false; error: string } {
    if (!spec || typeof spec !== "object") return { ok: false, error: "Artifact must be an object with title and sections." };
    const raw = spec as Record<string, unknown>;
    // JSON-string slippage tolerance: recover before rejecting.
    const candidate: Record<string, unknown> = { ...raw };
    for (const field of COERCIBLE_FIELDS) {
        if (candidate[field] !== undefined) candidate[field] = coerceJsonField(candidate[field]);
    }
    const title = typeof candidate.title === "string" ? candidate.title.trim() : "";
    if (!title) return { ok: false, error: "Artifact needs a non-empty title." };
    if (title.length > ARTIFACT_LIMITS.maxTitleChars) return { ok: false, error: `Title exceeds ${ARTIFACT_LIMITS.maxTitleChars} chars.` };
    const summary = candidate.summary == null ? undefined : String(candidate.summary).trim() || undefined;
    if (summary && summary.length > ARTIFACT_LIMITS.maxSummaryChars) return { ok: false, error: `Summary exceeds ${ARTIFACT_LIMITS.maxSummaryChars} chars.` };
    if (!Array.isArray(candidate.sections) || candidate.sections.length === 0) return { ok: false, error: "Artifact needs at least one section." };
    if (candidate.sections.length > ARTIFACT_LIMITS.maxSections) return { ok: false, error: `Artifact exceeds ${ARTIFACT_LIMITS.maxSections} sections.` };
    const sections: ArtifactSection[] = [];
    let totalChars = title.length + (summary?.length ?? 0);
    for (const entry of candidate.sections) {
        if (!entry || typeof entry !== "object") return { ok: false, error: "Each section must be an object with a body." };
        const record = entry as Record<string, unknown>;
        const body = typeof record.body === "string" ? record.body.trim() : "";
        if (!body) return { ok: false, error: "Each section needs a non-empty body." };
        if (body.length > ARTIFACT_LIMITS.maxBodyChars) return { ok: false, error: `Section body exceeds ${ARTIFACT_LIMITS.maxBodyChars} chars.` };
        const heading = typeof record.heading === "string" && record.heading.trim() ? record.heading.trim().slice(0, ARTIFACT_LIMITS.maxHeadingChars) : undefined;
        totalChars += body.length + (heading?.length ?? 0);
        let thumbnailUrl: string | undefined;
        if (record.thumbnail_url != null || record.thumbnailUrl != null) {
            const rawUrl = String(record.thumbnail_url ?? record.thumbnailUrl).trim();
            const mediaError = validateMediaUrl(rawUrl, "Section thumbnail_url");
            if (mediaError) return { ok: false, error: mediaError };
            thumbnailUrl = rawUrl;
        }
        let accessoryButton: ArtifactSection["accessoryButton"];
        if (record.accessory_button != null || record.accessoryButton != null) {
            const raw = (record.accessory_button ?? record.accessoryButton) as Record<string, unknown>;
            if (!raw || typeof raw !== "object") return { ok: false, error: "Section accessoryButton must be an object." };
            const buttonError = validateButtonForSection(raw);
            if (buttonError) return { ok: false, error: buttonError };
            accessoryButton = {
                label: String(raw.label).trim(),
                url: raw.url != null ? String(raw.url).trim() : undefined,
                customId: raw.customId != null ? String(raw.customId).trim() : raw.custom_id != null ? String(raw.custom_id).trim() : undefined,
                style: raw.style != null ? Number(raw.style) : undefined,
                emoji: raw.emoji != null ? String(raw.emoji).trim() : undefined,
                disabled: raw.disabled != null ? Boolean(raw.disabled) : undefined,
            };
        }
        if (thumbnailUrl && accessoryButton) {
            return { ok: false, error: "A section cannot have both thumbnailUrl and accessoryButton — pick one." };
        }
        sections.push({ heading, body, ...(thumbnailUrl ? { thumbnailUrl } : {}), ...(accessoryButton ? { accessoryButton } : {}) });
    }

    let gallery: string[] | undefined;
    if (candidate.gallery != null) {
        if (!Array.isArray(candidate.gallery)) return { ok: false, error: "gallery must be an array of image URLs." };
        if (candidate.gallery.length > ARTIFACT_LIMITS.maxGalleryItems) {
            return { ok: false, error: `gallery exceeds ${ARTIFACT_LIMITS.maxGalleryItems} images.` };
        }
        gallery = [];
        for (const entry of candidate.gallery) {
            const url = typeof entry === "string" ? entry.trim() : "";
            const mediaError = validateMediaUrl(url, "gallery image URL");
            if (mediaError) return { ok: false, error: mediaError };
            gallery.push(url);
        }
    }

    let files: string[] | undefined;
    if (candidate.files != null) {
        if (!Array.isArray(candidate.files)) return { ok: false, error: "files must be an array of https URLs." };
        if (candidate.files.length > ARTIFACT_LIMITS.maxFiles) {
            return { ok: false, error: `files exceeds ${ARTIFACT_LIMITS.maxFiles} files.` };
        }
        files = [];
        for (const entry of candidate.files) {
            const url = typeof entry === "string" ? entry.trim() : "";
            const fileError = validateFileUrl(url, "file URL");
            if (fileError) return { ok: false, error: fileError };
            files.push(url);
        }
    }

    let actionRows: ArtifactActionRow[] | undefined;
    if (candidate.action_rows != null || candidate.actionRows != null) {
        const raw = candidate.action_rows ?? candidate.actionRows;
        if (!Array.isArray(raw)) return { ok: false, error: "actionRows must be an array." };
        if (raw.length > ARTIFACT_LIMITS.maxActionRows) return { ok: false, error: `actionRows exceeds ${ARTIFACT_LIMITS.maxActionRows} rows.` };
        actionRows = [];
        for (const entry of raw) {
            const rowError = validateActionRow(entry);
            if (rowError) return { ok: false, error: rowError };
            const record = entry as Record<string, unknown>;
            const type = String(record.type);
            if (type === "buttons") {
                const buttons = record.buttons as Array<Record<string, unknown>>;
                actionRows.push({
                    type: "buttons",
                    buttons: buttons.map((button) => ({
                        label: String(button.label).trim(),
                        style: button.style != null ? Number(button.style) : undefined,
                        customId: button.customId != null ? String(button.customId).trim() : button.custom_id != null ? String(button.custom_id).trim() : undefined,
                        url: button.url != null ? String(button.url).trim() : undefined,
                        emoji: button.emoji != null ? String(button.emoji).trim() : undefined,
                        disabled: button.disabled != null ? Boolean(button.disabled) : undefined,
                    })),
                });
                totalChars += buttons.reduce((sum, button) => sum + String(button.label).length, 0);
            } else if (type === "stringSelect") {
                actionRows.push({
                    type: "stringSelect",
                    customId: record.customId != null ? String(record.customId).trim() : record.custom_id != null ? String(record.custom_id).trim() : undefined,
                    placeholder: record.placeholder != null ? String(record.placeholder).trim() : undefined,
                    minValues: record.minValues != null ? Number(record.minValues) : record.min_values != null ? Number(record.min_values) : undefined,
                    maxValues: record.maxValues != null ? Number(record.maxValues) : record.max_values != null ? Number(record.max_values) : undefined,
                    disabled: record.disabled != null ? Boolean(record.disabled) : undefined,
                    options: ((record.options as unknown[]) as Array<Record<string, unknown>>).map((option) => ({
                        label: String(option.label).trim(),
                        value: String(option.value).trim(),
                        description: option.description != null ? String(option.description).trim() : undefined,
                        emoji: option.emoji != null ? String(option.emoji).trim() : undefined,
                        default: option.default != null ? Boolean(option.default) : undefined,
                    })),
                });
            } else if (type === "userSelect" || type === "roleSelect" || type === "mentionableSelect") {
                actionRows.push({
                    type: type as ArtifactActionRow["type"],
                    customId: record.customId != null ? String(record.customId).trim() : record.custom_id != null ? String(record.custom_id).trim() : undefined,
                    placeholder: record.placeholder != null ? String(record.placeholder).trim() : undefined,
                    minValues: record.minValues != null ? Number(record.minValues) : record.min_values != null ? Number(record.min_values) : undefined,
                    maxValues: record.maxValues != null ? Number(record.maxValues) : record.max_values != null ? Number(record.max_values) : undefined,
                    disabled: record.disabled != null ? Boolean(record.disabled) : undefined,
                } as ArtifactActionRow);
            } else if (type === "channelSelect") {
                actionRows.push({
                    type: "channelSelect",
                    customId: record.customId != null ? String(record.customId).trim() : record.custom_id != null ? String(record.custom_id).trim() : undefined,
                    placeholder: record.placeholder != null ? String(record.placeholder).trim() : undefined,
                    channelTypes: Array.isArray(record.channelTypes) ? (record.channelTypes as number[]) : Array.isArray(record.channel_types) ? (record.channel_types as number[]) : undefined,
                    minValues: record.minValues != null ? Number(record.minValues) : record.min_values != null ? Number(record.min_values) : undefined,
                    maxValues: record.maxValues != null ? Number(record.maxValues) : record.max_values != null ? Number(record.max_values) : undefined,
                    disabled: record.disabled != null ? Boolean(record.disabled) : undefined,
                });
            }
        }
    }

    let gameState: Record<string, unknown> | undefined;
    if (candidate.game_state != null || candidate.gameState != null) {
        const raw = candidate.game_state ?? candidate.gameState;
        if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { ok: false, error: "gameState must be an object." };
        gameState = raw as Record<string, unknown>;
        const serialized = JSON.stringify(gameState);
        if (serialized.length > 4000) return { ok: false, error: "gameState is too large (max ~4000 chars serialized)." };
    }

    let handlers: Record<string, string> | undefined;
    if (candidate.handlers != null) {
        if (!candidate.handlers || typeof candidate.handlers !== "object" || Array.isArray(candidate.handlers)) {
            return { ok: false, error: "handlers must be an object mapping customId to JS source." };
        }
        const entries = Object.entries(candidate.handlers as Record<string, unknown>);
        if (entries.length > SCRIPT_LIMITS.maxHandlers) {
            return { ok: false, error: `handlers exceeds ${SCRIPT_LIMITS.maxHandlers} entries.` };
        }
        handlers = {};
        for (const [key, value] of entries) {
            if (!key.trim() || key.length > 100) return { ok: false, error: `handler key "${key.slice(0, 30)}" must be 1-100 chars.` };
            if (typeof value !== "string" || !value.trim()) return { ok: false, error: `handler "${key}" must be a non-empty JS string.` };
            if (value.length > SCRIPT_LIMITS.maxCodeChars) return { ok: false, error: `handler "${key}" exceeds ${SCRIPT_LIMITS.maxCodeChars} chars.` };
            handlers[key.trim()] = value;
        }
    }

    const accent = normalizeAccentColor(candidate.accent_color ?? candidate.accentColor);
    if (!accent.ok) return { ok: false, error: accent.error };
    const spoiler = candidate.spoiler == null ? false : Boolean(candidate.spoiler);

    let navigation: ArtifactSpec["navigation"];
    if (candidate.navigation != null) {
        if (typeof candidate.navigation !== "object") return { ok: false, error: "navigation must be an object." };
        const navType = (candidate.navigation as Record<string, unknown>).type;
        if (navType !== "select" && navType !== "pagination") {
            return { ok: false, error: "navigation.type must be 'select' or 'pagination'." };
        }
        if (sections.length < 2) return { ok: false, error: "Navigation needs at least 2 sections." };
        navigation = { type: navType };
    }

    let linkButtons: ArtifactSpec["linkButtons"];
    if (candidate.link_buttons != null || candidate.linkButtons != null) {
        const raw = candidate.link_buttons ?? candidate.linkButtons;
        if (!Array.isArray(raw)) return { ok: false, error: "link_buttons must be an array." };
        if (raw.length > ARTIFACT_LIMITS.maxLinkButtons) return { ok: false, error: `link_buttons exceeds ${ARTIFACT_LIMITS.maxLinkButtons} buttons.` };
        linkButtons = [];
        for (const entry of raw) {
            if (!entry || typeof entry !== "object") return { ok: false, error: "Each link button must be an object." };
            const record = entry as Record<string, unknown>;
            const label = typeof record.label === "string" ? record.label.trim() : "";
            const url = typeof record.url === "string" ? record.url.trim() : "";
            if (!label) return { ok: false, error: "Each link button needs a label." };
            if (label.length > ARTIFACT_LIMITS.maxButtonLabelChars) return { ok: false, error: `Link button label exceeds ${ARTIFACT_LIMITS.maxButtonLabelChars} chars.` };
            if (!/^https:\/\/\S+$/.test(url)) return { ok: false, error: "Link button URLs must be absolute https URLs." };
            const emoji = typeof record.emoji === "string" && record.emoji.trim() ? record.emoji.trim() : undefined;
            totalChars += label.length;
            linkButtons.push(emoji ? { label, url, emoji } : { label, url });
        }
    }

    if (totalChars > ARTIFACT_LIMITS.maxTotalChars) {
        return { ok: false, error: `Card exceeds the ${ARTIFACT_LIMITS.maxTotalChars} char Components V2 budget (has ${totalChars}). Trim sections or split into two cards.` };
    }

    const ephemeral = candidate.ephemeral == null ? false : Boolean(candidate.ephemeral);
    const ttlRaw = candidate.ttl_days ?? candidate.ttlDays ?? 0;
    const ttlNumber = typeof ttlRaw === "number" ? ttlRaw : Number(ttlRaw);
    if (!Number.isFinite(ttlNumber) || ttlNumber < 0 || ttlNumber > ARTIFACT_LIMITS.maxTtlDays) {
        return { ok: false, error: `ttl_days must be between 0 and ${ARTIFACT_LIMITS.maxTtlDays}.` };
    }
    return {
        ok: true,
        value: {
            title,
            summary,
            sections,
            gallery,
            files,
            accentColor: accent.value,
            spoiler,
            navigation,
            linkButtons,
            actionRows,
            gameState,
            handlers,
            ephemeral,
            ttlDays: Math.floor(ttlNumber),
        },
    };
}

/** Renders one section's text, with a thumbnail or button accessory when present. */
function renderSectionInto(
    container: ContainerBuilder,
    entry: ArtifactSection,
): void {
    const lines: string[] = [];
    if (entry.heading) lines.push(`**${entry.heading}**`);
    lines.push(entry.body);
    const textDisplay = new TextDisplayBuilder().setContent(lines.join("\n"));
    if (entry.thumbnailUrl) {
        container.addSectionComponents(
            new SectionBuilder()
                .addTextDisplayComponents(textDisplay)
                .setThumbnailAccessory(
                    new ThumbnailBuilder().setURL(entry.thumbnailUrl).setDescription(entry.heading ?? "Imagem"),
                ),
        );
    } else if (entry.accessoryButton) {
        const button = entry.accessoryButton;
        const builder = new ButtonBuilder().setLabel(button.label).setStyle((button.style as never) ?? ButtonStyle.Primary);
        if (button.url) builder.setURL(button.url).setStyle(ButtonStyle.Link);
        else if (button.customId) builder.setCustomId(button.customId);
        if (button.emoji) builder.setEmoji(button.emoji);
        if (button.disabled) builder.setDisabled(true);
        container.addSectionComponents(
            new SectionBuilder().addTextDisplayComponents(textDisplay).setButtonAccessory(builder),
        );
    } else {
        container.addTextDisplayComponents(textDisplay);
    }
}

function renderMediaInto(container: ContainerBuilder, spec: ArtifactSpec): void {
    if (spec.gallery?.length) {
        container.addMediaGalleryComponents(
            new MediaGalleryBuilder().addItems(
                ...spec.gallery.map((url, index) =>
                    new MediaGalleryItemBuilder().setURL(url).setDescription(`Imagem ${index + 1}`),
                ),
            ),
        );
    }
    if (spec.files?.length) {
        for (const url of spec.files) {
            container.addFileComponents(
                new FileBuilder().setURL(`attachment://${attachmentFileNameFromUrl(url, spec.files.indexOf(url))}`),
            );
        }
    }
}

function renderActionRowsInto(
    components: BuiltArtifactComponents["components"],
    spec: ArtifactSpec,
    nonce: string,
    disabled: boolean,
): void {
    if (!spec.actionRows?.length) return;
    spec.actionRows.forEach((row, rowIndex) => {
        if (row.type === "buttons") {
            components.push(
                new ActionRowBuilder<ButtonBuilder>().addComponents(
                    ...row.buttons.map((button, colIndex) => {
                        const builder = new ButtonBuilder().setLabel(button.label);
                        if (button.url) {
                            builder.setStyle(ButtonStyle.Link).setURL(button.url);
                        } else {
                            const customId = button.customId ?? `action:${nonce}:${rowIndex}:${colIndex}`;
                            builder.setCustomId(customId).setStyle((button.style as never) ?? ButtonStyle.Primary);
                        }
                        if (button.emoji) builder.setEmoji(button.emoji);
                        if (disabled || button.disabled) builder.setDisabled(true);
                        return builder;
                    }),
                ),
            );
        } else if (row.type === "stringSelect") {
            const builder = new StringSelectMenuBuilder()
                .setCustomId(row.customId ?? `action:${nonce}:${rowIndex}:select`)
                .setDisabled(disabled || Boolean(row.disabled));
            if (row.placeholder) builder.setPlaceholder(row.placeholder);
            if (row.minValues != null) builder.setMinValues(row.minValues);
            if (row.maxValues != null) builder.setMaxValues(row.maxValues);
            builder.addOptions(
                row.options.map((option) => {
                    const data: Record<string, unknown> = { label: option.label, value: option.value };
                    if (option.description) data.description = option.description;
                    if (option.emoji) data.emoji = { name: option.emoji };
                    if (option.default) data.default = true;
                    return data as never;
                }),
            );
            components.push(new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(builder));
        } else if (row.type === "userSelect") {
            const builder = new UserSelectMenuBuilder()
                .setCustomId(row.customId ?? `action:${nonce}:${rowIndex}:user`)
                .setDisabled(disabled || Boolean(row.disabled));
            if (row.placeholder) builder.setPlaceholder(row.placeholder);
            if (row.minValues != null) builder.setMinValues(row.minValues);
            if (row.maxValues != null) builder.setMaxValues(row.maxValues);
            components.push(new ActionRowBuilder<UserSelectMenuBuilder>().addComponents(builder));
        } else if (row.type === "roleSelect") {
            const builder = new RoleSelectMenuBuilder()
                .setCustomId(row.customId ?? `action:${nonce}:${rowIndex}:role`)
                .setDisabled(disabled || Boolean(row.disabled));
            if (row.placeholder) builder.setPlaceholder(row.placeholder);
            if (row.minValues != null) builder.setMinValues(row.minValues);
            if (row.maxValues != null) builder.setMaxValues(row.maxValues);
            components.push(new ActionRowBuilder<RoleSelectMenuBuilder>().addComponents(builder));
        } else if (row.type === "mentionableSelect") {
            const builder = new MentionableSelectMenuBuilder()
                .setCustomId(row.customId ?? `action:${nonce}:${rowIndex}:mentionable`)
                .setDisabled(disabled || Boolean(row.disabled));
            if (row.placeholder) builder.setPlaceholder(row.placeholder);
            if (row.minValues != null) builder.setMinValues(row.minValues);
            if (row.maxValues != null) builder.setMaxValues(row.maxValues);
            components.push(new ActionRowBuilder<MentionableSelectMenuBuilder>().addComponents(builder));
        } else if (row.type === "channelSelect") {
            const builder = new ChannelSelectMenuBuilder()
                .setCustomId(row.customId ?? `action:${nonce}:${rowIndex}:channel`)
                .setDisabled(disabled || Boolean(row.disabled));
            if (row.placeholder) builder.setPlaceholder(row.placeholder);
            if (row.channelTypes?.length) builder.setChannelTypes(row.channelTypes as never);
            if (row.minValues != null) builder.setMinValues(row.minValues);
            if (row.maxValues != null) builder.setMaxValues(row.maxValues);
            components.push(new ActionRowBuilder<ChannelSelectMenuBuilder>().addComponents(builder));
        }
    });
}

/** Single-view container rendering every section (used by tests and non-interactive sends). */
export function buildArtifactContainer(spec: ArtifactSpec): ContainerBuilder {
    const container = new ContainerBuilder().setAccentColor(0x9aa7ff);
    const blocks: string[] = [`## ${spec.title}`];
    if (spec.summary) blocks.push(spec.summary);
    container.addTextDisplayComponents(new TextDisplayBuilder().setContent(blocks.join("\n")));
    spec.sections.forEach((section, index) => {
        if (index > 0) {
            container.addSeparatorComponents(new SeparatorBuilder().setDivider(true).setSpacing(SeparatorSpacingSize.Small));
        }
        renderSectionInto(container, section);
    });
    renderMediaInto(container, spec);
    return container;
}

export interface ArtifactViewState {
    section: number;
}

export interface BuiltArtifactComponents {
    components: Array<ContainerBuilder | ActionRowBuilder<any>>;
    pageCount: number;
    section: number;
}

/**
 * Renders the artifact for a given view state. `nonce` namespaces the custom
 * IDs so the interaction collector can route updates back to this card, and
 * `disabled` renders every control inert (used when the session expires).
 */
export function buildArtifactComponents(
    spec: ArtifactSpec,
    state: ArtifactViewState = { section: 0 },
    nonce: string,
    opts: { disabled?: boolean } = {},
): BuiltArtifactComponents {
    const nav = spec.navigation?.type;
    const pageCount = spec.sections.length;
    const section = Math.min(Math.max(state.section, 0), pageCount - 1);
    const disabled = opts.disabled === true;

    const container = new ContainerBuilder();
    if (spec.accentColor != null) container.setAccentColor(spec.accentColor);
    else container.setAccentColor(0x9aa7ff);
    if (spec.spoiler) container.setSpoiler(true);
    const blocks: string[] = [`## ${spec.title}`];
    if (spec.summary) blocks.push(spec.summary);
    container.addTextDisplayComponents(new TextDisplayBuilder().setContent(blocks.join("\n")));

    const visible = nav ? [spec.sections[section]] : spec.sections;
    visible.forEach((entry, index) => {
        if (index > 0 || (nav && spec.summary)) {
            container.addSeparatorComponents(new SeparatorBuilder().setDivider(true).setSpacing(SeparatorSpacingSize.Small));
        }
        renderSectionInto(container, entry);
    });

    renderMediaInto(container, spec);

    const components: BuiltArtifactComponents["components"] = [container];

    if (nav === "pagination" && pageCount > 1) {
        components.push(
            new ActionRowBuilder<ButtonBuilder>().addComponents(
                new ButtonBuilder()
                    .setCustomId(`artifact:${nonce}:prev`)
                    .setLabel("◀")
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(disabled || section === 0),
                new ButtonBuilder()
                    .setCustomId(`artifact:${nonce}:count`)
                    .setLabel(`${section + 1}/${pageCount}`)
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(true),
                new ButtonBuilder()
                    .setCustomId(`artifact:${nonce}:next`)
                    .setLabel("▶")
                    .setStyle(ButtonStyle.Secondary)
                    .setDisabled(disabled || section === pageCount - 1),
            ),
        );
    }

    if (nav === "select" && pageCount > 1) {
        components.push(
            new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                new StringSelectMenuBuilder()
                    .setCustomId(`artifact:${nonce}:tab`)
                    .setPlaceholder("Escolher secção")
                    .setDisabled(disabled)
                    .addOptions(
                        spec.sections.map((entry, index) => ({
                            label: (entry.heading || `Secção ${index + 1}`).slice(0, 100),
                            value: String(index),
                            default: index === section,
                        })),
                    ),
            ),
        );
    }

    if (spec.linkButtons?.length) {
        components.push(
            new ActionRowBuilder<ButtonBuilder>().addComponents(
                ...spec.linkButtons.map((button) => {
                    const builder = new ButtonBuilder()
                        .setLabel(button.label)
                        .setStyle(ButtonStyle.Link)
                        .setURL(button.url);
                    if (button.emoji) builder.setEmoji(button.emoji);
                    if (disabled) builder.setDisabled(true);
                    return builder;
                }),
            ),
        );
    }

    renderActionRowsInto(components, spec, nonce, disabled);

    return { components, pageCount, section };
}

export function artifactExpiryTimestamp(ttlDays: number | undefined, fromTimestamp = Date.now()): number | null {
    if (!ttlDays || ttlDays <= 0) return null;
    return fromTimestamp + Math.floor(ttlDays) * 86_400_000;
}
