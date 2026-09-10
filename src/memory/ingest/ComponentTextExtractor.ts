import type { Message } from "discord.js";

/**
 * Flattens a message's Components V2 tree into plain text so artifact cards
 * and other component-only messages surface in the local retrieval index
 * (their raw `content` is always empty).
 */

interface RawComponent {
    type?: number;
    [key: string]: unknown;
}

const COMPONENT_TYPES = {
    ActionRow: 1,
    Button: 2,
    StringSelect: 3,
    UserSelect: 5,
    RoleSelect: 6,
    MentionableSelect: 7,
    ChannelSelect: 8,
    Section: 9,
    TextDisplay: 10,
    Thumbnail: 11,
    MediaGallery: 12,
    File: 13,
    Separator: 14,
    Container: 17,
} as const;

function toRawComponent(component: unknown): RawComponent | null {
    if (!component || typeof component !== "object") return null;
    const candidate = component as { toJSON?: () => unknown; data?: unknown };
    const raw = (typeof candidate.toJSON === "function" ? candidate.toJSON() : candidate.data ?? component) as unknown;
    return raw && typeof raw === "object" ? (raw as RawComponent) : null;
}

function childComponents(component: RawComponent): RawComponent[] {
    const children = component.components ?? component.items;
    if (!Array.isArray(children)) return [];
    return children
        .map(toRawComponent)
        .filter((entry): entry is RawComponent => entry !== null);
}

function pushText(out: string[], value: unknown): void {
    if (typeof value === "string" && value.trim()) out.push(value.trim());
}

function describeComponent(component: RawComponent, out: string[]): void {
    switch (component.type) {
        case COMPONENT_TYPES.ActionRow: {
            for (const child of childComponents(component)) describeComponent(child, out);
            break;
        }
        case COMPONENT_TYPES.Button: {
            const label = typeof component.label === "string" ? component.label : "";
            const url = typeof component.url === "string" ? component.url : "";
            pushText(out, url ? `${label} (${url})` : label);
            break;
        }
        case COMPONENT_TYPES.StringSelect: {
            pushText(out, component.placeholder);
            const options = Array.isArray(component.options) ? component.options : [];
            for (const option of options) {
                if (option && typeof option === "object") pushText(out, (option as RawComponent).label);
            }
            break;
        }
        case COMPONENT_TYPES.UserSelect:
        case COMPONENT_TYPES.RoleSelect:
        case COMPONENT_TYPES.MentionableSelect:
        case COMPONENT_TYPES.ChannelSelect: {
            pushText(out, component.placeholder);
            break;
        }
        case COMPONENT_TYPES.Section:
        case COMPONENT_TYPES.Container: {
            for (const child of childComponents(component)) describeComponent(child, out);
            break;
        }
        case COMPONENT_TYPES.TextDisplay: {
            pushText(out, component.content);
            break;
        }
        case COMPONENT_TYPES.Thumbnail: {
            const media = component.media as RawComponent | undefined;
            if (media) {
                pushText(out, media.description);
                pushText(out, media.url);
            }
            break;
        }
        case COMPONENT_TYPES.MediaGallery: {
            for (const child of childComponents(component)) {
                const media = child.media as RawComponent | undefined;
                if (media) {
                    pushText(out, media.description);
                    pushText(out, media.url);
                }
            }
            break;
        }
        case COMPONENT_TYPES.File: {
            const file = component.file as RawComponent | undefined;
            if (file) pushText(out, file.url);
            break;
        }
        default:
            break;
    }
}

/** Plain-text rendering of every text-bearing node in the component tree. */
export function extractComponentsText(message: Message): string {
    const components = message.components as readonly unknown[] | undefined;
    if (!components?.length) return "";
    const out: string[] = [];
    for (const component of components) {
        const raw = toRawComponent(component);
        if (raw) describeComponent(raw, out);
    }
    return out.join("\n");
}
