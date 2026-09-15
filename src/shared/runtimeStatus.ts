import type { Message } from "discord.js";

/** Bot-owned task controls are transient UI, never conversation evidence. */
export function isRuntimeStatus(message: Message): boolean {
    if (!message.client?.user || message.author?.id !== message.client.user.id) return false;
    const ids = (message.components ?? []).flatMap(row => "components" in row ? row.components.flatMap(component => "customId" in component && typeof component.customId === "string" ? [component.customId] : []) : []);
    return ids.some(id => id.startsWith("task:stop:") && ids.includes(id.replace("task:stop:", "task:details:")));
}
