import type { ToolArguments } from "@/runtime/contracts";

const fields = new Set(["channel_id", "author_id", "member_id", "message_id", "role_id", "thread_id", "mentions", "before", "after", "authorId", "aroundMessageId"]);
const arrays = new Set(["message_ids", "channelIds", "excludedMessageIds", "role_ids"]);
function unwrap(value: string): string {
    return /^<(?:#|@!?|@&)(\d{15,22})>$/.exec(value)?.[1] ?? value;
}

/** Only unwrap complete Discord mentions. Never repair a corrupted identifier by guessing digits. */
export function normalizeDiscordIdentifiers(args: ToolArguments): void {
    for (const [key, value] of Object.entries(args)) {
        if (fields.has(key) && typeof value === "string") args[key] = unwrap(value);
        if (arrays.has(key) && Array.isArray(value)) args[key] = value.map(item => typeof item === "string" ? unwrap(item) : item);
    }
}
