import type { Guild } from "discord.js";
export async function enrichMessagesWithGuildMembers(guild: Guild | null, messages: Array<any>): Promise<void> {
    if (!guild?.members?.fetch) {
        return;
    }

    const memberPromises = new Map<string, Promise<any>>();

    for (const message of messages) {
        const authorId = typeof message?.author?.id === "string" ? message.author.id : null;
        if (!authorId || message.member) {
            continue;
        }

        if (!memberPromises.has(authorId)) {
            memberPromises.set(
                authorId,
                Promise.resolve(guild.members.cache?.get?.(authorId) ?? null).then(async (cached) => {
                    if (cached) {
                        return cached;
                    }

                    try {
                        return await guild.members.fetch(authorId);
                    } catch {
                        return null;
                    }
                })
            );
        }
    }

    if (!memberPromises.size) {
        return;
    }

    for (const message of messages) {
        const authorId = typeof message?.author?.id === "string" ? message.author.id : null;
        if (!authorId || message.member || !memberPromises.has(authorId)) {
            continue;
        }

        const resolved = await memberPromises.get(authorId);
        try {
            message.member = resolved;
        } catch {
            // Discord.js Message objects expose `member` as a getter-only
            // property on cached instances.  Fall back to a non-enumerable
            // shadow property so downstream code can still read it.
            Object.defineProperty(message, "member", {
                value: resolved,
                writable: true,
                configurable: true,
            });
        }
    }
}
