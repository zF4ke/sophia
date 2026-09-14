import type { Collection, Message, PartialMessage } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
export = {
    name: "messageDeleteBulk",
    async execute(messages: Collection<string, Message | PartialMessage>) {
        for (const message of messages.values()) {
            try { await DiscordMemoryService.deleteMessage(message.id, message.channelId, message.guildId); }
            catch (error) { console.error("[messageDeleteBulk] Source invalidation failed", error); }
        }
    },
};
