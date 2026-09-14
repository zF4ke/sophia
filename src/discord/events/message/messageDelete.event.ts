import type { Message, PartialMessage } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
export = {
    name: "messageDelete",
    async execute(message: Message | PartialMessage) {
        try { await DiscordMemoryService.deleteMessage(message.id, message.channelId, message.guildId); }
        catch (error) { console.error("[messageDelete] Source invalidation failed", error); }
    },
};
