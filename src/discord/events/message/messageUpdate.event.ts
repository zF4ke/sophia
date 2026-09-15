import type { Message, PartialMessage } from "discord.js";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { isGuildAllowed } from "@/security/guildAllowlist";
import { ExecutionControl } from "@/runtime/ExecutionControl";

export = {
    name: "messageUpdate",
    async execute(_previous: Message | PartialMessage, updated: Message | PartialMessage) {
        if (!isGuildAllowed(updated.guildId)) return;
        try {
            const message = updated.partial ? await updated.fetch() : updated;
            await DiscordMemoryService.ingestMessage(message);
        } catch (error) {
            ExecutionControl.invalidateSource(updated.id, { kind: "unavailable", url: updated.url });
            console.error("[messageUpdate] Could not refresh edited source", error);
        }
    },
};
