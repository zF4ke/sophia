import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

export = {
    name: "clientReady",
    once: true,
    async execute(client: BotClient) {
        if (!client.user) return;

        console.log(`${client.user.username} is online.`);
        await SecurityService.initialize();
        await DiscordBackfillCrawler.start(client);
    },
};
