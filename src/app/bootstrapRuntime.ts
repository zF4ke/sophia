import { getAppConfig } from "@/app/AppConfig";
import { loadCommands } from "@/discord/loaders/commandLoader";
import { loadEvents } from "@/discord/loaders/eventLoader";
import type { BotClient } from "@/shared/appTypes";

export async function bootstrapRuntime(client: BotClient): Promise<void> {
    const config = getAppConfig();
    await loadEvents(client);
    await client.login(config.discordToken);
    await loadCommands(client);
}


