import { getAppConfig } from "@/app/AppConfig";
import { loadCommands } from "@/platform/loaders/commandLoader";
import { loadEvents } from "@/platform/loaders/eventLoader";
import type { BotClient } from "@/shared/appTypes";

export async function bootstrapRuntime(client: BotClient): Promise<void> {
    const config = getAppConfig();
    await loadEvents(client);
    await client.login(config.discordToken);
    await loadCommands(client);
}
