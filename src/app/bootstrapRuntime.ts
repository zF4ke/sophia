import { getAppConfig } from "@/app/AppConfig";
import { loadCommands } from "@/discord/loaders/commandLoader";
import { loadEvents } from "@/discord/loaders/eventLoader";
import { ArtifactStore } from "@/discord/artifacts/ArtifactStore";
import type { BotClient } from "@/shared/appTypes";

export async function bootstrapRuntime(client: BotClient): Promise<void> {
    const config = getAppConfig();
    await loadEvents(client);
    await client.login(config.discordToken);
    try {
        await loadCommands(client);
    } catch (error) {
        console.error("[commands] Failed to synchronize application commands. The bot will keep running.", error);
    }
    // Best-effort cleanup of expired artifact cards from previous sessions.
    void ArtifactStore.sweepExpired(client).catch((error) => {
        console.warn("[artifacts] TTL sweep failed (non-fatal):", (error as Error).message ?? error);
    });
}


