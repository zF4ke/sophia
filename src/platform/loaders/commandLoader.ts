import { AppPaths } from "@/app/AppPaths";
import { discoverModuleFiles } from "@/platform/loaders/moduleDiscovery";
import type { BotClient, BotCommand } from "@/shared/appTypes";
import { pathToFileURL } from "url";

export async function loadCommands(client: BotClient): Promise<void> {
    const loadedCommands: string[] = [];
    const commandPayloads: unknown[] = [];

    for (const filePath of discoverModuleFiles(AppPaths.commandModulesRoot, ".command")) {
        const loaded = await import(pathToFileURL(filePath).href);
        const command = (loaded.default || loaded) as BotCommand;

        if (!command?.data?.name || typeof command.execute !== "function") {
            continue;
        }

        client.commands.set(command.data.name, command);
        commandPayloads.push(command.data.toJSON());
        loadedCommands.push(command.data.name);
    }

    if (client.application) {
        await client.application.commands.set(commandPayloads as any[]);
    }

    console.log(`Loaded Commands: ${loadedCommands.join(", ")}`);
}
