import { ApplicationCommandType } from "discord.js";
import { AppPaths } from "@/app/AppPaths";
import { discoverModuleFiles } from "@/discord/loaders/moduleDiscovery";
import type { BotClient, BotCommand } from "@/shared/appTypes";
import { pathToFileURL } from "url";

// Discord's Primary Entry Point command type. Kept as a named alias of the
// discord.js enum so bulk command sync preserves it instead of dropping it.
const PRIMARY_ENTRY_POINT = ApplicationCommandType.PrimaryEntryPoint;

type ExistingCommand = {
    type: number;
    name: string;
    description?: string;
    handler?: number | null;
    integrationTypes?: number[] | null;
    contexts?: number[] | null;
};

function entryPointPayload(command: ExistingCommand): Record<string, unknown> {
    return {
        type: command.type,
        name: command.name,
        description: command.description || "Launch activity",
        ...(command.handler == null ? {} : { handler: command.handler }),
        ...(command.integrationTypes == null ? {} : { integration_types: command.integrationTypes }),
        ...(command.contexts == null ? {} : { contexts: command.contexts }),
    };
}

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
        const manager = client.application.commands;
        if (typeof manager.fetch === "function") {
            const existing = await manager.fetch();
            for (const command of existing.values()) {
                if (command.type === PRIMARY_ENTRY_POINT) {
                    commandPayloads.push(entryPointPayload(command));
                }
            }
        }
        await manager.set(commandPayloads as any[]);
    }

    console.log(`Loaded Commands: ${loadedCommands.join(", ")}`);
}

