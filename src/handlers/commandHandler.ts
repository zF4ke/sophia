import fs from "fs";
import path from "path";
import type { BotClient, BotCommand } from "@/types/app";

export async function loadCommands(client: BotClient): Promise<void> {
    const commandsRoot = path.join(process.cwd(), "src", "commands");
    const loadedCommands: string[] = [];
    const commandPayloads: unknown[] = [];

    const folders = fs.readdirSync(commandsRoot, { withFileTypes: true });
    for (const folder of folders) {
        if (!folder.isDirectory()) {
            continue;
        }

        const folderPath = path.join(commandsRoot, folder.name);
        const files = fs
            .readdirSync(folderPath)
            .filter((file) => file.endsWith(".ts") || file.endsWith(".js"));

        for (const file of files) {
            const loaded = require(path.join(folderPath, file));
            const command = (loaded.default || loaded) as BotCommand;

            if (!command?.data?.name || typeof command.execute !== "function") {
                continue;
            }

            client.commands.set(command.data.name, command);
            commandPayloads.push(command.data.toJSON());
            loadedCommands.push(command.data.name);
        }
    }

    if (client.application) {
        await client.application.commands.set(commandPayloads as any[]);
    }

    console.log(`Loaded Commands: ${loadedCommands.join(", ")}`);
}
