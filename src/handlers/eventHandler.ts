import fs from "fs";
import path from "path";
import type { BotClient } from "@/types/app";

type EventModule = {
    name: string;
    once?: boolean;
    execute: (...args: any[]) => Promise<void> | void;
};

export function loadEvents(client: BotClient): void {
    const eventsRoot = path.join(process.cwd(), "src", "events");
    const loadedEvents: string[] = [];

    const folders = fs.readdirSync(eventsRoot, { withFileTypes: true });
    for (const folder of folders) {
        if (!folder.isDirectory()) {
            continue;
        }

        const folderPath = path.join(eventsRoot, folder.name);
        const files = fs
            .readdirSync(folderPath)
            .filter((file) => file.endsWith(".ts") || file.endsWith(".js"));

        for (const file of files) {
            const loaded = require(path.join(folderPath, file));
            const event = (loaded.default || loaded) as EventModule;

            if (!event?.name || typeof event.execute !== "function") {
                continue;
            }

            if (event.once) {
                client.once(event.name, (...args) => event.execute(...args, client));
            } else {
                client.on(event.name, (...args) => event.execute(...args, client));
            }

            loadedEvents.push(event.name);
        }
    }

    console.log(`Loaded events: ${loadedEvents.join(", ")}`);
}
