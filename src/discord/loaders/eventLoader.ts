import { AppPaths } from "@/app/AppPaths";
import { discoverModuleFiles } from "@/discord/loaders/moduleDiscovery";
import type { BotClient } from "@/shared/appTypes";
import { pathToFileURL } from "url";

type EventModule = {
    name: string;
    once?: boolean;
    execute: (...args: any[]) => Promise<void> | void;
};

export async function loadEvents(client: BotClient): Promise<void> {
    const loadedEvents: string[] = [];

    for (const filePath of discoverModuleFiles(AppPaths.eventModulesRoot, ".event")) {
        const loaded = await import(pathToFileURL(filePath).href);
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

    console.log(`Loaded events: ${loadedEvents.join(", ")}`);
}

