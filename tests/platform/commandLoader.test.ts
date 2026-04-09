import { Collection } from "discord.js";
import { describe, expect, it, vi } from "vitest";
import { loadCommands } from "@/platform/loaders/commandLoader";
import type { BotClient } from "@/shared/appTypes";

describe("commandLoader", () => {
    it("loads nested command entrypoints and ignores helper files", async () => {
        const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
        const setCommands = vi.fn().mockResolvedValue(undefined);
        const client = {
            commands: new Collection(),
            application: {
                commands: {
                    set: setCommands,
                },
            },
        } as unknown as BotClient;

        await loadCommands(client);

        expect(client.commands.has("ask")).toBe(true);
        expect(client.commands.has("find")).toBe(true);
        expect(client.commands.has("access")).toBe(true);
        expect(client.commands.has("debug")).toBe(true);
        expect(client.commands.has("nth")).toBe(true);
        expect(client.commands.has("getmessage")).toBe(false);
        expect(client.commands.has("adminHandlers")).toBe(false);
        expect(setCommands).toHaveBeenCalledTimes(1);
        consoleSpy.mockRestore();
    });
});
