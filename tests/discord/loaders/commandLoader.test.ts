import { describe, expect, it, vi } from "vitest";
import { loadCommands } from "@/discord/loaders/commandLoader";
import type { BotClient } from "@/shared/appTypes";

describe("commandLoader", () => {
    it(
        "loads nested command entrypoints and ignores helper files",
        async () => {
            const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
            const setCommands = vi.fn().mockResolvedValue(undefined);
            const client = {
                commands: new Map(),
                application: {
                    commands: {
                        set: setCommands,
                    },
                },
            } as unknown as BotClient;

            await loadCommands(client);

            expect(client.commands.has("talk")).toBe(true);
            expect(client.commands.has("find")).toBe(true);
            expect(client.commands.has("index")).toBe(true);
            expect(setCommands).toHaveBeenCalledWith(
                expect.arrayContaining([
                    expect.objectContaining({ name: "talk" }),
                    expect.objectContaining({ name: "find" }),
                    expect.objectContaining({ name: "index" }),
                ])
            );
            consoleSpy.mockRestore();
        },
        15000
    );
});
