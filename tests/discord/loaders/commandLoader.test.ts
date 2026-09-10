import { describe, expect, it, vi } from "vitest";
import { loadCommands } from "@/discord/loaders/commandLoader";
import type { BotClient } from "@/shared/appTypes";

describe("commandLoader", () => {
    it(
        "loads nested command entrypoints and ignores helper files",
        async () => {
            const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
            const setCommands = vi.fn().mockResolvedValue(undefined);
            const fetchCommands = vi.fn().mockResolvedValue(new Map([
                ["entry", {
                    type: 4,
                    name: "launch",
                    description: "Launch Sophia",
                    handler: 2,
                    integrationTypes: [0, 1],
                    contexts: [0, 1, 2],
                }],
            ]));
            const client = {
                commands: new Map(),
                application: {
                    commands: {
                        fetch: fetchCommands,
                        set: setCommands,
                    },
                },
            } as unknown as BotClient;

            await loadCommands(client);

            expect(client.commands.has("talk")).toBe(true);
            expect(client.commands.has("index")).toBe(true);
            expect(setCommands).toHaveBeenCalledWith(
                expect.arrayContaining([
                    expect.objectContaining({ name: "talk" }),
                    expect.objectContaining({ name: "index" }),
                    expect.objectContaining({
                        type: 4,
                        name: "launch",
                        handler: 2,
                        integration_types: [0, 1],
                        contexts: [0, 1, 2],
                    }),
                ])
            );
            consoleSpy.mockRestore();
        },
        15000
    );
});
