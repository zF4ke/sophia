import { describe, expect, it, vi } from "vitest";
import { loadEvents } from "@/discord/loaders/eventLoader";
import type { BotClient } from "@/shared/appTypes";

describe("eventLoader", () => {
    it(
        "loads nested event entrypoints and registers them on the client",
        async () => {
            const consoleSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
            const client = {
                on: vi.fn(),
                once: vi.fn(),
            } as unknown as BotClient;

            await loadEvents(client);

            expect(client.once).toHaveBeenCalledWith("clientReady", expect.any(Function));
            expect(client.on).toHaveBeenCalledWith("interactionCreate", expect.any(Function));
            expect(client.on).toHaveBeenCalledWith("messageCreate", expect.any(Function));
            consoleSpy.mockRestore();
        },
        15000
    );
});
