import { expect, it } from "vitest";
import { GatewayIntentBits, Partials } from "discord.js";
import { createClient } from "@/app/createClient";
it("subscribes to direct-message events as well as guild messages", () => {
    const client = createClient();
    expect(client.options.intents.has(GatewayIntentBits.DirectMessages)).toBe(true);
    expect(client.options.intents.has(GatewayIntentBits.GuildMessages)).toBe(true);
    expect(client.options.partials).toContain(Partials.Channel);
    client.destroy();
});
