require("node:process").loadEnvFile();

import { bootstrapRuntime } from "@/app/bootstrapRuntime";
import { createClient } from "@/app/createClient";
import { registerProcessHandlers } from "@/app/registerProcessHandlers";
import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";

const client = createClient();
registerProcessHandlers(async () => {
    await DiscordBackfillCrawler.stop();
    const drained = await ActiveRequestTracker.waitForIdle(30_000);
    if (!drained) console.warn("[shutdown] Timed out waiting for active replies.");
    client.destroy();
    // Note: any `opencode serve` the bot spawned is deliberately left running.
    // Never kill opencode processes on shutdown — other sessions may rely on them.
});
void bootstrapRuntime(client).catch((error) => {
    console.error("[bootstrap] Sophia failed to start.", error);
    process.exitCode = 1;
});

export default client;
