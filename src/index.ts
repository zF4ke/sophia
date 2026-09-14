require("node:process").loadEnvFile();

import { bootstrapRuntime } from "@/app/bootstrapRuntime";
import { createClient } from "@/app/createClient";
import { registerProcessHandlers } from "@/app/registerProcessHandlers";
import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";
import { acquireInstanceGuard } from "@/app/InstanceGuard";
import { AppPaths } from "@/app/AppPaths";
import { DreamingService } from "@/memory/DreamingService";
import { Scheduler } from "@/runtime/scheduling/Scheduler";

const client = createClient();
let releaseInstance: (() => Promise<void>) | undefined;
registerProcessHandlers(async () => {
    DreamingService.stop();
    Scheduler.stop();
    await DiscordBackfillCrawler.stop();
    const drained = await ActiveRequestTracker.waitForIdle(30_000);
    if (!drained) console.warn("[shutdown] Timed out waiting for active replies.");
    client.destroy();
    await releaseInstance?.();
});
void acquireInstanceGuard(AppPaths.storageRoot).then(async release => {
    releaseInstance = release;
    await bootstrapRuntime(client);
}).catch(async (error) => {
    console.error("[bootstrap] Sophia failed to start.", error);
    client.destroy();
    await releaseInstance?.();
    process.exitCode = 1;
});

export default client;
