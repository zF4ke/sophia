import path from "path";
import { beforeEach, describe, expect, it } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { OperationalStore } from "@/runtime/storage/OperationalStore";

describe("DiscordBackfillCrawler", () => {
    beforeEach(async () => {
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        await OperationalStore.reset();
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `crawler-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
    });

    it("retires legacy startup crawls without cancelling demand-driven jobs", async () => {
        const client = OperationalStore.getClient();
        const now = Date.now();
        await client.batch([
            {
                sql: `INSERT INTO crawl_queue
                    (channel_id, guild_id, priority, reason, enqueued_at, state, messages_ingested, last_activity_at)
                    VALUES (:channelId, 'guild', 2, 'startup_refresh', :now, 'running', 12000, :now)`,
                args: { channelId: "legacy", now },
            },
            {
                sql: `INSERT INTO crawl_queue
                    (channel_id, guild_id, priority, reason, enqueued_at, state, messages_ingested, last_activity_at)
                    VALUES (:channelId, 'guild', 10, 'agent_deep', :now, 'queued', 0, :now)`,
                args: { channelId: "agent", now },
            },
        ]);

        await (DiscordBackfillCrawler as any).retireLegacyStartupJobs();

        const rows = (await client.execute(
            "SELECT channel_id, state FROM crawl_queue ORDER BY channel_id",
        )).rows;
        expect(rows).toEqual([
            expect.objectContaining({ channel_id: "agent", state: "queued" }),
            expect.objectContaining({ channel_id: "legacy", state: "done" }),
        ]);
    });

    it("recovers interrupted demand-driven jobs and pauses failed jobs", async () => {
        const client = OperationalStore.getClient();
        const now = Date.now();
        await client.execute({
            sql: `INSERT INTO crawl_queue
                (channel_id, guild_id, priority, reason, enqueued_at, state, messages_ingested, last_activity_at)
                VALUES ('agent', 'guild', 10, 'agent_deep', :now, 'running', 100, :now)`,
            args: { now },
        });

        await (DiscordBackfillCrawler as any).recoverInterruptedJobs();
        let row = (await client.execute(
            "SELECT state FROM crawl_queue WHERE channel_id = 'agent'",
        )).rows[0];
        expect(row.state).toBe("queued");

        await (DiscordBackfillCrawler as any).markError("agent", "Missing Access");
        row = (await client.execute(
            "SELECT state, last_error FROM crawl_queue WHERE channel_id = 'agent'",
        )).rows[0];
        expect(row).toMatchObject({ state: "paused", last_error: "Missing Access" });
    });

    it("returns an interrupted active job to the queue", async () => {
        const client = OperationalStore.getClient();
        const now = Date.now();
        await client.execute({
            sql: `INSERT INTO crawl_queue
                (channel_id, guild_id, priority, reason, enqueued_at, state, messages_ingested, last_activity_at)
                VALUES ('agent', 'guild', 10, 'agent_deep', :now, 'running', 100, :now)`,
            args: { now },
        });

        await (DiscordBackfillCrawler as any).markQueued("agent");

        const row = (await client.execute(
            "SELECT state, messages_ingested FROM crawl_queue WHERE channel_id = 'agent'",
        )).rows[0];
        expect(row).toMatchObject({ state: "queued", messages_ingested: 100 });
    });
});
