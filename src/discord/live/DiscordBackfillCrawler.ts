import type { Client, TextChannel, ThreadChannel } from "discord.js";
import { ChannelType } from "discord.js";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { SettingsService } from "@/app/SettingsService";
import { isGuildAllowed } from "@/security/guildAllowlist";
import {
    fetchAndIngestBatch,
    resumeBeforeId,
    type IndexableChannel,
} from "@/discord/live/DiscordBackfillService";

type CrawlQueueState = "queued" | "running" | "paused" | "done";

export interface CrawlQueueRow {
    channelId: string;
    guildId: string | null;
    priority: number;
    reason: string | null;
    enqueuedAt: number;
    state: CrawlQueueState;
    messagesIngested: number;
    lastError: string | null;
    lastActivityAt: number | null;
}

export interface CrawlChannelStatus {
    channelId: string;
    running: boolean;
    state: CrawlQueueState | "idle";
    queuedAhead: number;
    priority: number;
    messagesIngested: number;
    oldestFetchedMessageId: string | null;
    exhausted: boolean;
    lastError: string | null;
    enqueuedAt: number | null;
    lastActivityAt: number | null;
}

type CrawlerLogger = (message: string) => void;

const BATCH_DELAY_MS = 250;
const POLL_IDLE_MS = 5_000;

function rowToQueueRow(row: Record<string, unknown>): CrawlQueueRow {
    return {
        channelId: String(row.channel_id),
        guildId: row.guild_id == null ? null : String(row.guild_id),
        priority: Number(row.priority ?? 0),
        reason: row.reason == null ? null : String(row.reason),
        enqueuedAt: Number(row.enqueued_at ?? 0),
        state: String(row.state ?? "queued") as CrawlQueueState,
        messagesIngested: Number(row.messages_ingested ?? 0),
        lastError: row.last_error == null ? null : String(row.last_error),
        lastActivityAt: row.last_activity_at == null ? null : Number(row.last_activity_at),
    };
}

export class DiscordBackfillCrawler {
    private static client: Client | null = null;
    private static running = false;
    private static stopping = false;
    private static workerPromise: Promise<void> | null = null;
    private static paused = false;
    private static currentChannelId: string | null = null;
    private static logger: CrawlerLogger = (msg) => console.log(`[crawler] ${msg}`);
    private static readonly structuredLog: Array<{ at: number; line: string }> = [];
    private static readonly MAX_LOG_ENTRIES = 200;
    private static readonly stopRequested: Set<string> = new Set();

    public static setLogger(logger: CrawlerLogger): void {
        this.logger = logger;
    }

    public static getRecentLog(): Array<{ at: number; line: string }> {
        return [...this.structuredLog];
    }

    private static log(line: string): void {
        const at = Date.now();
        this.structuredLog.push({ at, line });
        while (this.structuredLog.length > this.MAX_LOG_ENTRIES) {
            this.structuredLog.shift();
        }
        try {
            this.logger(line);
        } catch {
            /* never throw from logger */
        }
    }

    public static async start(client: Client): Promise<void> {
        if (this.running) return;
        this.client = client;
        this.running = true;
        this.stopping = false;
        this.workerPromise = this.workerLoop().catch((err) => {
            this.log(`worker_crashed error=${err instanceof Error ? err.message : String(err)}`);
            this.running = false;
        });
        this.log("started");
        // Startup sweep: bounded recency pass per channel (closes the
        // "bot was offline" gap). NEVER walks full history here — channels
        // with hundreds of thousands of messages would crawl for hours.
        // Deep backfill is agent-gated via index_channel mode="deep".
        void this.sweepAllGuildChannels();
    }

    /**
     * Bounded recency sweep across every guild text channel.
     * Per channel: skip if already queued/running, skip if the newest
     * indexed message is < 1h old, otherwise fetch newest→older until we
     * hit known messages or the configured cap (default 1000 msgs).
     * Serialized so we never race the worker on the same channel.
     */
    public static async sweepAllGuildChannels(): Promise<{ swept: number; skipped: number; ingested: number }> {
        if (!this.client) return { swept: 0, skipped: 0, ingested: 0 };
        const settings = SettingsService.load();
        if (!settings.runtime.startupSweep) {
            this.log("startup_sweep disabled by settings");
            return { swept: 0, skipped: 0, ingested: 0 };
        }
        const cap = Math.max(100, Math.min(5000, settings.runtime.startupSweepMaxMessages || 1000));
        const queue = await this.listQueue();
        const busy = new Set(queue.map((q) => q.channelId));

        let swept = 0;
        let skipped = 0;
        let ingested = 0;

        for (const guild of this.client.guilds.cache.values()) {
            if (!isGuildAllowed(guild.id)) continue;
            const channels = await guild.channels.fetch().catch(() => null);
            if (!channels) continue;
            for (const channel of channels.values()) {
                if (!channel) continue;
                const type = (channel as { type?: number }).type;
                if (type !== ChannelType.GuildText && type !== ChannelType.GuildAnnouncement) continue;
                if (busy.has(channel.id)) {
                    skipped += 1;
                    continue;
                }

                const indexStates = await DiscordMemoryService.getIndexStateAsync(channel.id);
                const lastIndexed = indexStates[0]?.lastIndexedTimestamp ?? null;
                if (lastIndexed != null && Date.now() - lastIndexed < 3_600_000) {
                    skipped += 1;
                    continue;
                }

                const indexable = channel as unknown as IndexableChannel;
                try {
                    const result = await this.refreshChannel(indexable, cap);
                    swept += 1;
                    ingested += result.ingested;
                    this.log(`startup_sweep channel=${channel.id} ingested=${result.ingested} hitKnown=${result.hitKnown ? "yes" : "no"} cap=${cap}`);
                } catch (error) {
                    const message = error instanceof Error ? error.message : String(error);
                    this.log(`startup_sweep_error channel=${channel.id} error=${message}`);
                }
            }
        }
        this.log(`startup_sweep done swept=${swept} skipped=${skipped} ingested=${ingested}`);
        return { swept, skipped, ingested };
    }

    public static async stop(): Promise<void> {
        this.stopping = true;
        this.running = false;
        if (this.workerPromise) {
            await this.workerPromise.catch(() => {});
        }
        this.workerPromise = null;
        this.log("stopped");
    }

    public static pause(): void {
        this.paused = true;
        this.log("paused");
    }

    public static resume(): void {
        this.paused = false;
        this.log("resumed");
    }

    public static isPaused(): boolean {
        return this.paused;
    }

    public static isRunning(): boolean {
        return this.running;
    }

    public static async enqueue(
        channelId: string,
        options: { reason?: string; priority?: number; guildId?: string | null } = {}
    ): Promise<void> {
        if (options.guildId && !isGuildAllowed(options.guildId)) return;
        await OperationalStore.initialize();
        const client = OperationalStore.getClient();
        const existing = (
            await client.execute({
                sql: `SELECT state FROM crawl_queue WHERE channel_id = :channelId`,
                args: { channelId },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        const existingState = existing ? String(existing.state) : null;
        // If already queued/running we just bump priority; otherwise (re)insert as queued.
        const now = Date.now();
        const priority = options.priority ?? 0;

        if (existingState === "queued" || existingState === "running") {
            await client.execute({
                sql: `
                    UPDATE crawl_queue
                    SET priority = MAX(priority, :priority),
                        reason = COALESCE(:reason, reason),
                        last_activity_at = :now
                    WHERE channel_id = :channelId
                `,
                args: {
                    channelId,
                    priority,
                    reason: options.reason ?? null,
                    now,
                },
            });
        } else {
            await client.execute({
                sql: `
                    INSERT INTO crawl_queue (
                        channel_id, guild_id, priority, reason, enqueued_at,
                        state, messages_ingested, last_error, last_activity_at
                    ) VALUES (
                        :channelId, :guildId, :priority, :reason, :now,
                        'queued', 0, NULL, :now
                    )
                    ON CONFLICT(channel_id) DO UPDATE SET
                        guild_id = COALESCE(excluded.guild_id, crawl_queue.guild_id),
                        priority = MAX(crawl_queue.priority, excluded.priority),
                        reason = excluded.reason,
                        enqueued_at = excluded.enqueued_at,
                        state = 'queued',
                        messages_ingested = 0,
                        last_error = NULL,
                        last_activity_at = excluded.last_activity_at
                `,
                args: {
                    channelId,
                    guildId: options.guildId ?? null,
                    priority,
                    reason: options.reason ?? null,
                    now,
                },
            });
        }

        this.log(`enqueued channel=${channelId} reason=${options.reason ?? "none"} priority=${priority}`);
    }

    public static async stopChannel(channelId: string): Promise<void> {
        await OperationalStore.initialize();
        const client = OperationalStore.getClient();
        // Signal the worker to break mid-batch if this is the active job.
        this.stopRequested.add(channelId);
        await client.execute({
            sql: `
                UPDATE crawl_queue
                SET state = 'done', last_activity_at = :now
                WHERE channel_id = :channelId
            `,
            args: { channelId, now: Date.now() },
        });
        this.log(`stopped_channel channel=${channelId}`);
    }

    public static async listQueue(): Promise<CrawlQueueRow[]> {
        await OperationalStore.initialize();
        const client = OperationalStore.getClient();
        const rows = (
            await client.execute(
                `SELECT * FROM crawl_queue
                 WHERE state IN ('queued', 'running', 'paused')
                 ORDER BY priority DESC, enqueued_at ASC`
            )
        ).rows as Array<Record<string, unknown>>;
        return rows.map(rowToQueueRow);
    }

    public static async getStatus(channelId: string): Promise<CrawlChannelStatus> {
        await OperationalStore.initialize();
        const client = OperationalStore.getClient();
        const row = (
            await client.execute({
                sql: `SELECT * FROM crawl_queue WHERE channel_id = :channelId`,
                args: { channelId },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        const crawlStates = await DiscordMemoryService.getChannelCrawlStateAsync(channelId);
        const crawlState = crawlStates[0];

        if (!row) {
            return {
                channelId,
                running: false,
                state: "idle",
                queuedAhead: 0,
                priority: 0,
                messagesIngested: 0,
                oldestFetchedMessageId: crawlState?.oldestFetchedMessageId ?? null,
                exhausted: Boolean(crawlState?.exhausted),
                lastError: null,
                enqueuedAt: null,
                lastActivityAt: crawlState?.lastCrawledTimestamp ?? null,
            };
        }

        const queueRow = rowToQueueRow(row);
        const queuedAheadRow = (
            await client.execute({
                sql: `
                    SELECT COUNT(*) AS cnt FROM crawl_queue
                    WHERE state = 'queued'
                      AND (priority > :priority
                           OR (priority = :priority AND enqueued_at < :enqueuedAt))
                `,
                args: { priority: queueRow.priority, enqueuedAt: queueRow.enqueuedAt },
            })
        ).rows[0] as Record<string, unknown> | undefined;

        return {
            channelId,
            running: this.currentChannelId === channelId,
            state: queueRow.state,
            queuedAhead: Number(queuedAheadRow?.cnt ?? 0),
            priority: queueRow.priority,
            messagesIngested: queueRow.messagesIngested,
            oldestFetchedMessageId: crawlState?.oldestFetchedMessageId ?? null,
            exhausted: Boolean(crawlState?.exhausted),
            lastError: queueRow.lastError,
            enqueuedAt: queueRow.enqueuedAt,
            lastActivityAt: queueRow.lastActivityAt,
        };
    }

    // ── Worker ───────────────────────────────────────────────────────

    private static async pickNext(): Promise<CrawlQueueRow | null> {
        const client = OperationalStore.getClient();
        const row = (
            await client.execute(
                `SELECT * FROM crawl_queue
                 WHERE state = 'queued'
                 ORDER BY priority DESC, enqueued_at ASC
                 LIMIT 1`
            )
        ).rows[0] as Record<string, unknown> | undefined;
        return row ? rowToQueueRow(row) : null;
    }

    private static async markRunning(channelId: string): Promise<void> {
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `UPDATE crawl_queue SET state = 'running', last_activity_at = :now WHERE channel_id = :channelId`,
            args: { channelId, now: Date.now() },
        });
    }

    private static async markDone(channelId: string, ingested: number): Promise<void> {
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `
                UPDATE crawl_queue
                SET state = 'done',
                    messages_ingested = messages_ingested + :ingested,
                    last_activity_at = :now
                WHERE channel_id = :channelId
            `,
            args: { channelId, ingested, now: Date.now() },
        });
    }

    private static async markError(channelId: string, message: string): Promise<void> {
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `
                UPDATE crawl_queue
                SET state = 'queued',
                    last_error = :error,
                    last_activity_at = :now
                WHERE channel_id = :channelId
            `,
            args: { channelId, error: message.slice(0, 500), now: Date.now() },
        });
    }

    private static async incrementIngested(channelId: string, ingested: number): Promise<void> {
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `
                UPDATE crawl_queue
                SET messages_ingested = messages_ingested + :ingested,
                    last_activity_at = :now
                WHERE channel_id = :channelId
            `,
            args: { channelId, ingested, now: Date.now() },
        });
    }

    private static async resolveChannel(
        channelId: string
    ): Promise<IndexableChannel | null> {
        if (!this.client) return null;
        const cached = this.client.channels.cache.get(channelId);
        const channel = cached ?? (await this.client.channels.fetch(channelId).catch(() => null));
        if (!channel) return null;
        const type = (channel as { type?: number }).type;
        if (
            type === ChannelType.GuildText ||
            type === ChannelType.PublicThread ||
            type === ChannelType.PrivateThread ||
            type === ChannelType.GuildAnnouncement
        ) {
            return channel as TextChannel | ThreadChannel;
        }
        return null;
    }

    private static async crawlOneChannel(job: CrawlQueueRow): Promise<void> {
        // Legacy rows from before the guild allowlist existed may still sit in
        // the queue — drop them lazily instead of crawling foreign guilds.
        if (job.guildId && !isGuildAllowed(job.guildId)) {
            this.log(`skipped_disallowed_guild channel=${job.channelId} guild=${job.guildId}`);
            await this.markDone(job.channelId, 0);
            return;
        }
        const channel = await this.resolveChannel(job.channelId);
        if (!channel) {
            this.log(`channel_unresolvable channel=${job.channelId}`);
            await this.markDone(job.channelId, 0);
            return;
        }

        let before = await resumeBeforeId(job.channelId);
        let totalIngestedInJob = 0;
        this.currentChannelId = job.channelId;

        while (this.running && !this.stopping) {
            if (this.stopRequested.has(job.channelId)) {
                this.log(`channel_stopped_midbatch channel=${job.channelId} ingested_this_job=${totalIngestedInJob}`);
                this.stopRequested.delete(job.channelId);
                this.currentChannelId = null;
                return;
            }
            if (this.paused) {
                await new Promise((r) => setTimeout(r, POLL_IDLE_MS));
                continue;
            }

            try {
                const result = await fetchAndIngestBatch(channel, before, 100);
                totalIngestedInJob += result.ingested;

                if (result.ingested) {
                    await this.incrementIngested(job.channelId, result.ingested);
                }

                this.log(
                    `channel=${job.channelId} ingested=${result.ingested} total=${totalIngestedInJob} oldest=${result.nextBeforeId ?? "-"} exhausted=${result.reachedEnd && result.ingested === 0 ? "yes" : "no"}`
                );

                if (result.reachedEnd && result.ingested === 0) {
                    break;
                }

                before = result.nextBeforeId;
                await new Promise((r) => setTimeout(r, BATCH_DELAY_MS));
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                this.log(`channel_error channel=${job.channelId} error=${message}`);
                await this.markError(job.channelId, message);
                this.currentChannelId = null;
                return;
            }
        }

        this.currentChannelId = null;
        await this.markDone(job.channelId, 0);
        this.log(`job_done channel=${job.channelId} ingested_this_job=${totalIngestedInJob}`);
    }

    private static async workerLoop(): Promise<void> {
        while (this.running && !this.stopping) {
            if (this.paused) {
                await new Promise((r) => setTimeout(r, POLL_IDLE_MS));
                continue;
            }

            try {
                await OperationalStore.initialize();
                const next = await this.pickNext();
                if (!next) {
                    await new Promise((r) => setTimeout(r, POLL_IDLE_MS));
                    continue;
                }

                await this.markRunning(next.channelId);
                await this.crawlOneChannel(next);
            } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                this.log(`worker_loop_error error=${message}`);
                await new Promise((r) => setTimeout(r, POLL_IDLE_MS));
            }
        }
    }

    /**
     * Freshness sweep for one channel: walk newest→older closing the
     * "offline gap" (messages sent while the bot was down) until we hit
     * messages the index already has, or `maxMessages` is reached.
     * Returns the number of messages actually ingested.
     */
    public static async refreshChannel(
        channel: IndexableChannel,
        maxMessages = 500
    ): Promise<{ ingested: number; hitKnown: boolean; lastIndexedTimestamp: number | null }> {
        const states = await DiscordMemoryService.getChannelCrawlStateAsync(channel.id);
        // lastIndexedTimestamp lives on index_state, not crawl_state.
        const indexStates = await DiscordMemoryService.getIndexStateAsync(channel.id);
        const lastIndexedTimestamp = indexStates[0]?.lastIndexedTimestamp ?? null;

        let ingested = 0;
        let before: string | null | undefined = undefined; // newest first
        let hitKnown = false;

        while (ingested < maxMessages) {
            const batchSize = Math.min(100, maxMessages - ingested);
            const result = await fetchAndIngestBatch(channel, before ?? null, batchSize);
            ingested += result.ingested;
            if (result.reachedEnd && result.ingested === 0) break;
            if (result.ingested === 0) break;
            if (
                lastIndexedTimestamp != null &&
                result.oldestTimestampInBatch != null &&
                result.oldestTimestampInBatch <= lastIndexedTimestamp
            ) {
                hitKnown = true;
                break;
            }
            before = result.nextBeforeId;
            await new Promise((r) => setTimeout(r, BATCH_DELAY_MS));
        }
        return { ingested, hitKnown, lastIndexedTimestamp };
    }
}
