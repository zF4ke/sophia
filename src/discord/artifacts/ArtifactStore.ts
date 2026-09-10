import { OperationalStore } from "@/runtime/storage/OperationalStore";

export interface ArtifactRow {
    messageId: string;
    channelId: string;
    guildId: string | null;
    expiresAt: number | null;
    specJson: string | null;
    viewSection: number;
    gameStateJson: string | null;
    createdTimestamp: number;
}

/**
 * Storage for artifact cards: the persisted spec (so cards can be edited and
 * their controls keep working across restarts), the current view state, and
 * TTL bookkeeping. Rows are best-effort deletion hints, not a source of
 * truth: a missed sweep only means an expired card lingers until the next boot.
 */
export class ArtifactStore {
    public static async ensureTable(): Promise<void> {
        await OperationalStore.initialize();
        const client = OperationalStore.getClient();
        await client.executeMultiple(`
            CREATE TABLE IF NOT EXISTS artifacts (
                message_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                guild_id TEXT,
                expires_at INTEGER,
                spec_json TEXT,
                view_section INTEGER NOT NULL DEFAULT 0,
                game_state TEXT,
                created_timestamp INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_expiry ON artifacts(expires_at);
        `);
        // Migrations for tables created before these columns existed.
        for (const column of ["spec_json TEXT", "view_section INTEGER NOT NULL DEFAULT 0", "game_state TEXT"]) {
            try {
                await client.execute(`ALTER TABLE artifacts ADD COLUMN ${column}`);
            } catch {
                // Column already exists; nothing to migrate.
            }
        }
    }

    public static async record(entry: {
        messageId: string;
        channelId: string;
        guildId: string | null;
        expiresAt: number | null;
        specJson: string;
    }): Promise<void> {
        await this.ensureTable();
        const client = OperationalStore.getClient();
        const initialGameState = (() => {
            try {
                const parsed = JSON.parse(entry.specJson) as Record<string, unknown>;
                const state = (parsed as { gameState?: unknown }).gameState ?? (parsed as { game_state?: unknown }).game_state;
                return state && typeof state === "object" ? JSON.stringify(state) : null;
            } catch {
                return null;
            }
        })();
        await client.execute({
            sql: `INSERT OR REPLACE INTO artifacts (message_id, channel_id, guild_id, expires_at, spec_json, view_section, game_state, created_timestamp)
                  VALUES (:messageId, :channelId, :guildId, :expiresAt, :specJson, 0, :gameState, :ts)`,
            args: {
                messageId: entry.messageId,
                channelId: entry.channelId,
                guildId: entry.guildId,
                expiresAt: entry.expiresAt,
                specJson: entry.specJson,
                gameState: initialGameState,
                ts: Date.now(),
            },
        });
    }

    public static async get(messageId: string): Promise<ArtifactRow | null> {
        await this.ensureTable();
        const client = OperationalStore.getClient();
        const row = (
            await client.execute({ sql: `SELECT message_id, channel_id, guild_id, expires_at, spec_json, view_section, game_state, created_timestamp FROM artifacts WHERE message_id = :messageId`, args: { messageId } })
        ).rows[0] as Record<string, unknown> | undefined;
        if (!row) return null;
        return {
            messageId: String(row.message_id),
            channelId: String(row.channel_id),
            guildId: row.guild_id == null ? null : String(row.guild_id),
            expiresAt: row.expires_at == null ? null : Number(row.expires_at),
            specJson: row.spec_json == null ? null : String(row.spec_json),
            viewSection: row.view_section == null ? 0 : Number(row.view_section),
            gameStateJson: row.game_state == null ? null : String(row.game_state),
            createdTimestamp: Number(row.created_timestamp ?? 0),
        };
    }

    public static async updateViewState(messageId: string, section: number): Promise<void> {
        await this.ensureTable();
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `UPDATE artifacts SET view_section = :section WHERE message_id = :messageId`,
            args: { section: Math.max(0, Math.floor(section)), messageId },
        });
    }

    public static async updateGameState(messageId: string, gameState: Record<string, unknown> | null): Promise<void> {
        await this.ensureTable();
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `UPDATE artifacts SET game_state = :gameState WHERE message_id = :messageId`,
            args: { gameState: gameState ? JSON.stringify(gameState) : null, messageId },
        });
    }

    public static async updateSpec(messageId: string, specJson: string, expiresAt: number | null): Promise<void> {
        await this.ensureTable();
        const client = OperationalStore.getClient();
        await client.execute({
            sql: `UPDATE artifacts SET spec_json = :specJson, expires_at = :expiresAt, view_section = 0 WHERE message_id = :messageId`,
            args: { specJson, expiresAt, messageId },
        });
    }

    /**
     * Delete every expired card, best-effort per row. Returns how many cards
     * were actually removed from Discord.
     */
    public static async sweepExpired(client: {
        channels: { fetch(channelId: string): Promise<unknown> };
    }): Promise<{ removed: number; failed: number }> {
        await this.ensureTable();
        const store = OperationalStore.getClient();
        const now = Date.now();
        const rows = (
            await store.execute({ sql: `SELECT message_id, channel_id FROM artifacts WHERE expires_at IS NOT NULL AND expires_at <= :now`, args: { now } })
        ).rows as Array<Record<string, unknown>>;

        let removed = 0;
        let failed = 0;
        for (const row of rows) {
            const messageId = String(row.message_id);
            const channelId = String(row.channel_id);
            try {
                const channel = (await client.channels.fetch(channelId)) as { messages?: { delete(id: string): Promise<unknown> } } | null;
                if (channel?.messages) {
                    await channel.messages.delete(messageId);
                    removed += 1;
                }
            } catch {
                failed += 1;
            }
            try {
                await store.execute({ sql: `DELETE FROM artifacts WHERE message_id = :messageId`, args: { messageId } });
            } catch {
                // Row lingers until the next sweep; harmless.
            }
        }
        return { removed, failed };
    }
}
