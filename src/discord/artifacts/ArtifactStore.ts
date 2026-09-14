import { ProductStore } from "@/runtime/storage/ProductStore";
import { KeyedLock } from "@/shared/KeyedLock";

export interface ArtifactRow {
    messageId: string;
    channelId: string;
    guildId: string | null;
    expiresAt: number | null;
    specJson: string | null;
    viewSection: number;
    gameStateJson: string | null;
    createdTimestamp: number;
    revision: number;
}

/**
 * Storage for artifact cards: the persisted spec (so cards can be edited and
 * their controls keep working across restarts), the current view state, and
 * TTL bookkeeping. This is the authoritative card state. Expiry failures retain
 * their records so a later sweep can retry deletion.
 */
export class ArtifactStore {
    public static async sources(messageId: string): Promise<string[]> {
        await this.ensureTable();
        return (await ProductStore.getClient().execute({ sql: "SELECT source FROM artifact_sources WHERE message_id=? ORDER BY source", args: [messageId] })).rows.map(row => String(row.source));
    }
    public static async addSources(messageId: string, sources: string[]): Promise<void> {
        await this.ensureTable();
        if (sources.length) await ProductStore.getClient().batch(sources.map(source => ({ sql: "INSERT OR IGNORE INTO artifact_sources VALUES(?,?)", args: [messageId, source] })), "write");
    }
    private static snapshot(messageId: string) {
        return { sql: `INSERT INTO artifact_revisions(message_id,revision,spec_json,game_state,created_at)
            SELECT message_id,COALESCE((SELECT MAX(revision) FROM artifact_revisions r WHERE r.message_id=artifacts.message_id),0)+1,spec_json,game_state,?
            FROM artifacts WHERE message_id=?`, args: [Date.now(), messageId] };
    }
    public static async revision(messageId: string, revision: number) {
        await this.ensureTable();
        const row = (await ProductStore.getClient().execute({ sql: "SELECT spec_json,game_state,created_at FROM artifact_revisions WHERE message_id=? AND revision=?", args: [messageId, revision] })).rows[0];
        return row ? { revision, spec: JSON.parse(String(row.spec_json)), gameState: row.game_state ? JSON.parse(String(row.game_state)) : null, createdAt: Number(row.created_at) } : null;
    }
    public static async owner(messageId: string): Promise<string | null> {
        await this.ensureTable();
        const row = (await ProductStore.getClient().execute({ sql: "SELECT owner_id FROM artifact_owners WHERE message_id=?", args: [messageId] })).rows[0];
        return row ? String(row.owner_id) : null;
    }
    public static readonly locks = new KeyedLock();
    public static async saveInteraction(messageId: string, specJson: string, gameState: Record<string, unknown>, section: number): Promise<void> {
        await this.ensureTable();
        await ProductStore.getClient().batch([{ sql: "UPDATE artifacts SET spec_json=?,game_state=?,view_section=? WHERE message_id=?", args: [specJson, JSON.stringify(gameState), section, messageId] }, this.snapshot(messageId)], "write");
    }
    public static async ensureTable(): Promise<void> {
        await ProductStore.initialize();
    }

    public static async record(entry: {
        messageId: string;
        channelId: string;
        guildId: string | null;
        expiresAt: number | null;
        specJson: string;
        ownerId?: string | null;
        sources?: string[];
    }): Promise<void> {
        await this.ensureTable();
        const client = ProductStore.getClient();
        const initialGameState = (() => {
            try {
                const parsed = JSON.parse(entry.specJson) as Record<string, unknown>;
                const state = (parsed as { gameState?: unknown }).gameState ?? (parsed as { game_state?: unknown }).game_state;
                return state && typeof state === "object" ? JSON.stringify(state) : null;
            } catch {
                return null;
            }
        })();
        await client.batch([{
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
        }, ...(entry.ownerId ? [{ sql: "INSERT INTO artifact_owners(message_id,owner_id) VALUES(?,?) ON CONFLICT(message_id) DO NOTHING", args: [entry.messageId, entry.ownerId] }] : []),
        ...(entry.sources ?? []).map(source => ({ sql: "INSERT OR IGNORE INTO artifact_sources VALUES(?,?)", args: [entry.messageId, source] })), this.snapshot(entry.messageId)], "write");
    }

    public static async get(messageId: string): Promise<ArtifactRow | null> {
        await this.ensureTable();
        const client = ProductStore.getClient();
        const row = (
            await client.execute({ sql: `SELECT message_id, channel_id, guild_id, expires_at, spec_json, view_section, game_state, created_timestamp,(SELECT COALESCE(MAX(revision),0) FROM artifact_revisions r WHERE r.message_id=artifacts.message_id) AS revision FROM artifacts WHERE message_id = :messageId`, args: { messageId } })
        ).rows[0] as Record<string, unknown> | undefined;
        if (!row) return null;
        return {
            messageId: String(row.message_id),
            revision: Number(row.revision),
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
        const client = ProductStore.getClient();
        await client.execute({
            sql: `UPDATE artifacts SET view_section = :section WHERE message_id = :messageId`,
            args: { section: Math.max(0, Math.floor(section)), messageId },
        });
    }

    public static async updateGameState(messageId: string, gameState: Record<string, unknown> | null): Promise<void> {
        await this.ensureTable();
        const client = ProductStore.getClient();
        await client.execute({
            sql: `UPDATE artifacts SET game_state = :gameState WHERE message_id = :messageId`,
            args: { gameState: gameState ? JSON.stringify(gameState) : null, messageId },
        });
    }

    public static async updateSpec(messageId: string, specJson: string, expiresAt: number | null, gameState?: Record<string, unknown>): Promise<void> {
        await this.ensureTable();
        const client = ProductStore.getClient();
        await client.batch([{
            sql: `UPDATE artifacts SET spec_json = :specJson, expires_at = :expiresAt, view_section = 0,
                game_state = CASE WHEN :replaceGame THEN :gameState ELSE game_state END WHERE message_id = :messageId`,
            args: { specJson, expiresAt, messageId, replaceGame: gameState === undefined ? 0 : 1, gameState: gameState === undefined ? null : JSON.stringify(gameState) },
        }, this.snapshot(messageId)], "write");
    }

    /**
     * Delete every expired card, best-effort per row. Returns how many cards
     * were actually removed from Discord.
     */
    public static async sweepExpired(client: {
        channels: { fetch(channelId: string): Promise<unknown> };
    }): Promise<{ removed: number; failed: number }> {
        await this.ensureTable();
        const store = ProductStore.getClient();
        const now = Date.now();
        const rows = (
            await store.execute({ sql: `SELECT message_id, channel_id FROM artifacts WHERE expires_at IS NOT NULL AND expires_at <= :now`, args: { now } })
        ).rows as Array<Record<string, unknown>>;

        let removed = 0;
        let failed = 0;
        for (const row of rows) {
            const messageId = String(row.message_id);
            const channelId = String(row.channel_id);
            let deleted = false;
            try {
                const channel = (await client.channels.fetch(channelId)) as { messages?: { delete(id: string): Promise<unknown> } } | null;
                if (channel?.messages) {
                    await channel.messages.delete(messageId);
                    removed += 1;
                    deleted = true;
                } else {
                    failed += 1;
                }
            } catch (error) {
                // Discord's Unknown Message is an already-completed deletion.
                if ((error as { code?: number }).code === 10008) deleted = true;
                else failed += 1;
            }
            if (!deleted) continue;
            try {
                await store.batch(["artifacts", "artifact_sources", "artifact_owners", "artifact_revisions"].map(table => ({ sql: `DELETE FROM ${table} WHERE message_id=?`, args: [messageId] })), "write");
            } catch {
                // Row lingers until the next sweep; harmless.
            }
        }
        return { removed, failed };
    }
}
