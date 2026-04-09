import { MemoryDatabase } from "@/memory/MemoryDatabase";

export class ResponseCounterRepository {
    public static nextGuildResponseOrdinal(guildId: string | null): number | null {
        if (!guildId) {
            return null;
        }

        const db = MemoryDatabase.get();
        const current = db
            .prepare(
                `SELECT response_ordinal FROM guild_response_counters WHERE guild_id = ?`
            )
            .get(guildId) as { response_ordinal?: number } | undefined;

        const nextOrdinal = (current?.response_ordinal ?? 0) + 1;
        db.prepare(
            `
                INSERT INTO guild_response_counters (guild_id, response_ordinal)
                VALUES (?, ?)
                ON CONFLICT(guild_id) DO UPDATE SET
                    response_ordinal = excluded.response_ordinal
            `
        ).run(guildId, nextOrdinal);

        return nextOrdinal;
    }
}
