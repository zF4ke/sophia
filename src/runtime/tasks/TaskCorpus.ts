import { randomUUID } from "node:crypto";
import type { Client } from "@libsql/client";

export interface CorpusFilters { channelIds: string[]; authorId?: string; beforeTimestamp?: number; afterTimestamp?: number }
export interface CorpusPage { messages: Array<{ messageId: string; channelId: string; createdTimestamp: number; [key: string]: unknown }>; cursor: unknown; coverage: unknown }

/** Task-owned research material. Revisions make concurrent pagination explicit. */
export class TaskCorpus {
    constructor(private readonly client: Client, private readonly taskId: string) {}
    async create(filters: CorpusFilters) {
        const id = randomUUID();
        await this.client.execute({ sql: "INSERT INTO task_corpora(id,task_id,filters_json,revision,cursor_json,coverage_json) VALUES(?,?,?,0,'null','null')", args: [id, this.taskId, JSON.stringify(filters)] });
        return this.status(id);
    }
    async status(id: string) {
        const row = (await this.client.execute({ sql: "SELECT *, (SELECT COUNT(*) FROM corpus_messages WHERE corpus_id=task_corpora.id) AS count FROM task_corpora WHERE id=? AND task_id=?", args: [id, this.taskId] })).rows[0];
        if (!row) throw new Error("Corpus does not belong to this task.");
        return { id, revision: Number(row.revision), count: Number(row.count), filters: JSON.parse(String(row.filters_json)) as CorpusFilters,
            cursor: JSON.parse(String(row.cursor_json)), coverage: JSON.parse(String(row.coverage_json)) };
    }
    async append(id: string, revision: number, page: CorpusPage) {
        const state = await this.status(id);
        if (state.revision !== revision) throw new Error("Corpus revision changed. Read its current state before collecting again.");
        if (page.messages.some(message => !message.messageId || !state.filters.channelIds.includes(message.channelId))) throw new Error("Retrieved message is outside the corpus channel scope.");
        const commit = randomUUID();
        const result = await this.client.batch([
            { sql: `UPDATE task_corpora SET revision=revision+1,cursor_json=?,coverage_json=?,last_commit=? WHERE id=? AND task_id=? AND revision=?
                AND NOT EXISTS(SELECT 1 FROM json_each(?) page JOIN current_message_revisions current ON current.message_id=json_extract(page.value,'$.messageId') WHERE current.source_url<>COALESCE(json_extract(page.value,'$.jumpLink'),''))`, args: [JSON.stringify(page.cursor ?? null), JSON.stringify(page.coverage ?? null), commit, id, this.taskId, revision, JSON.stringify(page.messages)] },
            ...page.messages.map(message => ({ sql: "INSERT OR IGNORE INTO corpus_messages(corpus_id,message_id,timestamp,message_json) SELECT ?,?,?,? WHERE EXISTS(SELECT 1 FROM task_corpora WHERE id=? AND last_commit=?) AND NOT EXISTS(SELECT 1 FROM deleted_corpus_messages WHERE message_id=?)", args: [id, message.messageId, message.createdTimestamp, JSON.stringify(message), id, commit, message.messageId] })),
        ], "write");
        if (result[0].rowsAffected !== 1) throw new Error("Corpus revision changed while collecting. The competing page was not saved.");
        return this.status(id);
    }
    async read(id: string, revision: number, offset: number, limit: number) {
        const state = await this.status(id);
        if (state.revision !== revision) throw new Error("Corpus revision changed. Restart reading at the current revision.");
        const rows = (await this.client.execute({ sql: "SELECT message_json FROM corpus_messages WHERE corpus_id=? ORDER BY timestamp DESC,message_id DESC LIMIT ? OFFSET ?", args: [id, limit, offset] })).rows;
        if ((await this.status(id)).revision !== revision) throw new Error("Corpus changed during the read. Retry at its current revision.");
        return { ...state, offset, nextOffset: offset + rows.length < state.count ? offset + rows.length : null, messages: rows.map(row => JSON.parse(String(row.message_json))) };
    }
}
