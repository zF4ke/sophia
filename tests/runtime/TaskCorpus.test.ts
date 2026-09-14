import path from "node:path";
import { randomUUID } from "node:crypto";
import { expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { TaskStore } from "@/runtime/tasks/TaskStore";

it("owns corpora by task, deduplicates overlapping pages and rejects stale revisions", async () => {
    const store = new TaskStore(path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`));
    try {
        const owner = { actorId: "owner", guildId: "guild", channelId: "channel", conversationId: "channel", objective: "Research" };
        const id = await store.create(owner), other = await store.create(owner);
        const corpus = await store.corpus(id, "owner", "channel", "guild");
        const state = await corpus.create({ channelIds: ["source"] });
        await expect(store.corpus(id, "stranger", "channel", "guild")).rejects.toThrow("owned");
        await expect((await store.corpus(other, "owner", "channel", "guild")).status(state.id)).rejects.toThrow("belong");
        const message = (messageId: string, createdTimestamp: number) => ({ messageId, channelId: "source", createdTimestamp, content: messageId });
        await corpus.append(state.id, 0, { messages: [message("a", 3), message("b", 2)], cursor: { history: { source: "b" } }, coverage: { partial: true } });
        const next = await corpus.append(state.id, 1, { messages: [message("b", 2), message("c", 1)], cursor: { history: { source: "c" } }, coverage: { partial: true } });
        expect(next.count).toBe(3);
        await expect(corpus.append(state.id, 1, { messages: [message("d", 0)], cursor: null, coverage: null })).rejects.toThrow("revision");
        await expect(corpus.read(state.id, 1, 0, 2)).rejects.toThrow("revision");
        const first = await corpus.read(state.id, 2, 0, 2);
        expect(first.messages.map(m => m.messageId)).toEqual(["a", "b"]);
        expect(first.nextOffset).toBe(2);
        expect((await corpus.read(state.id, 2, 2, 2)).messages.map(m => m.messageId)).toEqual(["c"]);
        const race = await Promise.allSettled([0, 1].map(index => corpus.append(state.id, 2, { messages: [message(`race${index}`, 0)], cursor: null, coverage: null })));
        expect(race.filter(result => result.status === "fulfilled")).toHaveLength(1);
        expect((await corpus.status(state.id)).count).toBe(4);
        await store.invalidateCorpusMessage("a");
        const invalidated = await corpus.status(state.id);
        expect(invalidated.count).toBe(3);
        expect(invalidated.revision).toBe(4);
        await corpus.append(state.id, 4, { messages: [message("a", 3)], cursor: null, coverage: null });
        expect((await corpus.status(state.id)).count).toBe(3);
        const file = { path: "evidence.json", data: Buffer.from("evidence").toString("base64"), sourceMessageIds: ["b"], sourceChannelIds: ["source"] };
        await store.replaceFiles(id, "owner", "channel", "guild", [file]);
        expect(await store.files(id, "owner", "channel", "guild")).toEqual([file]);
        await store.invalidateCorpusMessage("b");
        expect(await store.files(id, "owner", "channel", "guild")).toEqual([]);
        await expect(store.replaceFiles(id, "owner", "channel", "guild", [{ ...file, path: "derived.txt" }])).rejects.toThrow("source was deleted");
        expect(await store.files(id, "owner", "channel", "guild")).toEqual([]);
    } finally { await store.close(); }
});
