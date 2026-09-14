import path from "node:path";
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";
import { expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { KnowledgeStore } from "@/memory/KnowledgeStore";

it("imports legacy memory once without broadening unknown audiences or deleting the source", async () => {
    const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
    const filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`);
    const legacy = createClient({ url: pathToFileURL(filename).toString() });
    const store = new KnowledgeStore(path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`));
    try {
        await legacy.executeMultiple(`CREATE TABLE long_term_memories(id TEXT, guild_id TEXT, user_id TEXT,kind TEXT,key TEXT,value TEXT,created_timestamp INTEGER,updated_timestamp INTEGER);
            INSERT INTO long_term_memories VALUES('shared','g1',NULL,'guild','Guild fact','Shared fact',1,1);
            INSERT INTO long_term_memories VALUES('private','g1','u1','channel','Private fact','Unknown channel',1,1);`);
        await store.migrateLegacy(filename);
        await store.migrateLegacy(filename);
        const audience = { actorId: "u1", guildId: "g1", channelId: "c1" };
        expect(await store.search(audience)).toEqual([]);
        expect(await store.quarantine("g1")).toHaveLength(2);
        await expect(store.adoptQuarantined({ ...audience, guildId: "g2" }, "shared", 1)).rejects.toThrow("unavailable");
        await store.adoptQuarantined(audience, "shared", 1);
        expect(await store.search(audience)).toEqual([]);
        expect(await store.search({ ...audience, privateResponse: true })).toMatchObject([{ id: "shared", scope: "user", revision: 2 }]);
        await expect(store.adoptQuarantined(audience, "shared", 1)).rejects.toThrow("unavailable");
        expect(await store.search({ actorId: "u1", guildId: "g2", channelId: "c2" })).toEqual([]);
        expect((await legacy.execute("SELECT * FROM long_term_memories")).rows).toHaveLength(2);
    } finally { legacy.close(); await store.close(); }
});
