import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { ProductStore } from "@/runtime/storage/ProductStore";
import { ArtifactStore } from "@/discord/artifacts/ArtifactStore";

it("copies legacy cards and workflows once, preserving edits after migration", async () => {
    const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
    const source = path.join(AppPaths.storageRoot, "legacy.sqlite");
    const legacy = createClient({ url: pathToFileURL(source).toString() });
    await legacy.executeMultiple(`CREATE TABLE artifacts(message_id TEXT PRIMARY KEY,channel_id TEXT,guild_id TEXT,spec_json TEXT,created_timestamp INTEGER);
        INSERT INTO artifacts VALUES('legacy-card','channel','guild','{"title":"Before"}',1);
        CREATE TABLE workflows(id TEXT PRIMARY KEY,guild_id TEXT,name TEXT,description TEXT,steps_json TEXT,created_by TEXT,created_timestamp INTEGER,updated_timestamp INTEGER);
        INSERT INTO workflows VALUES('legacy-procedure','guild','weekly','Weekly report','[]','owner',1,1);`);
    legacy.close();
    await ProductStore.migrateLegacy(source);
    expect(await ArtifactStore.get("legacy-card")).toMatchObject({ channelId: "channel", viewSection: 0 });
    await ArtifactStore.updateSpec("legacy-card", '{"title":"After"}', null);
    await ProductStore.migrateLegacy(source);
    expect((await ArtifactStore.get("legacy-card"))?.specJson).toContain("After");
    expect((await ProductStore.getClient().execute("SELECT created_by FROM workflows WHERE id='legacy-procedure'")).rows[0].created_by).toBe("owner");
    expect(fs.existsSync(source)).toBe(true);
});

it("keeps failed expiry deletions for retry and retires already-deleted messages", async () => {
    await ArtifactStore.record({ messageId: "expired", channelId: "c", guildId: "g", expiresAt: 1, specJson: '{"sections":[{"body":"x"}]}' });
    const failed = await ArtifactStore.sweepExpired({ channels: { fetch: async () => { throw new Error("temporary outage"); } } });
    expect(failed.failed).toBe(1);
    expect(await ArtifactStore.get("expired")).not.toBeNull();
    await ArtifactStore.sweepExpired({ channels: { fetch: async () => ({ messages: { delete: async () => { throw { code: 10008 }; } } }) } });
    expect(await ArtifactStore.get("expired")).toBeNull();
});
