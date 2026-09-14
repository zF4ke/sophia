import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";
import { createClient } from "@libsql/client";
import { expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { MigrationBackup } from "@/shared/storage/MigrationBackup";

it("preserves committed SQLite data before migration and never replaces an existing backup", async () => {
    const filename = path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`);
    const client = createClient({ url: pathToFileURL(filename).toString() });
    let backup: ReturnType<typeof createClient> | undefined;
    try {
        await client.execute("CREATE TABLE retained(value TEXT)");
        await client.execute("INSERT INTO retained VALUES('before')");
        const [first, second] = await Promise.all([MigrationBackup.database(client, filename), MigrationBackup.database(client, filename)]);
        expect(first).toBe(second);
        await client.execute("UPDATE retained SET value='after'");
        expect(await MigrationBackup.database(client, filename)).toBe(first);
        backup = createClient({ url: pathToFileURL(first!).toString() });
        expect((await backup.execute("SELECT value FROM retained")).rows[0].value).toBe("before");
    } finally { backup?.close(); client.close(); }
});

it("retains each distinct settings input and reuses an identical snapshot", () => {
    const filename = path.join(AppPaths.storageRoot, `${randomUUID()}.json`);
    fs.mkdirSync(path.dirname(filename), { recursive: true });
    fs.writeFileSync(filename, '{"grants":["owner"]}');
    const first = MigrationBackup.file(filename);
    expect(MigrationBackup.file(filename)).toBe(first);
    fs.writeFileSync(filename, '{"grants":["other"]}');
    expect(MigrationBackup.file(filename)).not.toBe(first);
    expect(fs.readFileSync(first, "utf8")).toBe('{"grants":["owner"]}');
});
