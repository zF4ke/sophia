import fs from "node:fs";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import type { Client } from "@libsql/client";
import { KeyedLock } from "@/shared/KeyedLock";

/** Preserve the input before a schema/config rewrite; a failed backup blocks migration. */
export class MigrationBackup {
    private static readonly locks = new KeyedLock();
    static file(filename: string): string {
        const bytes = fs.readFileSync(filename);
        const hash = createHash("sha256").update(bytes).digest("hex").slice(0, 16);
        const directory = path.join(path.dirname(filename), "backups");
        fs.mkdirSync(directory, { recursive: true });
        const target = path.join(directory, `${path.basename(filename)}.pre-v5.${hash}`);
        try { fs.writeFileSync(target, bytes, { flag: "wx" }); }
        catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; }
        return target;
    }

    static async database(client: Client, filename: string, revision = "v5"): Promise<string | null> {
        if (!/^[a-z0-9-]+$/.test(revision)) throw new Error("Invalid migration revision.");
        if (!(await client.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")).rows.length) return null;
        const directory = path.join(path.dirname(filename), "backups");
        fs.mkdirSync(directory, { recursive: true });
        const sourceHash = createHash("sha256").update(path.resolve(filename)).digest("hex").slice(0, 12);
        const target = path.join(directory, `${sourceHash}.${revision}.sqlite`);
        return this.locks.run(target, async () => {
            if (fs.existsSync(target)) return target;
            // Keep SQLite paths short for Windows builds without long-path support.
            const temporary = path.join(directory, `${randomUUID().slice(0, 8)}.tmp`);
            try {
                // VACUUM INTO captures committed WAL content in one consistent database.
                await client.execute({ sql: "VACUUM INTO ?", args: [temporary] });
                fs.writeFileSync(`${target}.source.json`, JSON.stringify({ source: path.resolve(filename), revision, createdAt: Date.now() }), "utf8");
                fs.renameSync(temporary, target);
            } finally { if (fs.existsSync(temporary)) fs.unlinkSync(temporary); }
            return target;
        });
    }
}
