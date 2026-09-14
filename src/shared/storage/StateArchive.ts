import fs from "node:fs";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { pathToFileURL } from "node:url";
import { z } from "zod";
import { acquireInstanceGuard } from "@/app/InstanceGuard";

const names = ["settings.json", "tasks.sqlite", "knowledge.sqlite", "products.sqlite", "security/admins.json", "security/moderators.json", "security/commands_config.json"] as const;
const manifestSchema = z.object({ version: z.literal(5), createdAt: z.string(), files: z.array(z.object({ path: z.enum(names), sha256: z.string().regex(/^[a-f0-9]{64}$/) }).strict()).min(1) }).strict();
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
function containedFile(root: string, name: string): string {
    const candidate = path.join(root, name);
    const relative = path.relative(fs.realpathSync(root), fs.realpathSync(candidate));
    if (relative.startsWith("..") || path.isAbsolute(relative) || fs.lstatSync(candidate).isSymbolicLink()) throw new Error("Archive file escaped its directory.");
    return candidate;
}

/** Explicit offline transfer of durable state. Never includes credentials or logs. */
export class StateArchive {
    static async create(storageRoot: string, destination: string): Promise<void> {
        const release = await acquireInstanceGuard(storageRoot);
        try {
            const settingsPath = path.join(storageRoot, "settings.json");
            if (!fs.existsSync(settingsPath) || JSON.parse(fs.readFileSync(settingsPath, "utf8")).schemaVersion !== 5) throw new Error("Complete v5 startup migration before archiving durable state.");
            if (fs.existsSync(destination)) throw new Error("Archive destination must not already exist.");
            fs.mkdirSync(destination);
            const files: Array<{ path: typeof names[number]; sha256: string }> = [];
            for (const name of names) {
                if (!fs.existsSync(path.join(storageRoot, name))) continue;
                const source = containedFile(storageRoot, name);
                const target = path.join(destination, name);
                fs.mkdirSync(path.dirname(target), { recursive: true });
                if (name.endsWith(".sqlite")) {
                    const { createClient } = require("@libsql/client") as typeof import("@libsql/client");
                    const client = createClient({ url: pathToFileURL(source).toString() });
                    try { await client.execute({ sql: "VACUUM INTO ?", args: [path.resolve(target)] }); }
                    finally { client.close(); }
                } else fs.copyFileSync(source, target, fs.constants.COPYFILE_EXCL);
                files.push({ path: name, sha256: hash(fs.readFileSync(target)) });
            }
            const manifest = manifestSchema.parse({ version: 5, createdAt: new Date().toISOString(), files });
            fs.writeFileSync(path.join(destination, "manifest.json"), JSON.stringify(manifest, null, 2), { flag: "wx" });
        } finally { await release(); }
    }

    static async restore(archive: string, destination: string): Promise<void> {
        const manifest = manifestSchema.parse(JSON.parse(fs.readFileSync(containedFile(archive, "manifest.json"), "utf8")));
        if (new Set(manifest.files.map(file => file.path)).size !== manifest.files.length) throw new Error("Duplicate archive file.");
        const files = manifest.files.map(file => {
            const bytes = fs.readFileSync(containedFile(archive, file.path));
            if (hash(bytes) !== file.sha256) throw new Error(`Archive checksum failed for ${file.path}.`);
            return { ...file, bytes };
        });
        if (fs.existsSync(destination) && (fs.lstatSync(destination).isSymbolicLink() || fs.readdirSync(destination).length)) throw new Error("Restore destination must be an empty directory.");
        const release = await acquireInstanceGuard(destination);
        const stage = path.join(path.dirname(path.resolve(destination)), `.sophia-restore-${randomUUID().slice(0, 8)}`);
        try {
            fs.mkdirSync(stage);
            for (const file of files) {
                const target = path.join(stage, file.path);
                fs.mkdirSync(path.dirname(target), { recursive: true });
                if (file.path === "settings.json") {
                    const settings = JSON.parse(file.bytes.toString("utf8"));
                    settings.runtime = { ...settings.runtime, operationalDbPath: path.join(path.resolve(destination), "runtime", "operational.sqlite") };
                    fs.writeFileSync(target, JSON.stringify(settings, null, 2), { flag: "wx" });
                } else fs.writeFileSync(target, file.bytes, { flag: "wx" });
            }
            // Refuse new contents rather than replacing an existing deployment.
            fs.rmdirSync(destination);
            fs.renameSync(stage, destination);
        } finally { await release(); }
    }
}
