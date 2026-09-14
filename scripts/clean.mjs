import fs from "node:fs";
import path from "node:path";
if (fs.existsSync('.env')) process.loadEnvFile('.env');

// Runtime databases may still contain records awaiting migration. Keep them,
// settings, security configuration, task workspaces and durable knowledge.
const root = path.resolve(process.env.SOPHIA_STORAGE_ROOT || "storage");
for (const name of ["logs", "test-memory", "test-runtime-store"]) {
    const target = path.resolve(root, name);
    if (path.dirname(target) !== root) throw new Error("Cleanup target escaped storage.");
    if (!fs.existsSync(target)) continue;
    if (fs.lstatSync(target).isSymbolicLink()) throw new Error(`Refusing linked cleanup directory: ${target}`);
    fs.rmSync(target, { recursive: true, force: true });
    fs.mkdirSync(target, { recursive: true });
    console.log(`Cleaned ${name}`);
}
console.log("Preserved settings, access rules and databases. Use the index reset command to rebuild retrieval data after startup migration.");
