import path from "node:path";
import fs from "node:fs";
if (fs.existsSync('.env')) process.loadEnvFile('.env');
const { StateArchive } = require('../src/shared/storage/StateArchive') as typeof import('../src/shared/storage/StateArchive');
const { AppPaths } = require('../src/app/AppPaths') as typeof import('../src/app/AppPaths');

async function main() {
    const [action, source, destination] = process.argv.slice(2);
    if (action === "backup" && source && !destination) {
        await StateArchive.create(AppPaths.storageRoot, path.resolve(source));
        console.log(`Durable state saved to ${path.resolve(source)}. Credentials and operational indexes are excluded.`);
    } else if (action === "restore" && source && destination) {
        await StateArchive.restore(path.resolve(source), path.resolve(destination));
        console.log(`State restored to ${path.resolve(destination)}. Start Sophia with this storage root; startup pauses interrupted work and rechecks access.`);
    } else throw new Error("Usage: npm run state -- backup <new-directory> | restore <archive-directory> <empty-storage-directory>. Stop Sophia first.");
}
void main().catch(error => { console.error(error instanceof Error ? error.message : "State transfer failed."); process.exitCode = 1; });
