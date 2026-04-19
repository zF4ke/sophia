import fs from "fs";
import path from "path";

// Redirect all mutable state to an isolated per-run directory so tests
// never touch the real storage/settings.json or storage/runtime DBs.
// Must run before any module imports AppPaths (vitest setupFiles do).
const testStorageRoot = path.join(
    process.cwd(),
    "storage",
    "test-runtime-store",
    "__vitest__",
    "storage",
);
fs.mkdirSync(testStorageRoot, { recursive: true });
process.env.SOPHIA_STORAGE_ROOT = testStorageRoot;
