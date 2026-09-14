import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
// Both live and deterministic suites keep settings and databases out of the
// deployment storage. This file deliberately leaves networking untouched.
const root = path.join(process.cwd(), "storage", "test-runtime-store", "__vitest__", randomUUID(), "storage");
fs.mkdirSync(root, { recursive: true });
process.env.SOPHIA_STORAGE_ROOT = root;
