/**
 * Probes each tool schema individually against the Gemini model via OpenRouter.
 * Reports which ones cause 400 errors (individual + binary search).
 *
 * Usage: node scripts/probe-schemas.mjs
 */
import { readFileSync, writeFileSync, unlinkSync, mkdtempSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { execSync } from "child_process";
import os from "os";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

// ── Load .env ─────────────────────────────────────────────────────────────────
const envLines = readFileSync(join(root, ".env"), "utf8").split(/\r?\n/);
const env = Object.fromEntries(
    envLines
        .filter((l) => l.includes("="))
        .map((l) => {
            const idx = l.indexOf("=");
            return [l.slice(0, idx).trim(), l.slice(idx + 1).trim()];
        })
);
const OPENROUTER_API_KEY = env.OPENROUTER_API_KEY;
const MODEL = "google/gemini-3.1-flash-lite-preview";

if (!OPENROUTER_API_KEY) { console.error("Missing OPENROUTER_API_KEY in .env"); process.exit(1); }

// ── Load tool schemas via tsx (use tmpdir to avoid spaces in path) ────────────
const tmpDir = mkdtempSync(join(os.tmpdir(), "sphprb-"));
const entryFile = join(tmpDir, "entry.ts");
// Use forward slashes inside the import so Windows tsx handles it correctly
const schemasImport = join(root, "src", "runtime", "toolSchemas.ts").replace(/\\/g, "/");
writeFileSync(entryFile, `import { TOOL_DEFINITIONS } from "${schemasImport}";\nconsole.log(JSON.stringify(TOOL_DEFINITIONS));\n`);

let toolDefs;
try {
    const out = execSync(`npx tsx "${entryFile}"`, {
        cwd: root,
        env: { ...process.env, DISCORD_TOKEN: "x", OPENROUTER_API_KEY },
        encoding: "utf8",
        timeout: 30000,
    });
    toolDefs = JSON.parse(out.trim());
} finally {
    try { unlinkSync(entryFile); } catch { /**/ }
}

const names = toolDefs.map((t) => t.function.name);
console.log(`Loaded ${toolDefs.length} tools: ${names.join(", ")}\n`);

// ── API probe ─────────────────────────────────────────────────────────────────
const MESSAGES = [{ role: "user", content: "hi" }];

async function probe(tools) {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${OPENROUTER_API_KEY}` },
        body: JSON.stringify({ model: MODEL, max_tokens: 16, messages: MESSAGES, tools }),
    });
    if (res.ok) return { ok: true };
    let body;
    try { body = await res.json(); } catch { body = await res.text(); }
    return { ok: false, status: res.status, body };
}

// ── Step 1: all tools together ────────────────────────────────────────────────
process.stdout.write("Testing ALL tools together... ");
const allResult = await probe(toolDefs);
if (allResult.ok) {
    console.log("✅ PASS — no schema issue detected right now.");
    process.exit(0);
}
console.log(`❌ FAIL (${allResult.status})\n${JSON.stringify(allResult.body, null, 2)}\n`);

// ── Step 2: each tool individually ───────────────────────────────────────────
console.log("=== Testing each tool individually ===");
const failing = [];
for (const tool of toolDefs) {
    process.stdout.write(`  ${tool.function.name.padEnd(32)} `);
    const r = await probe([tool]);
    if (r.ok) {
        console.log("✅");
    } else {
        console.log(`❌ (${r.status})`);
        console.log("    " + JSON.stringify(r.body?.error ?? r.body).slice(0, 400));
        failing.push(tool.function.name);
    }
}

if (failing.length > 0) {
    console.log(`\n🔴 Offending tools: ${failing.join(", ")}`);
} else {
    console.log("\n⚠️  No single tool failed alone — issue may require a specific combination.\n");
    console.log("=== Binary search over all tools ===");

    async function binarySearch(tools, depth = 0) {
        if (tools.length <= 1) {
            const r = await probe(tools);
            const label = tools[0]?.function.name ?? "(empty)";
            console.log(`${"  ".repeat(depth)}${r.ok ? "✅" : "❌"} ${label}`);
            return;
        }
        const mid = Math.floor(tools.length / 2);
        const left = tools.slice(0, mid);
        const right = tools.slice(mid);
        const lNames = left.map((t) => t.function.name).join(",");
        const rNames = right.map((t) => t.function.name).join(",");
        const [lRes, rRes] = await Promise.all([probe(left), probe(right)]);
        const i = "  ".repeat(depth);
        console.log(`${i}${lRes.ok ? "✅" : "❌"} [${lNames}]`);
        if (!lRes.ok) await binarySearch(left, depth + 1);
        console.log(`${i}${rRes.ok ? "✅" : "❌"} [${rNames}]`);
        if (!rRes.ok) await binarySearch(right, depth + 1);
    }

    await binarySearch(toolDefs);
}

console.log("\n=== Done ===");
