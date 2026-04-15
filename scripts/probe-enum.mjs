/**
 * Minimal test: create_thread schema with and without numeric enum.
 */
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "..");

const envLines = readFileSync(join(root, ".env"), "utf8").split(/\r?\n/);
const env = Object.fromEntries(envLines.filter(l => l.includes("=")).map(l => {
    const idx = l.indexOf("=");
    return [l.slice(0, idx).trim(), l.slice(idx + 1).trim()];
}));
const KEY = env.OPENROUTER_API_KEY;
const MODEL = "google/gemini-3.1-flash-lite-preview";
const MESSAGES = [{ role: "user", content: "hi" }];

async function probe(label, parameters) {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${KEY}` },
        body: JSON.stringify({ model: MODEL, max_tokens: 16, messages: MESSAGES, tools: [{
            type: "function",
            function: { name: "create_thread", description: "Create a thread", parameters }
        }]})
    });
    let body; try { body = await res.json(); } catch { body = await res.text(); }
    const status = res.ok ? "✅ PASS" : `❌ FAIL (${res.status})`;
    console.log(`${label}: ${status}`);
    if (!res.ok) console.log("  ", JSON.stringify(body?.error?.message ?? body?.error ?? body).slice(0, 300));
}

// Test 1: full schema with numeric enum (current state)
await probe("WITH numeric enum", {
    type: "object",
    properties: {
        channel_id: { type: "string", description: "Parent channel ID." },
        name: { type: "string", description: "Thread name." },
        message: { type: "string", description: "Optional initial message." },
        auto_archive_duration: { type: "number", enum: [60, 1440, 4320, 10080], description: "Duration in minutes." }
    },
    required: ["channel_id", "name"]
});

// Test 2: no enum on auto_archive_duration
await probe("WITHOUT enum (plain number)", {
    type: "object",
    properties: {
        channel_id: { type: "string", description: "Parent channel ID." },
        name: { type: "string", description: "Thread name." },
        message: { type: "string", description: "Optional initial message." },
        auto_archive_duration: { type: "number", description: "Duration in minutes: 60, 1440, 4320, 10080. Default 1440." }
    },
    required: ["channel_id", "name"]
});

// Test 3: string enum instead
await probe("WITH string enum", {
    type: "object",
    properties: {
        channel_id: { type: "string", description: "Parent channel ID." },
        name: { type: "string", description: "Thread name." },
        message: { type: "string", description: "Optional initial message." },
        auto_archive_duration: { type: "string", enum: ["60", "1440", "4320", "10080"], description: "Duration in minutes." }
    },
    required: ["channel_id", "name"]
});

// Test 4: no auto_archive_duration at all
await probe("WITHOUT auto_archive_duration", {
    type: "object",
    properties: {
        channel_id: { type: "string", description: "Parent channel ID." },
        name: { type: "string", description: "Thread name." },
        message: { type: "string", description: "Optional initial message." }
    },
    required: ["channel_id", "name"]
});

console.log("Done.");
