import path from "path";
import { beforeEach, describe, expect, it } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { memoryRememberTool, memorySearchTool } from "@/tools/longTermMemory";
import type { CapabilityContext } from "@/tools/types";

function context(guildId: string | null, actorId: string | null): CapabilityContext {
    return {
        guild: guildId ? ({ id: guildId } as any) : null,
        question: "memory test",
        actorId,
    } as CapabilityContext;
}

async function remember(
    guildId: string | null,
    actorId: string | null,
    args: { key: string; value: string; scope?: string },
) {
    return memoryRememberTool.capability.run(context(guildId, actorId), args as any);
}

function keysOf(data: unknown): string[] {
    return (data as { memories: Array<{ key: string }> }).memories.map((m) => m.key);
}

describe("long-term memory tools (FTS search)", () => {
    beforeEach(async () => {
        const operationalDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `ltm-${Date.now()}-${Math.random()}.sqlite`,
        );
        const checkpointDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `ltm-checkpoint-${Date.now()}-${Math.random()}.sqlite`,
        );
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath,
                checkpointDbPath,
            },
        });
        await DiscordMemoryService.resetForTests();
    });

    it("finds memories by keyword with diacritic-insensitive prefix matching", async () => {
        await remember("g1", "u1", { key: "deploy-prefs", value: "Preferem pnpm nos deploys em São Paulo" });
        await remember("g1", "u1", { key: "owner-note", value: "O aniversário do servidor é em março" });

        const diacritic = await memorySearchTool.capability.run(context("g1", "u1"), { query: "sao paulo" });
        expect(keysOf(diacritic.data)).toEqual(["deploy-prefs"]);

        const prefix = await memorySearchTool.capability.run(context("g1", "u1"), { query: "dep" });
        expect(keysOf(prefix.data)).toContain("deploy-prefs");
    });

    it("keeps memories guild-scoped", async () => {
        await remember("g1", "u1", { key: "secret-g1", value: "only in guild one" });
        await remember("g2", "u1", { key: "secret-g2", value: "only in guild two" });

        const fromG1 = await memorySearchTool.capability.run(context("g1", "u1"), { query: "secret" });
        expect(keysOf(fromG1.data)).toEqual(["secret-g1"]);
    });

    it("hides other users' user-scoped memories but shows their own", async () => {
        await remember("g1", "u1", { key: "alice-pref", value: "alice likes night mode", scope: "user" });
        await remember("g1", "u2", { key: "bob-pref", value: "bob likes light mode", scope: "user" });
        await remember("g1", null, { key: "shared-rule", value: "no politics in general" });

        const forAlice = await memorySearchTool.capability.run(context("g1", "u1"), { query: "" });
        const aliceKeys = keysOf(forAlice.data);
        expect(aliceKeys).toContain("alice-pref");
        expect(aliceKeys).not.toContain("bob-pref");
        expect(aliceKeys).toContain("shared-rule");

        const forBob = await memorySearchTool.capability.run(context("g1", "u2"), { query: "mode" });
        expect(keysOf(forBob.data)).toEqual(["bob-pref"]);
    });

    it("lists recent memories when query is empty", async () => {
        await remember("g1", "u1", { key: "old", value: "first saved" });
        await remember("g1", "u1", { key: "new", value: "second saved" });

        const result = await memorySearchTool.capability.run(context("g1", "u1"), {});
        expect(keysOf(result.data)).toEqual(["new", "old"]);
    });

    it("rebuilds the FTS index for rows written before the index existed", async () => {
        await remember("g1", "u1", { key: "legacy", value: "written before backfill ran" });

        // Simulate a legacy DB: wipe the FTS mirror + marker, then re-init.
        const c = OperationalStore.getClient();
        await c.executeMultiple(`
            DELETE FROM long_term_memories_fts;
            DELETE FROM runtime_metadata WHERE key = 'ltm_fts_backfilled';
        `);
        await OperationalStore.reset();
        await OperationalStore.initialize();

        const result = await memorySearchTool.capability.run(context("g1", "u1"), { query: "legacy" });
        expect(keysOf(result.data)).toEqual(["legacy"]);
    });
});
