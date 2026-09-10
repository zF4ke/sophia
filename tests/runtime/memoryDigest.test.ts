import path from "path";
import { beforeEach, describe, expect, it } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { buildMemoryDigest } from "@/runtime/memoryDigest";
import { memoryRememberTool } from "@/tools/longTermMemory";
import type { CapabilityContext } from "@/tools/types";

function context(guildId: string | null, actorId: string | null): CapabilityContext {
    return {
        guild: guildId ? ({ id: guildId } as any) : null,
        question: "digest test",
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

describe("buildMemoryDigest", () => {
    beforeEach(async () => {
        const operationalDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `digest-${Date.now()}-${Math.random()}.sqlite`,
        );
        const checkpointDbPath = path.join(
            process.cwd(),
            "storage",
            "test-memory",
            `digest-checkpoint-${Date.now()}-${Math.random()}.sqlite`,
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

    it("returns an empty-state hint when nothing is saved", async () => {
        const digest = await buildMemoryDigest("g1", "u1");
        expect(digest).toContain("No long-term memories saved yet");
        expect(digest).toContain("memory_remember");
    });

    it("summarizes counts and recent keys without leaking other users' memories", async () => {
        await remember("g1", null, { key: "deploy-checklist", value: "run pnpm build then deploy" });
        await remember("g1", null, { key: "owner-prefs", value: "prefers respostas curtas" });
        await remember("g1", null, { key: "server-rule", value: "sem politica no geral" });
        await remember("g1", "u1", { key: "alice-pref", value: "alice likes night mode", scope: "user" });
        await remember("g1", "u2", { key: "bob-pref", value: "bob likes light mode", scope: "user" });

        const digest = await buildMemoryDigest("g1", "u1");
        expect(digest).toContain("3 shared, 1 from this user");
        expect(digest).toContain("deploy-checklist");
        expect(digest).toContain("alice-pref");
        expect(digest).not.toContain("bob-pref");
    });

    it("caps the preview length", async () => {
        await remember("g1", null, {
            key: "long-memory-key-that-keeps-going-and-going-and-going",
            value: "x".repeat(300),
        });

        const digest = await buildMemoryDigest("g1", "u1");
        expect(digest.length).toBeLessThan(600);
        expect(digest).toContain("long-memory-key-that-keeps-going-and-going-and-going".slice(0, 40));
    });

    it("is not available in DMs", async () => {
        const digest = await buildMemoryDigest(null, "u1");
        expect(digest).toContain("Not available in DMs.");
    });
});
