import path from "node:path";
import { randomUUID } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { KnowledgeStore, knowledgeStore } from "@/memory/KnowledgeStore";
import { memoryRememberTool, memorySearchTool, memoryUpdateTool, memoryForgetTool } from "@/tools/longTermMemory";
import type { CapabilityContext } from "@/tools/types";

let store: KnowledgeStore;
const audience = { actorId: "u1", guildId: "g1", channelId: "c1" };
function context(actorId = "u1", guildId: string | null = "g1", channelId = "c1"): CapabilityContext {
    return { actorId, guild: guildId ? { id: guildId, members: { fetch: async () => ({ id: actorId }) }, channels: { fetch: async () => ({ isTextBased: () => true, permissionsFor: () => ({ has: () => true }) }) } } as never : null, currentChannelId: channelId, question: "Memory" };
}
beforeEach(() => {
    store = new KnowledgeStore(path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`));
    vi.spyOn(knowledgeStore, "remember").mockImplementation(store.remember.bind(store));
    vi.spyOn(knowledgeStore, "search").mockImplementation(store.search.bind(store));
    vi.spyOn(knowledgeStore, "revise").mockImplementation(store.revise.bind(store));
    vi.spyOn(knowledgeStore, "ownedMetadata").mockImplementation(store.ownedMetadata.bind(store));
});
afterEach(async () => { vi.restoreAllMocks(); await store.close(); });

describe("durable memory tools", () => {
    it("keeps private memories out of public discovery while recalling them privately in another guild", async () => {
        await memoryRememberTool.capability.run({ ...context(), privateResponse: true }, { key: "personal-project", value: "Private project checkpoint" });
        expect((await memorySearchTool.capability.run(context(), {})).data).toMatchObject({ memories: [] });
        expect((await memorySearchTool.capability.run({ ...context("u1", "g2", "c2"), privateResponse: true }, {})).data).toMatchObject({ memories: [{ key: "personal-project", scope: "user" }] });
        expect((await memorySearchTool.capability.run({ ...context("u2", "g2", "c2"), privateResponse: true }, {})).data).toMatchObject({ memories: [] });
        await expect(memoryRememberTool.capability.run({ ...context(), privateResponse: true }, { key: "private-fact", value: "Sensitive fact", scope: "guild" })).rejects.toThrow("private conversation");
    });
    it("finds accented words and prefixes while preserving source links", async () => {
        await memoryRememberTool.capability.run(context(), { key: "deploy-prefs", value: "Preferem pnpm em São Paulo", sources: ["https://discord.com/channels/g1/c1/m1"] });
        for (const query of ["sao paulo", "dep"]) {
            const output = await memorySearchTool.capability.run(context(), { query });
            expect(output.data).toMatchObject({ memories: [{ key: "deploy-prefs", sources: ["https://discord.com/channels/g1/c1/m1"] }] });
        }
    });
    it("enforces channel, guild and personal audiences across locations", async () => {
        await store.remember(audience, { key: "channel", value: "private channel fact", scope: "channel" });
        await store.remember(audience, { key: "guild", value: "shared guild fact", scope: "guild" });
        await store.remember(audience, { key: "personal", value: "owner preference", scope: "user" });
        expect((await store.search({ ...audience, actorId: "u2" })).map(m => m.key)).toEqual(["guild", "channel"]);
        expect((await store.search({ ...audience, channelId: "c2" })).map(m => m.key)).toEqual(["guild"]);
        expect(await store.search({ ...audience, guildId: "g2", channelId: "c2" })).toEqual([]);
        expect((await store.search({ actorId: "u1", guildId: null, channelId: "dm" })).map(m => m.key)).toEqual(["personal"]);
        expect(await store.search({ actorId: "u2", guildId: null, channelId: "dm" })).toEqual([]);
    });
    it("uses revision checks and ownership for correction and forgetting", async () => {
        const memory = await store.remember(audience, { key: "preference", value: "old preference", scope: "channel" });
        const args = { memory_id: memory.id, revision: 1, value: "new preference" };
        await expect(memoryUpdateTool.capability.run(context("u2"), args)).rejects.toThrow();
        await memoryUpdateTool.capability.run(context(), args);
        await expect(memoryUpdateTool.capability.run(context(), args)).rejects.toThrow();
        await memoryForgetTool.capability.run(context(), { memory_id: memory.id, revision: 2 });
        expect(await store.search(audience)).toEqual([]);
        await expect(store.remember(audience, { key: "preference", value: "old preference", scope: "channel" })).rejects.toThrow("forgotten");
    });
    it("keeps the same identity and knowledge after reopening", async () => {
        const first = await store.identity();
        await store.remember(audience, { key: "retained", value: "durable fact", scope: "channel" });
        await store.close();
        expect(await store.identity()).toEqual(first);
        expect(await store.search(audience)).toHaveLength(1);
    });
});
