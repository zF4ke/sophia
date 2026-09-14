import path from "node:path";
import { randomUUID } from "node:crypto";
import { expect, it } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { KnowledgeStore } from "@/memory/KnowledgeStore";

it("carries presentation preferences across locations without sharing facts, source links or another person's settings", async () => {
    const store = new KnowledgeStore(path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`));
    const original = { actorId: "owner", guildId: "guild-one", channelId: "staff" };
    const elsewhere = { ...original, guildId: "guild-two", channelId: "public" };
    try {
        const source = "https://discord.com/channels/guild-one/staff/preference";
        await store.remember(original, { key: "Project", value: "Private project", scope: "channel", sources: [source] });
        await store.enqueueDream("preference-dream", { audience: original, question: "Keep replies brief in general", answer: "Okay", sources: [source] });
        await store.nextDream();
        await store.finishDream("preference-dream", [{ key: "response_length", value: "brief", scope: "preference" }]);
        const preferences = await store.search(elsewhere);
        expect(preferences).toHaveLength(1);
        expect(preferences[0]).toMatchObject({ key: "response_length", value: "brief", sources: [], scope: "preference" });
        expect(await store.preferences("owner")).toEqual({ response_length: "brief" });
        expect(await store.search({ ...elsewhere, actorId: "other" })).toEqual([]);
        await expect(store.remember(elsewhere, { key: "private_fact", value: "secret", scope: "preference" })).rejects.toThrow("Portable preferences");
        await expect(store.revise(elsewhere, preferences[0].id, 1, "Secret fact")).rejects.toThrow("Portable preferences");
        await store.revise(elsewhere, preferences[0].id, 1, "detailed");
        expect((await store.search(original)).find(memory => memory.scope === "preference")?.value).toBe("detailed");
        await store.revise(elsewhere, preferences[0].id, 2, null);
        await store.enqueueDream("recreated", { audience: elsewhere, question: "Brief", answer: "Okay", sources: [] });
        await store.nextDream();
        await store.finishDream("recreated", [{ key: "response_length", value: "brief", scope: "preference" }]);
        expect(await store.search(elsewhere)).toEqual([]);
        expect((await store.dreamContext(elsewhere)).suppressedLabels).toContain("response_length");
    } finally { await store.close(); }
});
