import path from "node:path";
import { randomUUID } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppPaths } from "@/app/AppPaths";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";
import { ModelGateway } from "@/ai/ModelGateway";
import { KnowledgeStore, knowledgeStore } from "@/memory/KnowledgeStore";
import { DreamingService } from "@/memory/DreamingService";
import { SkillStore } from "@/memory/SkillStore";
let store: KnowledgeStore;
const audience = { actorId: "u1", guildId: "g1", channelId: "c1" };
const input = { audience, question: "Please use Portuguese", answer: "Combinado.", sources: ["https://discord.com/channels/g1/c1/m1"] };
beforeEach(async () => {
    store = new KnowledgeStore(path.join(AppPaths.storageRoot, `${randomUUID()}.sqlite`));
    vi.spyOn(knowledgeStore, "nextDream").mockImplementation(store.nextDream.bind(store));
    vi.spyOn(knowledgeStore, "finishDream").mockImplementation(store.finishDream.bind(store));
    vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({ memories: [{ key: "Language preference", value: "The requester prefers Portuguese." }] });
    await store.enqueueDream("job", input);
});
afterEach(async () => { vi.restoreAllMocks(); await store.close(); });
describe("idle dreaming", () => {
    it("consolidates private experience into owner-only memory and excludes it from public dreams", async () => {
        await store.enqueueDream("private-job", { audience: { ...audience, privateResponse: true }, question: "Private project", answer: "Private checkpoint", sources: [] });
        await DreamingService.tick(async () => true);
        vi.mocked(ModelGateway.generateJson).mockResolvedValue({ memories: [{ key: "project", value: "Private project checkpoint" }] });
        await DreamingService.tick(async () => true);
        expect((await store.search(audience)).some(memory => memory.key === "project")).toBe(false);
        expect(await store.search({ ...audience, guildId: "g2", channelId: "c2", privateResponse: true }, "project")).toMatchObject([{ scope: "user" }]);
        expect(JSON.stringify(await store.dreamContext(audience))).not.toContain("Private project checkpoint");
        expect(JSON.stringify(await store.dreamContext({ ...audience, privateResponse: true }))).toContain("Private project checkpoint");
    });
    it("drafts only demonstrated procedures and never publishes them automatically", async () => {
        const draft = vi.spyOn(SkillStore, "draftFromDream").mockResolvedValue();
        vi.spyOn(knowledgeStore, "nextDream").mockResolvedValue({ id: "learning-job", input: { ...input, taskId: "task", toolEvidence: [{ tool: "retrieve_messages", summary: "Read a chronological page", succeeded: true }] } });
        vi.mocked(ModelGateway.generateJson).mockResolvedValue({ memories: [], skill: { name: "Read history", description: "Collect a timeline with cursors", instructions: "Keep the returned cursor and the same channel filters across pages.", capabilities: ["retrieve_messages"], examples: ["Read a requested period."] } });
        await DreamingService.tick(async () => true);
        expect(draft).toHaveBeenCalledWith(audience, expect.objectContaining({ status: "draft" }), { dreamId: "learning-job", taskId: "task", sources: input.sources });
        draft.mockClear();
        vi.mocked(ModelGateway.generateJson).mockResolvedValue({ memories: [], skill: { name: "Delete channels", description: "Unsupported procedure", instructions: "A method that was never demonstrated.", capabilities: ["delete_channel"], examples: [] } });
        await DreamingService.tick(async () => true);
        expect(draft).not.toHaveBeenCalled();
    });
    it("consolidates once with fixed sources and audience", async () => {
        await DreamingService.tick(async () => true);
        expect(await store.search(audience)).toMatchObject([{ scope: "channel", sources: input.sources }]);
        expect(await store.search({ ...audience, channelId: "c2" })).toEqual([]);
        await store.enqueueDream("job", input);
        await DreamingService.tick(async () => true);
        expect(ModelGateway.generateJson).toHaveBeenCalledOnce();
    });
    it("does no paid work while a user request is active", async () => {
        const release = ActiveRequestTracker.begin();
        try { await DreamingService.tick(async () => true); }
        finally { release(); }
        expect(ModelGateway.generateJson).not.toHaveBeenCalled();
        await DreamingService.tick(async () => true);
        expect(ModelGateway.generateJson).toHaveBeenCalledOnce();
    });
    it("discards consolidation when access is revoked during generation", async () => {
        const authorize = vi.fn().mockResolvedValueOnce(true).mockResolvedValue(false);
        await DreamingService.tick(authorize);
        expect(await store.search(audience)).toEqual([]);
    });
    it("does not recreate a forgotten label from a later conversation", async () => {
        await DreamingService.tick(async () => true);
        const [memory] = await store.search(audience);
        await store.revise(audience, memory.id, memory.revision, null);
        await store.enqueueDream("another-job", input);
        await DreamingService.tick(async () => true);
        expect(await store.search(audience)).toEqual([]);
    });
});
