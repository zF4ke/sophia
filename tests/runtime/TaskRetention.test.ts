import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { forgetTask } from "@/runtime/tasks/TaskRetention";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { SkillStore, type SkillBody } from "@/memory/SkillStore";

const owner = { actorId: "owner", guildId: "guild", channelId: "channel", conversationId: "conversation", objective: "Private research" };
beforeEach(() => {
    vi.stubEnv("DISCORD_TOKEN", "test-token");
    vi.stubEnv("OPENROUTER_API_KEY", "test-key");
});
afterEach(() => vi.unstubAllEnvs());
it("removes only owned inactive work and prevents pending learning from recreating its drafts", async () => {
    const id = await taskStore.create(owner);
    const neighbor = await taskStore.create(owner);
    const workspace = await taskStore.workspace(id, owner.actorId, owner.channelId, owner.guildId);
    await taskStore.recordEvidenceSources({ ...owner, taskId: id, requestId: "turn" }, []);
    await workspace.addRequestNote({ requestId: "turn", threadId: "conversation", kind: "note", label: null, body: "Derived private note" });
    await taskStore.replaceFiles(id, owner.actorId, owner.channelId, owner.guildId, [{ path: "report.txt", data: Buffer.from("private").toString("base64") }]);
    const audience = { actorId: owner.actorId, guildId: owner.guildId, channelId: owner.channelId };
    const body: SkillBody = { name: "Count text", description: "Measure a requested text", instructions: "Use the text length tool with the supplied text.", capabilities: ["measure_text_length"], examples: [], status: "draft" };
    const provenance = { dreamId: "dream", taskId: id, sources: [] };
    await knowledgeStore.enqueueDream("dream", { audience, taskId: id, question: "private question", answer: "private answer", sources: [] });
    await SkillStore.draftFromDream(audience, body, provenance);
    await expect(forgetTask(id, owner.actorId, owner.channelId, owner.guildId)).rejects.toThrow("inactive");
    await taskStore.finish(id, owner.actorId, "completed", "done");
    await expect(forgetTask(id, "stranger", owner.channelId, owner.guildId)).rejects.toThrow("owned");
    await forgetTask(id, owner.actorId, owner.channelId, owner.guildId);
    await forgetTask(id, owner.actorId, owner.channelId, owner.guildId);
    expect(await taskStore.snapshot(id, owner.actorId, owner.channelId, owner.guildId)).toBeNull();
    expect(await taskStore.files(id, owner.actorId, owner.channelId, owner.guildId)).toEqual([]);
    expect(await taskStore.requestSources("turn")).toBeNull();
    expect(await taskStore.snapshot(neighbor, owner.actorId, owner.channelId, owner.guildId)).not.toBeNull();
    expect(await knowledgeStore.nextDream()).toBeNull();
    await SkillStore.draftFromDream(audience, body, provenance);
    expect(await SkillStore.load(audience, "dream:dream")).toBeNull();
});

it("retains unresolved action records until they have been verified", async () => {
    const id = await taskStore.create(owner);
    const actionId = await taskStore.beginAction({ ...owner, taskId: id, invocationId: "unknown", tool: "send_message", arguments: {} });
    await taskStore.settleAction(actionId, owner.actorId, "unknown", null);
    await taskStore.finish(id, owner.actorId, "paused", "verify first");
    await expect(forgetTask(id, owner.actorId, owner.channelId, owner.guildId)).rejects.toThrow("resolved actions");
    expect((await taskStore.snapshot(id, owner.actorId, owner.channelId, owner.guildId))?.actions[0].status).toBe("unknown");
});
