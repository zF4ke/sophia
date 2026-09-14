import { afterEach, expect, it, vi } from "vitest";
import { SkillStore, type SkillBody } from "@/memory/SkillStore";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { ModelGateway } from "@/ai/ModelGateway";
import { skillEvaluateTool } from "@/tools/skillEvaluate";

afterEach(() => vi.restoreAllMocks());
it("evaluates demonstrated work without executing a skill and binds promotion to its content", async () => {
    const audience = { actorId: "owner", guildId: "g", channelId: "c" };
    const taskId = await taskStore.create({ ...audience, conversationId: "c", objective: "Calculate" });
    await taskStore.recordToolRun({ ...audience, taskId, requestId: "r", invocationId: "call", record: { tool: "evaluate_math", arguments: { expression: "2+2" }, summary: "2+2 = 4", learned: "4", confidenceImproved: true, durationMs: 1, output: { tool: "evaluate_math", summary: "2+2 = 4", data: { result: 4 } } } });
    await taskStore.finish(taskId, "owner", "completed", "4");
    const body: SkillBody = { name: "Check arithmetic", description: "Use measured arithmetic", instructions: "Evaluate the requested expression with evaluate_math and report its returned value.", capabilities: ["evaluate_math"], examples: ["2+2 returns 4"], status: "draft" };
    await SkillStore.draftFromDream(audience, body, { dreamId: "evaluate-dream", taskId, sources: [] });
    const draft = (await SkillStore.search(audience, body.name))[0];
    const judge = vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({ passed: true, findings: ["Matches the demonstrated tool and returned result."] });
    const context = { actorId: "owner", guild: { id: "g" } as never, currentChannelId: "c", question: "Review" };
    const result = await skillEvaluateTool.capability.run(context, { id: draft.id, revision: 1 });
    expect(result.data).toMatchObject({ passed: true, taskId });
    expect(judge).toHaveBeenCalledOnce();
    expect((await SkillStore.load(audience, draft.id))?.status).toBe("draft");
    await SkillStore.save(audience, { ...body, status: "ready" }, { id: draft.id, revision: 1 });
    await expect(skillEvaluateTool.capability.run({ ...context, actorId: "other" }, { id: draft.id, revision: 2 })).rejects.toThrow("unavailable");
    const unknown = await SkillStore.save(audience, { ...body, name: "Undemonstrated", capabilities: ["delete_channel"] });
    judge.mockClear();
    const rejected = await skillEvaluateTool.capability.run(context, { id: unknown.id, revision: 1, task_id: taskId });
    expect(rejected.data).toMatchObject({ passed: false, model: "capability-check" });
    expect(judge).not.toHaveBeenCalled();
});
