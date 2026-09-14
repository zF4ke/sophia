import { afterEach, describe, expect, it, vi } from "vitest";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { taskSearchTool, taskControlTool } from "@/tools/taskControl";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { Runtime } from "@/runtime/Runtime";
import type { TurnInput } from "@/runtime/contracts";
import type { CapabilityContext } from "@/tools/types";

const binding = { actorId: "conversational-owner", guildId: null, channelId: "dm", conversationId: "dm", objective: "Compare event attendance" };
const context: CapabilityContext = { actorId: binding.actorId, guild: null, currentChannelId: "dm", question: "Stop discussing prizes; focus on attendance.", privateResponse: true, authorize: async () => "allow" };
afterEach(() => vi.restoreAllMocks());

describe("conversational task controls", () => {
    it("finds owned objectives and excludes another owner, location and current task", async () => {
        const id = await taskStore.create(binding);
        await taskStore.create({ ...binding, actorId: "other" });
        await taskStore.create({ ...binding, channelId: "other" });
        const current = await taskStore.create(binding);
        const result = await taskSearchTool.capability.run({ ...context, taskId: current }, { query: "attendance" });
        expect((result.data as any).tasks.map((task: any) => task.taskId)).toEqual([id]);
        await expect(taskControlTool.capability.run({ ...context, actorId: "other" }, { task_id: id, action: "stop" })).rejects.toThrow("unavailable");
    });
    it("steers the actual instruction and stops only the selected execution", async () => {
        const id = await taskStore.create(binding);
        const otherId = await taskStore.create(binding);
        const execution = new ExecutionControl(binding.actorId, "dm");
        const other = new ExecutionControl(binding.actorId, "dm");
        execution.bindTask(id, text => taskStore.appendSteering(id, binding.actorId, "dm", null, text));
        other.bindTask(otherId, async () => {});
        const releases = [execution.register(), other.register()];
        try {
            await taskControlTool.capability.run(context, { task_id: id, action: "steer" });
            expect(execution.steering).toEqual([context.question]);
            expect(execution.signal.aborted).toBe(false);
            await taskControlTool.capability.run(context, { task_id: id, action: "stop" });
            expect(execution.signal.aborted).toBe(true);
            expect(other.signal.aborted).toBe(false);
        } finally { releases.forEach(release => release()); }
    });
    it("reopens completed work with its saved workspace and rejects unresolved actions", async () => {
        const id = await taskStore.create(binding);
        const workspace = await taskStore.workspace(id, binding.actorId, "dm", null);
        await workspace.addRequestNote({ requestId: "old", threadId: "dm", kind: "note", label: null, body: "Use unique registrations" });
        await taskStore.finish(id, binding.actorId, "completed", "Old result");
        expect(await taskStore.canResume(id, binding.actorId, "dm", null)).toBe(true);
        const answerTurn = vi.spyOn(Runtime as any, "answerTurn").mockImplementation(async (...args: any[]) => {
            const input = args[0] as TurnInput;
            if (!input.resumeTaskId) await input.requestTaskResume!(id);
            else expect((await taskStore.snapshot(id, binding.actorId, "dm", null))?.notes[0].body).toBe("Use unique registrations");
            return { requestId: "result", answer: "Updated", outcome: "completed", citations: [], toolRuns: [] };
        });
        const input = { user: { id: binding.actorId }, guild: null, currentChannelId: "dm", question: "Revise the event comparison", conversation: { key: "dm" }, trigger: "mention", authorize: async () => "allow" } as unknown as TurnInput;
        expect((await Runtime.answer(input)).taskId).toBe(id);
        expect(answerTurn).toHaveBeenCalledTimes(2);
        await taskStore.resume(id, binding.actorId, "dm", null);
        const action = await taskStore.beginAction({ ...binding, taskId: id, invocationId: "uncertain", tool: "send_message", arguments: {} });
        await taskStore.settleAction(action, binding.actorId, "unknown", null);
        await taskStore.finish(id, binding.actorId, "paused", "Verify first");
        expect(await taskStore.canResume(id, binding.actorId, "dm", null)).toBe(false);
        expect(await taskStore.resume(id, binding.actorId, "dm", null)).toBeNull();
    });
    it("does not disclose a private task in a public response", async () => {
        const id = await taskStore.create({ ...binding, privateResponse: true });
        const result = await taskSearchTool.capability.run({ ...context, privateResponse: false }, { query: "attendance" });
        expect((result.data as any).tasks.some((task: any) => task.taskId === id)).toBe(false);
        await expect(taskControlTool.capability.run({ ...context, privateResponse: false }, { task_id: id, action: "stop" })).rejects.toThrow("private");
    });
});
