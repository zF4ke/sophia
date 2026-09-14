import { describe, expect, it } from "vitest";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import type { CapabilityContext } from "@/tools/types";

describe("task-scoped working tools", () => {
    it("reads earlier legs by default and cannot update a neighboring task's goal", async () => {
        const input = { actorId: "owner", guildId: "guild", channelId: "channel", conversationId: "same-thread", objective: "Work" };
        const a = await taskStore.create(input);
        const b = await taskStore.create(input);
        const context: CapabilityContext = { taskId: a, actorId: "owner", guild: { id: "guild" } as never,
            currentChannelId: "channel", threadId: "same-thread", requestId: "first-leg", question: "Work" };
        await CapabilityRegistry.get("note_add").run(context, { body: "Useful checkpoint" });
        await CapabilityRegistry.get("plan_update").run(context, { body: "Remaining steps" });
        const noteList = await CapabilityRegistry.get("note_list").run({ ...context, requestId: "next-leg" }, {});
        expect(noteList.data).toMatchObject({ plan: "Remaining steps", notes: [{ body: "Useful checkpoint" }] });
        const goal = await CapabilityRegistry.get("goal_open").run(context, { body: "Evaluate a+b" });
        const different = await CapabilityRegistry.get("goal_open").run(context, { body: "Evaluate a-b" });
        expect((different.data as any).seq).not.toBe((goal.data as any).seq);
        const workspace = await CapabilityRegistry.get("note_list").run({ ...context, requestId: "later-leg" }, {});
        expect(workspace.data).toMatchObject({ goals: [{ seq: (goal.data as any).seq, status: "open" }, { seq: (different.data as any).seq, status: "open" }] });
        const foreignUpdate = await CapabilityRegistry.get("goal_done").run({ ...context, taskId: b }, { seq: (goal.data as any).seq });
        expect(foreignUpdate.errorMessage).toBe("not_found");
        const other = await CapabilityRegistry.get("note_list").run({ ...context, taskId: b }, { include_thread_history: true });
        expect(other.data).toMatchObject({ plan: null, notes: [] });
        await taskStore.finish(a, "owner", "paused", "test end");
        await taskStore.finish(b, "owner", "completed", "test end");
    });
});
