import { afterEach, expect, it, vi } from "vitest";
import { randomUUID } from "node:crypto";
import { Runtime } from "@/runtime/Runtime";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { boundLiveModelCalls } from "./modelBudget";
afterEach(() => vi.restoreAllMocks());
const live = process.env.LIVE_MODEL_TESTS === "1" ? it : it.skip;
for (const action of ["steer", "stop", "ambiguous"] as const) live(`interprets natural task ${action} without user IDs`, async () => {
    boundLiveModelCalls(6);
    const actorId = randomUUID();
    const channelId = randomUUID();
    const binding = { actorId, channelId, guildId: null, conversationId: channelId, objective: "Compare attendance at the last three community events" };
    const first = await taskStore.create(binding);
    const execution = new ExecutionControl(actorId, channelId);
    execution.bindTask(first, text => taskStore.appendSteering(first, actorId, channelId, null, text));
    const release = execution.register();
    if (action === "ambiguous") await taskStore.create({ ...binding, objective: "Compare attendance at the last three staff events" });
    try {
        const question = action === "steer" ? "Sophia, stop discussing prizes in the community event comparison. Focus on scheduling conflicts instead."
            : action === "stop" ? "Sophia, stop the community event comparison you're working on. I don't need it anymore."
            : "Sophia, stop that event comparison.";
        const result = await Runtime.answer({ question, user: { id: actorId } as never, requesterDisplayName: "Tester", guild: null,
            currentChannelId: channelId, trigger: "mention", authorize: async effect => effect === "none" ? "allow" : "deny",
            conversation: { key: channelId, kind: "channel", trigger: "mention", replyAnchorMessageId: null, nativeThreadId: null } });
        expect(result.outcome).toBe("completed");
        if (action === "steer") { expect(execution.steering).toContain(question); expect(execution.signal.aborted).toBe(false); }
        if (action === "stop") expect(execution.signal.aborted).toBe(true);
        if (action === "ambiguous") { expect(execution.signal.aborted).toBe(false); expect(result.toolRuns.some(run => run.tool === "task_control")).toBe(false); expect(result.answer).toMatch(/\?/); }
        console.info(JSON.stringify({ action, answer: result.answer, tools: result.toolRuns.map(run => run.tool) }));
    } finally { release(); }
});

live("continues completed work by description and uses its saved notes", async () => {
    boundLiveModelCalls(8);
    const actorId = randomUUID(), channelId = randomUUID();
    const id = await taskStore.create({ actorId, channelId, guildId: null, conversationId: channelId, objective: "Draft the welcome message for the astronomy club" });
    const workspace = await taskStore.workspace(id, actorId, channelId, null);
    await workspace.addRequestNote({ requestId: "old", threadId: channelId, kind: "note", label: "Agreed meeting", body: "The club meets on Thursdays at 19:30. Preserve this schedule in revisions." });
    await taskStore.finish(id, actorId, "completed", "Welcome to the astronomy club. Our telescope is named Moonbeam.");
    const result = await Runtime.answer({ question: "Sophia, pick up the astronomy club welcome message we worked on. Make it friendlier and include the meeting time from your saved notes. Keep our telescope name from the original text. Just give me the revised text.", user: { id: actorId } as never, requesterDisplayName: "Tester", guild: null, currentChannelId: channelId, trigger: "mention", authorize: async effect => effect === "none" ? "allow" : "deny", conversation: { key: channelId, kind: "channel", trigger: "mention", replyAnchorMessageId: null, nativeThreadId: null } });
    expect(result.taskId).toBe(id);
    expect(result.outcome).toBe("completed");
    expect(result.answer).toMatch(/19:30|7:30/);
    expect(result.answer).toContain("Moonbeam");
    console.info(JSON.stringify({ action: "continue", answer: result.answer }));
});
