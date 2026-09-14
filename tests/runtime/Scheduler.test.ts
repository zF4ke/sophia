import { afterEach, expect, it, vi } from "vitest";
import { Scheduler } from "@/runtime/scheduling/Scheduler";
import { scheduleStore, type ScheduleRecord } from "@/runtime/scheduling/ScheduleStore";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";
import { AccessPolicy } from "@/security/AccessPolicy";
import { Runtime } from "@/runtime/Runtime";
afterEach(() => vi.restoreAllMocks());
it("records an unchanged conditional check without sending, but delivers a paused outcome", async () => {
    vi.spyOn(scheduleStore, "current").mockResolvedValue(true);
    vi.spyOn(scheduleStore, "runs").mockResolvedValue([]);
    vi.spyOn(AccessPolicy, "decide").mockResolvedValue("allow");
    const finish = vi.spyOn(scheduleStore, "finish").mockResolvedValue();
    const prepare = vi.spyOn(scheduleStore, "prepareDelivery").mockResolvedValue();
    const send = vi.fn().mockResolvedValue({ id: "message" });
    const client = { channels: { fetch: async () => ({ id: "dm", recipientId: "owner", isSendable: () => true, isDMBased: () => true, send }) }, users: { fetch: async () => ({ id: "owner", username: "Owner" }) } };
    const schedule = { id: "schedule", owner: { actorId: "owner", guildId: null, channelId: "dm" }, spec: { notificationPolicy: "conditional", name: "Watch", prompt: "Check for changes" } } as ScheduleRecord;
    const answer = vi.spyOn(Runtime, "answer").mockResolvedValue({ taskId: "task", outcome: "completed", notify: false, answer: "No change" } as never);
    await Scheduler.execute(client as never, schedule, "run");
    expect(send).not.toHaveBeenCalled();
    expect(prepare).not.toHaveBeenCalled();
    expect(finish).toHaveBeenCalledWith("run", schedule, { taskId: "task" });
    answer.mockResolvedValue({ taskId: "task", outcome: "paused", notify: false, answer: "Source unavailable" } as never);
    const { ConversationAdapter } = await import("@/discord/conversation/ConversationAdapter");
    vi.spyOn(ConversationAdapter, "bindResponseMessages").mockResolvedValue();
    await Scheduler.execute(client as never, schedule, "second");
    expect(send).toHaveBeenCalledOnce();
    expect(finish).toHaveBeenLastCalledWith("second", schedule, expect.objectContaining({ error: "Task task paused" }));
});
it("waits for foreground work before claiming a schedule", async () => {
    const claim = vi.spyOn(scheduleStore, "claim").mockResolvedValue(null);
    const release = ActiveRequestTracker.begin();
    try { await Scheduler.tick({} as never); } finally { release(); }
    expect(claim).not.toHaveBeenCalled();
    await Scheduler.tick({} as never);
    expect(claim).toHaveBeenCalledOnce();
});
it("checks current owner access before model work or user lookup", async () => {
    vi.spyOn(scheduleStore, "current").mockResolvedValue(true);
    vi.spyOn(AccessPolicy, "decide").mockResolvedValue("deny");
    const answer = vi.spyOn(Runtime, "answer");
    const lookup = vi.fn();
    const client = { guilds: { fetch: async () => ({ id: "g" }) }, channels: { fetch: async () => ({ isSendable: () => true, guildId: "g" }) }, users: { fetch: lookup } };
    const schedule = { id: "schedule", owner: { actorId: "owner", guildId: "g", channelId: "c" }, spec: {} } as ScheduleRecord;
    await expect(Scheduler.execute(client as never, schedule, "run")).rejects.toThrow("revoked");
    expect(answer).not.toHaveBeenCalled();
    expect(lookup).not.toHaveBeenCalled();
});
