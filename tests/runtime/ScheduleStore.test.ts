import { afterEach, expect, it } from "vitest";
import { ScheduleStore, nextScheduleTime, type ScheduleSpec } from "@/runtime/scheduling/ScheduleStore";
import { ProductStore } from "@/runtime/storage/ProductStore";
const owner = { actorId: "owner", guildId: "g", channelId: "c" };
const now = Date.parse("2026-09-20T10:00:00Z");
const spec: ScheduleSpec = { name: "Report", prompt: "Read the recent decisions", cadence: "once", timezone: "Europe/Lisbon", firstRunAt: "2026-09-20T10:01:00Z" };
afterEach(async () => { await ProductStore.initialize(); await ProductStore.getClient().executeMultiple("DELETE FROM scheduled_runs; DELETE FROM schedules;").catch(() => {}); });
it("keeps local daily time through DST and recalculates a timezone change", () => {
    const daily: ScheduleSpec = { ...spec, cadence: "daily", localTime: "09:00" };
    expect(new Date(nextScheduleTime(daily, Date.parse("2026-03-28T09:00:00Z"))!).toISOString()).toBe("2026-03-29T08:00:00.000Z");
    expect(new Date(nextScheduleTime(daily, Date.parse("2026-10-24T08:00:00Z"))!).toISOString()).toBe("2026-10-25T09:00:00.000Z");
    expect(nextScheduleTime({ ...daily, timezone: "America/New_York" }, now)).not.toBe(nextScheduleTime(daily, now));
    expect(() => nextScheduleTime({ ...daily, timezone: "invented" }, now)).toThrow();
});
it("claims each due occurrence once and binds the right schedule to each run", async () => {
    const store = new ScheduleStore("one");
    const first = await store.save(owner, spec, {}, now);
    const second = await store.save(owner, { ...spec, name: "Other" }, {}, now);
    const claims = await Promise.all([store.claim(now + 120000), store.claim(now + 120000), store.claim(now + 120000)]);
    expect(claims.filter(Boolean)).toHaveLength(2);
    expect(new Set(claims.filter(Boolean).map(job => job!.schedule.id))).toEqual(new Set([first.id, second.id]));
    for (const job of claims.filter(Boolean)) {
        const row = (await ProductStore.getClient().execute({ sql: "SELECT schedule_id FROM scheduled_runs WHERE id=?", args: [job!.runId] })).rows[0];
        expect(row.schedule_id).toBe(job!.schedule.id);
    }
});
it("pauses an interrupted delivery on restart instead of replaying it", async () => {
    const store = new ScheduleStore("old");
    await store.save(owner, spec, {}, now);
    const job = (await store.claim(now + 120000))!;
    await store.prepareDelivery(job.runId, job.schedule, "task");
    await expect(store.prepareDelivery(job.runId, job.schedule, "task")).rejects.toThrow("already dispatched");
    const restarted = new ScheduleStore("new");
    await restarted.initialize();
    expect(await restarted.claim(now + 180000)).toBeNull();
    expect((await restarted.list(owner))[0]).toMatchObject({ status: "paused", reason: "interrupted" });
    expect((await ProductStore.getClient().execute({ sql: "SELECT error FROM scheduled_runs WHERE id=?", args: [job.runId] })).rows[0].error).toBe("delivery_unknown");
});
it("enforces owner, location and revision, and cancellation suppresses pending delivery", async () => {
    const store = new ScheduleStore();
    const saved = await store.save(owner, spec, {}, now);
    expect(await store.list({ ...owner, actorId: "other" })).toEqual([]);
    await expect(store.cancel({ ...owner, channelId: "other" }, saved.id, 1)).rejects.toThrow("owner/location");
    const job = (await store.claim(now + 120000))!;
    await store.cancel(owner, saved.id, 1);
    await expect(store.prepareDelivery(job.runId, job.schedule, "task")).rejects.toThrow("cancelled");
    await store.finish(job.runId, job.schedule, { error: "cancelled" });
    expect((await store.list(owner))[0].status).toBe("cancelled");
    await expect(store.save(owner, spec, { id: saved.id, revision: 1 }, now)).rejects.toThrow("revision changed");
});
it("coalesces missed intervals and retains unused schedules through restart", async () => {
    const store = new ScheduleStore("original");
    await store.save(owner, { ...spec, cadence: "interval", intervalMinutes: 5 }, {}, now);
    const restarted = new ScheduleStore("restarted");
    const job = (await restarted.claim(now + 31 * 60000))!;
    await restarted.finish(job.runId, job.schedule, { taskId: "task", messageId: "message" }, now + 32 * 60000);
    expect((await restarted.list(owner))[0].nextAt).toBe(now + 36 * 60000);
    expect(await restarted.claim(now + 33 * 60000)).toBeNull();
});
