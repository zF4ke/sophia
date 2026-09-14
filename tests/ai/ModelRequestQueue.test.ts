import { expect, it, vi } from "vitest";
import { ModelRequestQueue } from "@/ai/ModelRequestQueue";

it("gives waiting conversation priority and cancels queued work without dispatch", async () => {
    const queue = new ModelRequestQueue(() => 1);
    let release!: () => void;
    const hold = new Promise<void>(resolve => { release = resolve; });
    const order: string[] = [];
    const first = queue.run("first", false, undefined, () => hold);
    const background = queue.run("dream", true, undefined, async () => { order.push("dream"); });
    const foreground = queue.run("person", false, undefined, async () => { order.push("person"); });
    const abort = new AbortController();
    const run = vi.fn();
    const cancelled = queue.run("cancelled", false, abort.signal, run);
    const rejected = expect(cancelled).rejects.toThrow("cancelled before dispatch");
    abort.abort();
    await rejected;
    release();
    await Promise.all([first, foreground, background]);
    expect(order).toEqual(["person", "dream"]);
    expect(run).not.toHaveBeenCalled();
});

it("prevents one actor from occupying every slot without capping completed work", async () => {
    const queue = new ModelRequestQueue(() => 2);
    let release!: () => void;
    const hold = new Promise<void>(resolve => { release = resolve; });
    const order: string[] = [];
    const first = queue.run("same", false, undefined, () => hold);
    const second = queue.run("same", false, undefined, async () => { order.push("same"); });
    await queue.run("other", false, undefined, async () => { order.push("other"); });
    expect(order).toEqual(["other"]);
    release();
    await Promise.all([first, second]);
    expect(order).toEqual(["other", "same"]);
});
