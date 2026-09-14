import { describe, expect, it } from "vitest";
import { ExecutionControl, ExecutionStopped } from "@/runtime/ExecutionControl";

describe("execution control", () => {
    it("targets an exact task or bound reply message without redirecting another user's work", async () => {
        const first = new ExecutionControl("owner", "channel");
        const second = new ExecutionControl("owner", "channel");
        const saved: string[] = [];
        first.bindTask("first", async text => { saved.push(text); });
        second.bindTask("second", async () => {});
        const releases = [first.register(), second.register()];
        try {
            ExecutionControl.bindTaskMessage("first", "progress");
            expect(await ExecutionControl.steerForReply("stranger", "channel", "progress", "forged")).toBe("not_found");
            expect(await ExecutionControl.steerForReply("owner", "elsewhere", "progress", "wrong location")).toBe("not_found");
            expect(await ExecutionControl.steerForReply("owner", "channel", "progress", "Use the revised date")).toBe("queued");
            expect(saved).toEqual(["Use the revised date"]);
            expect(await ExecutionControl.steerForActor("owner", "channel", "Second task only", "second")).toBe("queued");
            expect(first.steering).toHaveLength(1);
            expect(second.steering).toEqual(["Second task only"]);
            first.closeSteering();
            expect(await ExecutionControl.steerForReply("owner", "channel", "progress", "late")).toBe("not_found");
        } finally { releases.forEach(release => release()); }
    });
    it("pauses readers of deleted sources while allowing the execution deleting them to confirm its action", () => {
        const deleting = new ExecutionControl("owner", "channel");
        const reading = new ExecutionControl("other", "channel");
        const unrelated = new ExecutionControl("third", "channel");
        const releases = [deleting.register(), reading.register(), unrelated.register()];
        try {
            deleting.watchSources(["message"]);
            reading.watchSources(["message"]);
            unrelated.watchSources(["different"]);
            deleting.expectSourceDeletion(["message"]);
            ExecutionControl.invalidateSource("message");
            expect(() => deleting.checkpoint()).not.toThrow();
            expect(() => unrelated.checkpoint()).not.toThrow();
            expect(reading.sourceInvalidated).toBe(true);
            expect(() => reading.checkpoint()).toThrow("A source used by this execution changed or was deleted.");
        } finally { releases.forEach(release => release()); }
    });
    it("routes steering only to a unique active execution owned by this actor in this channel", async () => {
        const own = new ExecutionControl("owner", "channel");
        const other = new ExecutionControl("other", "channel");
        const elsewhere = new ExecutionControl("owner", "elsewhere");
        const releases = [own.register(), other.register(), elsewhere.register()];
        try {
            expect(await ExecutionControl.steerForActor("stranger", "channel", "change")).toBe("not_found");
            expect(await ExecutionControl.steerForActor("owner", "channel", "Use Portuguese")).toBe("queued");
            expect(own.steering).toEqual(["Use Portuguese"]);
            expect(other.steeringRevision).toBe(0);
            expect(elsewhere.steeringRevision).toBe(0);
            const second = new ExecutionControl("owner", "channel");
            releases.push(second.register());
            expect(await ExecutionControl.steerForActor("owner", "channel", "Ambiguous")).toBe("ambiguous");
            expect(own.steeringRevision).toBe(1);
            own.cancel();
            expect(await ExecutionControl.steerForActor("owner", "channel", "Only second")).toBe("queued");
            expect(second.steering).toEqual(["Only second"]);
        } finally { releases.forEach(release => release()); }
        expect(await ExecutionControl.steerForActor("owner", "channel", "too late")).toBe("not_found");
    });

    it("retains ordered corrections across continuation legs without exposing mutable state", () => {
        const execution = new ExecutionControl("owner", "channel");
        execution.steer(" First instruction ");
        execution.steer("Second instruction");
        const copy = execution.steering as string[];
        copy.push("forged");
        expect(execution.steering).toEqual(["First instruction", "Second instruction"]);
        expect(() => execution.steer(" ")).toThrow();
    });
    it("allows sustained work with no default total call limit", () => {
        const execution = new ExecutionControl("owner", "channel");
        for (let i = 0; i < 10000; i++) execution.beforeTool();
        expect(execution.toolCalls).toBe(10000);
        expect(() => execution.checkpoint()).not.toThrow();
    });

    it("enforces an explicit limit without resetting the shared execution", () => {
        const execution = new ExecutionControl("owner", "channel", 2);
        execution.beforeTool();
        execution.beforeTool();
        expect(() => execution.beforeTool()).toThrow(ExecutionStopped);
        expect(execution.toolCalls).toBe(2);
    });

    it("cancels only the authenticated actor's work in the requested channel", () => {
        const own = new ExecutionControl("owner", "channel");
        const other = new ExecutionControl("other", "channel");
        const elsewhere = new ExecutionControl("owner", "elsewhere");
        const release = [own.register(), other.register(), elsewhere.register()];
        try {
            expect(ExecutionControl.cancelForActor("owner", "channel")).toBe(1);
            expect(() => own.beforeTool()).toThrow(ExecutionStopped);
            expect(() => other.beforeTool()).not.toThrow();
            expect(() => elsewhere.beforeTool()).not.toThrow();
        } finally { release.forEach(fn => fn()); }
        expect(ExecutionControl.cancelForActor("owner", "channel")).toBe(0);
    });
});
