import { it, expect } from "vitest";
import { randomUUID } from "node:crypto";
import { AppPaths } from "@/app/AppPaths";
import path from "node:path";
import { TaskStore } from "@/runtime/tasks/TaskStore";
import { ExecutionControl } from "@/runtime/ExecutionControl";

it("limits named collaborators to steering a public task and rechecks removal before persisting", async () => {
    const store = new TaskStore(path.join(AppPaths.storageRoot, `collaborators-${randomUUID()}.sqlite`));
    const control = new ExecutionControl("owner", "c1");
    const release = control.register();
    try {
        const id = await store.create({ actorId: "owner", guildId: "g1", channelId: "c1", conversationId: "conversation", objective: "Research" });
        const privateId = await store.create({ actorId: "owner", guildId: "g1", channelId: "c1", conversationId: "private", objective: "Private", privateResponse: true });
        await expect(store.setCollaborator(id, "stranger", "c1", "g1", "friend")).rejects.toThrow();
        await expect(store.setCollaborator(privateId, "owner", "c1", "g1", "friend")).rejects.toThrow();
        await store.setCollaborator(id, "owner", "c1", "g1", "friend");
        expect(await store.canCollaborate(id, "friend", "c2", "g1")).toBe(false);
        expect(await store.snapshot(id, "friend", "c1", "g1")).toBeNull();
        control.bindTask(id, (text, contributor) => store.appendSteering(id, "owner", "c1", "g1", text, contributor));
        const allowed = () => store.canCollaborate(id, "friend", "c1", "g1");
        expect(await ExecutionControl.steerAsCollaborator("friend", "c1", "Compare the original sources", id, allowed)).toBe("queued");
        expect(control.requiresOwnerApproval).toBe(true);
        const restored = new ExecutionControl("owner", "c1");
        restored.restoreSteering(await store.steering(id, "owner", "c1", "g1"));
        expect(restored.requiresOwnerApproval).toBe(true);
        expect(restored.steering[0]).toContain("[Collaborator friend]");
        await store.setCollaborator(id, "owner", "c1", "g1", "friend", true);
        expect(await ExecutionControl.steerAsCollaborator("friend", "c1", "Another correction", id, allowed)).toBe("not_found");
        await expect(store.appendSteering(id, "owner", "c1", "g1", "Late correction", "friend")).rejects.toThrow();
        await store.finish(id, "owner", "paused", "Paused");
        await store.setCollaborator(id, "owner", "c1", "g1", "friend");
        expect(await store.canCollaborate(id, "friend", "c1", "g1")).toBe(false);
    } finally { release(); await store.close(); }
});
