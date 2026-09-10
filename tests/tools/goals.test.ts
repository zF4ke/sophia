import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

describe("goal tools", () => {
    beforeEach(async () => {
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        await OperationalStore.reset();
        SettingsService.update({
            runtime: {
                ...SettingsService.load().runtime,
                operationalDbPath: path.join(
                    process.cwd(),
                    "storage",
                    "test-runtime-store",
                    `goals-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
    });

    it("opens, moves, and closes goals through tool calls", async () => {
        const context = {
            guild: { id: "guild" } as any,
            question: "run a newsletter",
            requestId: "req-1",
            threadId: "thread-1",
        };

        const opened = await CapabilityRegistry.get("goal_open").run(context, {
            body: "Run weekly newsletter",
            label: "newsletter",
        });
        expect(opened.errorMessage).toBeFalsy();
        expect(opened.data).toMatchObject({ seq: 1 });

        const updated = await CapabilityRegistry.get("goal_update").run(context, {
            seq: 1,
            status: "in_progress",
        });
        expect(updated.data).toMatchObject({ seq: 1, status: "in_progress" });

        const listed = await DiscordMemoryService.listRequestGoals({
            requestId: "req-1",
            threadId: "thread-1",
            includeThreadHistory: true,
        });
        expect(listed).toHaveLength(1);
        expect(listed[0]).toMatchObject({ body: "Run weekly newsletter", status: "in_progress" });

        const done = await CapabilityRegistry.get("goal_done").run(context, {
            seq: 1,
            note: "sent to #novidades",
        });
        expect(done.data).toMatchObject({ seq: 1, status: "done" });

        const after = await DiscordMemoryService.listRequestGoals({
            requestId: "req-1",
            threadId: "thread-1",
            includeThreadHistory: true,
        });
        expect(after[0].status).toBe("done");
        expect(after[0].body).toContain("sent to #novidades");
    });

    it("rejects unknown seqs and enforces one done state per goal", async () => {
        const context = {
            guild: { id: "guild" } as any,
            question: "cleanup",
            requestId: "req-2",
            threadId: "thread-2",
        };

        const missing = await CapabilityRegistry.get("goal_update").run(context, {
            seq: 99,
            status: "done",
        });
        expect(missing.errorMessage).toBe("not_found");

        await CapabilityRegistry.get("goal_open").run(context, { body: "Tidy #media" });
        const done = await CapabilityRegistry.get("goal_done").run(context, { seq: 1 });
        expect(done.errorMessage).toBeFalsy();
    });
});
