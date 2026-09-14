import path from "path";
import { beforeEach, describe, expect, it } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { OperationalStore } from "@/runtime/storage/OperationalStore";

describe("workflow tools", () => {
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
                    `workflows-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
    });

    it("loads steps without executing nested mutations outside the runtime", async () => {
        const context = {
            guild: { id: "guild" } as any, actorId: "owner", currentChannelId: crypto.randomUUID(),
            question: "create and run a workflow",
        };
        await CapabilityRegistry.get("workflow_create").run(context, {
            name: "setup",
            description: "Create a channel",
            steps: [{ tool: "create_channel", args: { name: "unsafe" } }],
        });

        const result = await CapabilityRegistry.get("workflow_run").run(context, {
            name: "setup",
        });

        expect(result.data).toMatchObject({
            name: "setup",
            executionRequired: true,
            steps: [{
                index: 0,
                tool: "create_channel",
                args: { name: "unsafe" },
                label: null,
            }],
        });
    });

    it("deletes a saved workflow and reports missing names", async () => {
        const context = {
            guild: { id: "guild" } as any, actorId: "owner", currentChannelId: crypto.randomUUID(),
            question: "delete a workflow",
        };
        await CapabilityRegistry.get("workflow_create").run(context, {
            name: "demo-resumo",
            description: "Temporary demo workflow",
            steps: [{ tool: "get_guild_context", args: {} }],
        });

        const deleted = await CapabilityRegistry.get("workflow_delete").run(context, {
            name: "demo-resumo",
        });
        expect(deleted.data).toMatchObject({ name: "demo-resumo", removed: true });

        const listed = await CapabilityRegistry.get("workflow_list").run(context, {});
        expect(listed.data).toEqual({ workflows: [] });

        const missing = await CapabilityRegistry.get("workflow_delete").run(context, {
            name: "demo-resumo",
        });
        expect(missing.data).toEqual({ name: "demo-resumo", removed: false });
    });
});
