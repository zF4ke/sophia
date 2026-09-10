import path from "path";
import { beforeEach, describe, expect, it } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { ArtifactStore } from "@/discord/artifacts/ArtifactStore";

describe("artifact_edit tool", () => {
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
                    `artifacts-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
    });

    it("rejects edits that change nothing", async () => {
        const context = { guild: { id: "g1" } as never, question: "edit card" };
        const result = await CapabilityRegistry.get("artifact_edit").run(context, {
            message_id: "m1",
        } as never);

        expect(result.errorMessage).toMatch(/nothing to change/i);
    });

    it("reports unknown message ids", async () => {
        const context = { guild: { id: "g1" } as never, question: "edit card" };
        const result = await CapabilityRegistry.get("artifact_edit").run(context, {
            message_id: "ghost",
            title: "New title",
        } as never);

        expect(result.errorMessage).toMatch(/no artifact card found/i);
    });

    it("blocks cards from other guilds", async () => {
        await ArtifactStore.record({
            messageId: "m2",
            channelId: "c2",
            guildId: "other-guild",
            expiresAt: null,
            specJson: JSON.stringify({ title: "Alheio", sections: [{ body: "x" }] }),
        });

        const context = { guild: { id: "g1" } as never, question: "edit card" };
        const result = await CapabilityRegistry.get("artifact_edit").run(context, {
            message_id: "m2",
            title: "Roubo",
        } as never);

        expect(result.errorMessage).toMatch(/another server/i);
    });

    it("validates the merged spec before touching Discord", async () => {
        await ArtifactStore.record({
            messageId: "m3",
            channelId: "c3",
            guildId: "g1",
            expiresAt: null,
            specJson: JSON.stringify({ title: "Card", sections: [{ body: "x" }] }),
        });

        const context = { guild: { id: "g1" } as never, question: "edit card" };
        const result = await CapabilityRegistry.get("artifact_edit").run(context, {
            message_id: "m3",
            sections: [],
        } as never);

        expect(result.errorMessage).toMatch(/section/i);
    });
});
