import path from "path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SettingsService } from "@/app/SettingsService";
import { OperationalStore } from "@/runtime/storage/OperationalStore";
import { ArtifactStore } from "@/discord/artifacts/ArtifactStore";
import { handleArtifactInteraction } from "@/discord/artifacts/ArtifactInteractions";

const spec = {
    title: "Dossiê",
    sections: [
        { heading: "Um", body: "Primeira." },
        { heading: "Dois", body: "Segunda." },
        { heading: "Três", body: "Terceira." },
    ],
    navigation: { type: "pagination" as const },
};

function makeInteraction(customId: string, messageId: string, values?: string[]) {
    return {
        customId,
        message: { id: messageId },
        guildId: "g1",
        values,
        user: { id: "u1", username: "F4zke" },
        isStringSelectMenu: () => values !== undefined,
        isButton: () => values === undefined,
        editReply: vi.fn(async () => undefined),
        followUp: vi.fn(async () => undefined),
        deferUpdate: vi.fn(async () => undefined),
    } as never as Parameters<typeof handleArtifactInteraction>[0] & {
        editReply: ReturnType<typeof vi.fn>;
        followUp: ReturnType<typeof vi.fn>;
        deferUpdate: ReturnType<typeof vi.fn>;
    };
}

describe("artifact interactions", () => {
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
                    `artifact-interactions-${Date.now()}-${Math.random()}.sqlite`,
                ),
            },
        });
        await OperationalStore.initialize();
    });

    it("ignores non-artifact custom ids", async () => {
        const interaction = makeInteraction("settings:tab:model", "m1");
        expect(await handleArtifactInteraction(interaction)).toBe(false);
        expect(interaction.editReply).not.toHaveBeenCalled();
    });

    it("advances and rewinds pages from persisted state", async () => {
        await ArtifactStore.record({ messageId: "m1", channelId: "c1", guildId: "g1", expiresAt: null, specJson: JSON.stringify(spec) });

        const next = makeInteraction("artifact:a1b2c3d4e5f6:next", "m1");
        expect(await handleArtifactInteraction(next)).toBe(true);
        expect(next.editReply).toHaveBeenCalled();
        expect(await ArtifactStore.get("m1")).toMatchObject({ viewSection: 1 });

        const nextAgain = makeInteraction("artifact:a1b2c3d4e5f6:next", "m1");
        await handleArtifactInteraction(nextAgain);
        expect(await ArtifactStore.get("m1")).toMatchObject({ viewSection: 2 });

        const prev = makeInteraction("artifact:a1b2c3d4e5f6:prev", "m1");
        await handleArtifactInteraction(prev);
        expect(await ArtifactStore.get("m1")).toMatchObject({ viewSection: 1 });
    });

    it("clamps pagination bounds and handles tab selection", async () => {
        await ArtifactStore.record({ messageId: "m2", channelId: "c1", guildId: "g1", expiresAt: null, specJson: JSON.stringify(spec) });

        const nextPastEnd = makeInteraction("artifact:a1b2c3d4e5f6:next", "m2");
        await handleArtifactInteraction(nextPastEnd);
        await handleArtifactInteraction(makeInteraction("artifact:a1b2c3d4e5f6:next", "m2"));
        await handleArtifactInteraction(nextPastEnd);
        expect(await ArtifactStore.get("m2")).toMatchObject({ viewSection: 2 });

        const tab = makeInteraction("artifact:a1b2c3d4e5f6:tab", "m2", ["0"]);
        await handleArtifactInteraction(tab);
        expect(await ArtifactStore.get("m2")).toMatchObject({ viewSection: 0 });
    });


    it("handles legacy custom ids without the artifact: prefix", async () => {
        await ArtifactStore.record({ messageId: "m3", channelId: "c1", guildId: "g1", expiresAt: null, specJson: JSON.stringify(spec) });

        const interaction = makeInteraction("a1b2c3d4e5f6:next", "m3");
        expect(await handleArtifactInteraction(interaction)).toBe(true);
        expect(interaction.editReply).toHaveBeenCalled();
        expect(await ArtifactStore.get("m3")).toMatchObject({ viewSection: 1 });
    });

    it("gives every game: suffix its own state key so grids stay distinct", async () => {
        await ArtifactStore.record({ messageId: "m4", channelId: "c1", guildId: "g1", expiresAt: null, specJson: JSON.stringify(spec) });

        const box0 = makeInteraction("game:cell:0", "m4");
        await handleArtifactInteraction(box0);
        await handleArtifactInteraction(box0);

        const box1 = makeInteraction("game:cell:1", "m4");
        await handleArtifactInteraction(box1);

        const row = await ArtifactStore.get("m4");
        const gameState = JSON.parse(row!.gameStateJson ?? "{}") as Record<string, unknown>;
        expect(gameState["cell:0"]).toBe(2);
        expect(gameState["cell:1"]).toBe(1);
        expect(gameState["cell:0"]).not.toBe(gameState["cell:1"]);
        expect(box1.followUp).toHaveBeenCalled();
    });

    it("tells the user when the card lost its stored state", async () => {
        const interaction = makeInteraction("artifact:a1b2c3d4e5f6:next", "ghost");
        expect(await handleArtifactInteraction(interaction)).toBe(true);
        expect(interaction.followUp).toHaveBeenCalledWith(expect.objectContaining({ flags: expect.anything() }));
        expect(interaction.editReply).not.toHaveBeenCalled();
    });

    it("preserves every increment from simultaneous clicks", async () => {
        await ArtifactStore.record({ messageId: "concurrent", channelId: "c1", guildId: "g1", expiresAt: null, specJson: JSON.stringify(spec) });
        await Promise.all(Array.from({ length: 8 }, () => handleArtifactInteraction(makeInteraction("game:score", "concurrent"))));
        expect(JSON.parse((await ArtifactStore.get("concurrent"))!.gameStateJson!).score).toBe(8);
    });
});
