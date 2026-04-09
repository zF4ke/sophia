import { Collection, TextChannel } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import { routeDiscordQuestion } from "@/agent/orchestration/router";
import type { ConversationResolutionContext } from "@/shared/appTypes";

function createChannel(id: string, name: string) {
    const channel = Object.create(TextChannel.prototype);
    Object.defineProperty(channel, "viewable", {
        value: true,
        configurable: true,
    });
    Object.assign(channel, {
        id,
        name,
    });
    return channel;
}

describe("routeDiscordQuestion", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("skips AI routing for explicit server questions", async () => {
        const generateJsonSpy = vi.spyOn(ModelGateway, "generateJson");

        const result = await routeDiscordQuestion({
            question: "que servidor é esse?",
            guild: { id: "g1" } as any,
            currentChannelId: "c1",
        });

        expect(result).toMatchObject({
            source: "deterministic",
            intent: "server_context",
        });
        expect(generateJsonSpy).not.toHaveBeenCalled();
    });

    it("routes ambiguous channel-style questions through AI and resolves channel ids", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            intent: "channel_target",
            targetText: "scart",
            confidence: 0.91,
            reason: "Looks like a channel target.",
        });
        const guild = {
            id: "g1",
            channels: {
                cache: new Collection([
                    ["c1", createChannel("c1", "scart")],
                    ["c2", createChannel("c2", "general")],
                ]),
            },
        } as any;

        const result = await routeDiscordQuestion({
            question: 'blz, tentando ser "neutra". escolha uma musica legal ai do scart',
            guild,
            currentChannelId: "c2",
        });

        expect(result).toMatchObject({
            source: "ai",
            intent: "channel_target",
            targetText: "scart",
            channelIds: ["c1"],
        });
    });

    it("routes profile-style questions to person target", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            intent: "person_target",
            targetText: "scart",
            confidence: 0.88,
            reason: "Asking about a person profile.",
        });

        const result = await routeDiscordQuestion({
            question: "qual é o perfil do scart?",
            guild: { id: "g1", channels: { cache: new Collection() } } as any,
            currentChannelId: "c1",
        });

        expect(result).toMatchObject({
            source: "ai",
            intent: "person_target",
            authorQuery: "scart",
            targetText: "scart",
        });
    });

    it("falls back to broad search when AI is uncertain", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            intent: "broad_search",
            targetText: "",
            confidence: 0.32,
            reason: "Not enough certainty.",
        });

        const result = await routeDiscordQuestion({
            question: "o que rolou com isso?",
            guild: { id: "g1", channels: { cache: new Collection() } } as any,
            currentChannelId: "c1",
        });

        expect(result).toMatchObject({
            source: "deterministic",
            intent: "broad_search",
        });
    });

    it("reuses a prior resolved person for referential follow-up questions", async () => {
        const guild = {
            id: "g1",
            channels: {
                cache: new Collection([
                    ["c1", createChannel("c1", "silksong")],
                    ["c2", createChannel("c2", "discussao")],
                ]),
            },
        } as any;
        const priorContext: ConversationResolutionContext = {
            guildId: "g1",
            channelId: "c2",
            routeIntent: "person_target",
            targetText: "One Person",
            authorId: "u-open",
            authorQuery: "oneperson",
            channelIds: [],
            topicText: null,
            channelHintText: null,
            resolvedPerson: {
                id: "u-open",
                username: "oneperson",
                displayName: "Openrosen",
                globalName: null,
                nickname: "One Person",
                roles: ["member"],
            },
            createdTimestamp: Date.now(),
            expiryTimestamp: Date.now() + 60_000,
            createdResponseOrdinal: 1,
        };

        const result = await routeDiscordQuestion({
            question: "o que é que ele falou sobre silksong",
            guild,
            currentChannelId: "c2",
            priorContext,
        });

        expect(result).toMatchObject({
            source: "deterministic",
            intent: "person_target",
            authorId: "u-open",
            authorQuery: "oneperson",
            topicText: "silksong",
            channelHintText: "silksong",
            channelIds: ["c1"],
        });
    });
});
