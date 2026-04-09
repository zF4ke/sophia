import { Collection, TextChannel } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway } from "@/ai/ModelGateway";
import { decideRetrievalAction } from "@/agent/orchestration/retrievalController";
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

describe("decideRetrievalAction", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("routes identity questions through the controller as person identity", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            questionIntent: "person_identity",
            nextAction: "get_member_profile",
            targetText: "openrose",
            authorQuery: "openrose",
            needsMessageEvidence: false,
            answerConfidence: "best_effort",
            confidence: 0.93,
            reason: "This is asking who that person is.",
        });

        const decision = await decideRetrievalAction({
            question: "quem é esse tal de openrose que continua aparecendo nos chats",
            guild: { id: "g1", channels: { cache: new Collection() } } as any,
            currentChannelId: "c1",
            toolRuns: [],
            context: {
                crawledChannelIds: new Set(),
                seededFromContext: false,
                initialToolRuns: [],
                latestControllerDecision: null,
            },
        });

        expect(decision).toMatchObject({
            source: "ai",
            questionIntent: "person_identity",
            routeIntent: "person_target",
            nextAction: "get_member_profile",
            targetText: "openrose",
            authorQuery: "openrose",
            answerConfidence: "best_effort",
        });
    });

    it("keeps person + topic + channel hints together for follow-up message questions", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            questionIntent: "person_messages",
            nextAction: "search_messages",
            targetText: "One Person",
            authorId: "u-open",
            authorQuery: "oneperson",
            topicText: "silksong",
            channelHintText: "silksong",
            searchQuery: "silksong",
            needsMessageEvidence: true,
            answerConfidence: "best_effort",
            confidence: 0.91,
            reason: "Search the person's messages in the hinted channel first.",
        });
        const priorContext: ConversationResolutionContext = {
            guildId: "g1",
            channelId: "c-now",
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

        const decision = await decideRetrievalAction({
            question: "o que ele falou sobre silksong",
            guild: {
                id: "g1",
                channels: {
                    cache: new Collection([
                        ["c-silk", createChannel("c-silk", "silksong")],
                    ]),
                },
            } as any,
            currentChannelId: "c-now",
            priorContext,
            toolRuns: [],
            context: {
                crawledChannelIds: new Set(),
                seededFromContext: false,
                initialToolRuns: [],
                latestControllerDecision: null,
            },
        });

        expect(decision).toMatchObject({
            source: "ai",
            questionIntent: "person_messages",
            routeIntent: "person_target",
            authorId: "u-open",
            authorQuery: "oneperson",
            topicText: "silksong",
            channelHintText: "silksong",
            channelIds: ["c-silk"],
            nextAction: "search_messages",
            searchQuery: "silksong",
        });
    });

    it("falls back to scoped channel search when the model is unavailable", async () => {
        vi.spyOn(ModelGateway, "generateJson").mockRejectedValue(new Error("offline"));

        const decision = await decideRetrievalAction({
            question: "escolha uma musica legal ai do <#12345>",
            guild: { id: "g1", channels: { cache: new Collection() } } as any,
            currentChannelId: "c-now",
            toolRuns: [],
            context: {
                crawledChannelIds: new Set(),
                seededFromContext: false,
                initialToolRuns: [],
                latestControllerDecision: null,
            },
        });

        expect(decision).toMatchObject({
            source: "deterministic",
            questionIntent: "channel_or_topic_search",
            routeIntent: "channel_target",
            nextAction: "search_messages",
            channelIds: ["12345"],
        });
    });
});
