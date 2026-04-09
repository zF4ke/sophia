import { describe, expect, it } from "vitest";
import { assessGrounding, collectMemberEvidence } from "@/agent/orchestration/grounding";

describe("grounding", () => {
    it("treats live guild context as sufficient for current-server questions", () => {
        const grounding = assessGrounding("que servidor é esse?", [
            {
                tool: "get_guild_context",
                summary: "Oz Synthesis: 17 membros e 68 canais.",
                data: {
                    id: "g1",
                    name: "Oz Synthesis",
                    memberCount: 17,
                    channelCount: 68,
                },
            },
        ]);

        expect(grounding.summary).toEqual({
            messageEvidenceCount: 0,
            liveEvidenceCount: 1,
            sufficient: true,
        });
        expect(grounding.citations).toEqual([]);
        expect(grounding.evidence).toContain("Server name: Oz Synthesis");
    });

    it("aggregates paged member evidence in join order", () => {
        const evidence = collectMemberEvidence([
            {
                tool: "list_members",
                summary: "page 2",
                data: {
                    members: [
                        {
                            id: "u3",
                            username: "user3",
                            displayName: "User 3",
                            joinedTimestamp: 3,
                        },
                    ],
                    totalCount: 3,
                    returnedCount: 1,
                    hasMore: false,
                    offset: 2,
                    limit: 2,
                    sort: "joined_at",
                    filters: null,
                },
            },
            {
                tool: "list_members",
                summary: "page 1",
                data: {
                    members: [
                        {
                            id: "u2",
                            username: "user2",
                            displayName: "User 2",
                            joinedTimestamp: 2,
                        },
                        {
                            id: "u1",
                            username: "user1",
                            displayName: "User 1",
                            joinedTimestamp: 1,
                        },
                    ],
                    totalCount: 3,
                    returnedCount: 2,
                    hasMore: true,
                    offset: 0,
                    limit: 2,
                    sort: "joined_at",
                    filters: null,
                },
            },
        ]);

        expect(evidence?.members.map((member) => member.id)).toEqual(["u1", "u2", "u3"]);
        expect(evidence?.hasMore).toBe(false);
        expect(evidence?.totalCount).toBe(3);
    });

    it("treats post-crawl message hits as sufficient for non-member questions", () => {
        const grounding = assessGrounding("pega as musicas do canal scart", [
            {
                tool: "crawl_channel_messages",
                summary: "Fetched 120 messages from scart and stored 120.",
                data: {
                    channelId: "c-scart",
                    channelName: "scart",
                    messagesFetched: 120,
                    messagesStored: 120,
                    exhausted: false,
                    queryHint: "pega as musicas do canal scart",
                },
            },
            {
                tool: "search_messages",
                summary: "Found 8 relevant message chunks.",
                data: [
                    {
                        messageId: "m1",
                        channelId: "c-scart",
                        channelName: "scart",
                        guildId: "g1",
                        authorId: "u1",
                        authorName: "F4zke",
                        content: "1. Musica A\n2. Musica B",
                        createdTimestamp: 1715874381035,
                        jumpLink: "https://discord.com/channels/g1/c-scart/m1",
                        lexicalScore: 0.48,
                        semanticScore: 0.42,
                        recencyScore: 0.04,
                        totalScore: 0.41,
                    },
                ],
            },
        ] as any);

        expect(grounding.summary).toEqual({
            messageEvidenceCount: 1,
            liveEvidenceCount: 0,
            sufficient: true,
        });
        expect(grounding.citations).toHaveLength(1);
        expect(grounding.evidence).toContain("[scart] F4zke: 1. Musica A");
    });
});
