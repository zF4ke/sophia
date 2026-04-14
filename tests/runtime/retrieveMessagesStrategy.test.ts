import { describe, expect, it } from "vitest";
import { retrieveMessagesTool } from "@/tools/retrieveMessages";
import type { DiscordToolResult } from "@/shared/appTypes";

const retrieveMessagesStrategy = retrieveMessagesTool.strategy;

function makeRetrieveRun(query: string, historyMessages: Array<Record<string, unknown>>): DiscordToolResult {
    return {
        tool: "retrieve_messages",
        summary: "ordered history evidence; 2 history and 0 semantic result(s) after refreshing Discord history.",
        data: {
            query,
            mode: "history",
            sourceOrigin: "cache_after_refresh",
            historyMessages,
            semanticMatches: [],
        },
    } as DiscordToolResult;
}

describe("retrieveMessagesStrategy", () => {
    it("keeps history rows unfiltered even when the query is specific", () => {
        const run = makeRetrieveRun("música", [
            {
                messageId: "m1",
                authorId: "u1",
                authorName: "Openrosen",
                channelId: "c1",
                channelName: "comandos",
                content: "MEU DEUS",
                jumpLink: "https://discord.com/channels/g1/c1/m1",
                createdTimestamp: 1,
            },
            {
                messageId: "m2",
                authorId: "u1",
                authorName: "Openrosen",
                channelId: "c1",
                channelName: "comandos",
                content: "Essa musica e boa.",
                jumpLink: "https://discord.com/channels/g1/c1/m2",
                createdTimestamp: 2,
            },
        ]);

        const evidence = retrieveMessagesStrategy.extractEvidence(run);

        expect(evidence).toHaveLength(2);
        expect(evidence.map((item) => item.content)).toEqual([
            "MEU DEUS",
            "Essa musica e boa.",
        ]);
    });

    it("still emits semantic evidence separately", () => {
        const run: DiscordToolResult = {
            tool: "retrieve_messages",
            summary: "ordered history evidence; 1 history and 1 semantic result(s) after refreshing Discord history.",
            data: {
                query: "m4rkim",
                mode: "history",
                sourceOrigin: "cache_after_refresh",
                historyMessages: [
                    {
                        messageId: "m1",
                        authorId: "u1",
                        authorName: "Openrosen",
                        channelId: "c1",
                        channelName: "comandos",
                        content: "A qualidade audiovisual desse vídeo é bizarra.",
                        jumpLink: "https://discord.com/channels/g1/c1/m1",
                        createdTimestamp: 1,
                    },
                ],
                semanticMatches: [
                    {
                        messageId: "m2",
                        authorId: "u1",
                        authorName: "Openrosen",
                        channelId: "c1",
                        channelName: "comandos",
                        content: "Acho surreal a capacidade do M4rkim.",
                        jumpLink: "https://discord.com/channels/g1/c1/m2",
                        createdTimestamp: 2,
                        lexicalScore: 2,
                    },
                ],
            },
        } as DiscordToolResult;

        const evidence = retrieveMessagesStrategy.extractEvidence(run);

        expect(evidence).toHaveLength(2);
        expect(evidence[0]?.evidenceRole).toBe("history_evidence");
        expect(evidence[1]?.evidenceRole).toBe("semantic_evidence");
        expect(evidence[1]?.strength).toBe("strong");
    });

    it("keeps broad history queries like * unfiltered", () => {
        const run = makeRetrieveRun("*", [
            {
                messageId: "m1",
                authorId: "u1",
                authorName: "Openrosen",
                channelId: "c1",
                channelName: "comandos",
                content: "MEU DEUS",
                jumpLink: "https://discord.com/channels/g1/c1/m1",
                createdTimestamp: 1,
            },
        ]);

        const evidence = retrieveMessagesStrategy.extractEvidence(run);

        expect(evidence).toHaveLength(1);
        expect(evidence[0]?.content).toBe("MEU DEUS");
    });
});