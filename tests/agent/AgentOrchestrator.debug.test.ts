import { beforeEach, describe, expect, it, vi } from "vitest";
import { AgentOrchestrator } from "@/agent/AgentOrchestrator";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import { RequestClassifier } from "@/agent/RequestClassifier";
import { ModelGateway } from "@/ai/ModelGateway";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";

describe("AgentOrchestrator debug reporting", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("reports direct-answer stages", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "direct_answer",
            reason: "general question",
        });
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("ok");

        const debugSession = {
            setClassifying: vi.fn().mockResolvedValue(undefined),
            setClassification: vi.fn().mockResolvedValue(undefined),
            setPlanning: vi.fn().mockResolvedValue(undefined),
            setToolRunning: vi.fn().mockResolvedValue(undefined),
            setToolResult: vi.fn().mockResolvedValue(undefined),
            setGroundingSummary: vi.fn().mockResolvedValue(undefined),
            setGenerating: vi.fn().mockResolvedValue(undefined),
            finishSuccess: vi.fn().mockResolvedValue(undefined),
            finishError: vi.fn().mockResolvedValue(undefined),
        };

        await AgentOrchestrator.answerQuestion({
            question: "O que é TCP?",
            user: { id: "u1" } as any,
            guild: null,
            currentChannelId: "c1",
            debugSession,
        });

        expect(debugSession.setClassifying).toHaveBeenCalledTimes(1);
        expect(debugSession.setClassification).toHaveBeenCalledWith("direct_answer");
        expect(debugSession.setGenerating).toHaveBeenCalledTimes(1);
        expect(debugSession.finishSuccess).toHaveBeenCalledWith(
            "Resposta direta concluída."
        );
        expect(debugSession.setToolRunning).not.toHaveBeenCalled();
    });

    it("reports grounded tool steps and insufficient evidence", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord memory",
        });
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({
                action: "search_messages",
                arguments: {
                    query: "roadmap",
                    limit: 8,
                },
                reason: "search first",
            } as any)
            .mockResolvedValueOnce({
                action: "finish",
                arguments: {},
                reason: "no relevant channels",
            } as any)
            .mockResolvedValueOnce({
                action: "finish",
                arguments: {},
                reason: "stop after one pass",
            } as any);
        vi.spyOn(DiscordToolService, "searchMessages").mockResolvedValue({
            tool: "search_messages",
            summary: "Found 1 relevant message chunks.",
            data: [
                {
                    totalScore: 0.1,
                    channelName: "produto",
                    authorName: "Ana",
                    jumpLink: "https://example.com",
                    content: "test",
                },
            ],
        } as any);
        vi.spyOn(DiscordToolService, "listRelevantChannels").mockResolvedValue({
            tool: "list_relevant_channels",
            summary: "No relevant channels found in local memory.",
            data: [],
        } as any);

        const debugSession = {
            setClassifying: vi.fn().mockResolvedValue(undefined),
            setClassification: vi.fn().mockResolvedValue(undefined),
            setPlanning: vi.fn().mockResolvedValue(undefined),
            setToolRunning: vi.fn().mockResolvedValue(undefined),
            setToolResult: vi.fn().mockResolvedValue(undefined),
            setGroundingSummary: vi.fn().mockResolvedValue(undefined),
            setGenerating: vi.fn().mockResolvedValue(undefined),
            finishSuccess: vi.fn().mockResolvedValue(undefined),
            finishError: vi.fn().mockResolvedValue(undefined),
        };

        await AgentOrchestrator.answerQuestion({
            question: "Qual foi a decisão do roadmap?",
            user: { id: "u1" } as any,
            guild: null,
            currentChannelId: "c1",
            debugSession,
        });

        expect(debugSession.setPlanning).toHaveBeenCalledWith(1);
        expect(debugSession.setToolRunning).toHaveBeenNthCalledWith(1,
            "search_messages",
            [
                "Onde: memória local disponível",
                'Busca: "Qual foi a decisão do roadmap?"',
                "Limite: até 8 resultados",
            ]
        );
        expect(debugSession.setToolResult).toHaveBeenCalledWith(
            "search_messages",
            "Found 1 relevant message chunks.",
            1
        );
        expect(debugSession.setGroundingSummary).toHaveBeenCalledWith({
            messageEvidenceCount: 0,
            liveEvidenceCount: 0,
            sufficient: false,
        });
        expect(debugSession.finishSuccess).toHaveBeenCalledWith(
            "Concluído sem evidência suficiente."
        );
        expect(debugSession.setGenerating).not.toHaveBeenCalled();
    });

    it("accepts live guild context as sufficient grounding", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord context",
        });
        vi.spyOn(ModelGateway, "generateJson");
        vi.spyOn(DiscordToolService, "searchMessages").mockResolvedValue({
            tool: "search_messages",
            summary: "No relevant stored messages found.",
            data: [],
        } as any);
        vi.spyOn(DiscordToolService, "listRelevantChannels").mockResolvedValue({
            tool: "list_relevant_channels",
            summary: "No relevant channels found in local memory.",
            data: [],
        } as any);
        vi.spyOn(DiscordToolService, "getGuildContext").mockResolvedValue({
            tool: "get_guild_context",
            summary: "Oz Synthesis: 17 membros e 68 canais.",
            data: {
                id: "g1",
                name: "Oz Synthesis",
                memberCount: 17,
                channelCount: 68,
            },
        } as any);
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Este servidor é o Oz Synthesis.");

        const debugSession = {
            setClassifying: vi.fn().mockResolvedValue(undefined),
            setClassification: vi.fn().mockResolvedValue(undefined),
            setPlanning: vi.fn().mockResolvedValue(undefined),
            setToolRunning: vi.fn().mockResolvedValue(undefined),
            setToolResult: vi.fn().mockResolvedValue(undefined),
            setGroundingSummary: vi.fn().mockResolvedValue(undefined),
            setGenerating: vi.fn().mockResolvedValue(undefined),
            finishSuccess: vi.fn().mockResolvedValue(undefined),
            finishError: vi.fn().mockResolvedValue(undefined),
        };

        const result = await AgentOrchestrator.answerQuestion({
            question: "Que servidor é esse?",
            user: { id: "u1" } as any,
            guild: { id: "g1", name: "Oz Synthesis" } as any,
            currentChannelId: "c1",
            debugSession,
        });

        expect(result.answer).toBe("Este servidor é o Oz Synthesis.");
        expect(result.citations).toEqual([]);
        expect(debugSession.setToolRunning).toHaveBeenNthCalledWith(1,
            "get_guild_context",
            ["Origem: metadados do servidor atual"]
        );
        expect(debugSession.setGroundingSummary).toHaveBeenCalledWith({
            messageEvidenceCount: 0,
            liveEvidenceCount: 1,
            sufficient: true,
        });
        expect(debugSession.setGenerating).toHaveBeenCalledTimes(1);
        expect(debugSession.finishSuccess).toHaveBeenCalledWith("Resposta concluída.");
    });

    it("uses a fallback answer when the model returns empty output", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "direct_answer",
            reason: "general question",
        });
        vi.spyOn(ModelGateway, "generateText").mockRejectedValue(
            new EmptyModelOutputError("direct_answer_generation")
        );

        const debugSession = {
            setClassifying: vi.fn().mockResolvedValue(undefined),
            setClassification: vi.fn().mockResolvedValue(undefined),
            setPlanning: vi.fn().mockResolvedValue(undefined),
            setToolRunning: vi.fn().mockResolvedValue(undefined),
            setToolResult: vi.fn().mockResolvedValue(undefined),
            setGroundingSummary: vi.fn().mockResolvedValue(undefined),
            setGenerating: vi.fn().mockResolvedValue(undefined),
            finishSuccess: vi.fn().mockResolvedValue(undefined),
            finishError: vi.fn().mockResolvedValue(undefined),
        };

        const result = await AgentOrchestrator.answerQuestion({
            question: "teste",
            user: { id: "u1" } as any,
            guild: null,
            currentChannelId: "c1",
            debugSession,
        });

        expect(result.answer).toContain("Não consegui gerar uma resposta válida");
        expect(debugSession.finishSuccess).toHaveBeenCalledWith(
            "Modelo devolveu resposta vazia; usei fallback."
        );
        expect(debugSession.finishError).not.toHaveBeenCalled();
    });

    it("keeps paging live members until an ordinal request is covered", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs member lookup",
        });
        const generateJsonSpy = vi.spyOn(ModelGateway, "generateJson");
        vi.spyOn(DiscordToolService, "getGuildContext").mockResolvedValue({
            tool: "get_guild_context",
            summary: "Oz Synthesis: 17 membros e 68 canais.",
            data: {
                id: "g1",
                name: "Oz Synthesis",
                memberCount: 17,
                channelCount: 68,
            },
        } as any);
        const listMembersSpy = vi
            .spyOn(DiscordToolService, "listMembers")
            .mockResolvedValueOnce({
                tool: "list_members",
                summary: "Mostrando 10 de 17 membros em ordem de entrada.",
                data: {
                    members: Array.from({ length: 10 }, (_, index) => ({
                        id: `u${index + 1}`,
                        username: `user${index + 1}`,
                        displayName: `User ${index + 1}`,
                        joinedTimestamp: index + 1,
                    })),
                    totalCount: 17,
                    returnedCount: 10,
                    hasMore: true,
                    offset: 0,
                    limit: 10,
                    sort: "joined_at",
                    filters: null,
                },
            } as any)
            .mockResolvedValueOnce({
                tool: "list_members",
                summary: "7 membros listados em ordem de entrada.",
                data: {
                    members: Array.from({ length: 7 }, (_, index) => ({
                        id: `u${index + 11}`,
                        username: `user${index + 11}`,
                        displayName: `User ${index + 11}`,
                        joinedTimestamp: index + 11,
                    })),
                    totalCount: 17,
                    returnedCount: 7,
                    hasMore: false,
                    offset: 10,
                    limit: 10,
                    sort: "joined_at",
                    filters: null,
                },
            } as any);
        const generateTextSpy = vi
            .spyOn(ModelGateway, "generateText")
            .mockResolvedValue("O 17º membro é User 17.");

        const result = await AgentOrchestrator.answerQuestion({
            question: "Quem é o décimo sétimo membro?",
            user: { id: "u1" } as any,
            guild: { id: "g1", name: "Oz Synthesis" } as any,
            currentChannelId: "c1",
        });

        expect(result.answer).toBe("O 17º membro é User 17.");
        expect(generateJsonSpy).not.toHaveBeenCalled();
        expect(listMembersSpy).toHaveBeenNthCalledWith(1, expect.anything(), {
            filters: undefined,
            limit: 100,
            offset: undefined,
            sort: "joined_at",
        });
        expect(listMembersSpy).toHaveBeenNthCalledWith(2, expect.anything(), {
            filters: undefined,
            limit: 100,
            offset: 10,
            sort: "joined_at",
        });
        expect(generateTextSpy).toHaveBeenCalledWith(
            expect.any(Array),
            expect.objectContaining({
                traceContext: expect.objectContaining({
                    traceLabel: "grounded_answer_generation",
                }),
            })
        );
    });

    it("falls back to live crawl when local memory search misses", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        const searchMessagesSpy = vi
            .spyOn(DiscordToolService, "searchMessages")
            .mockResolvedValueOnce({
                tool: "search_messages",
                summary: "No relevant stored messages found.",
                data: [],
            } as any)
            .mockResolvedValueOnce({
                tool: "search_messages",
                summary: "Found 1 relevant message chunks.",
                data: [
                    {
                        totalScore: 0.66,
                        channelName: "scart",
                        authorName: "scart",
                        jumpLink: "https://example.com/1",
                        content: "Escolhe a track X",
                    },
                ],
            } as any);
        vi.spyOn(DiscordToolService, "listRelevantChannels").mockResolvedValue({
            tool: "list_relevant_channels",
            summary: "Found 2 potentially relevant channels.",
            data: [
                {
                    channelId: "c-scart",
                    channelName: "scart",
                    hitCount: 0,
                    isIndexed: false,
                    matchSource: "live_name",
                    lastIndexedTimestamp: null,
                },
            ],
        } as any);
        const crawlSpy = vi.spyOn(DiscordToolService, "crawlChannelMessages").mockResolvedValue({
            tool: "crawl_channel_messages",
            summary: "Fetched 120 messages from scart and stored 120.",
            data: {
                channelId: "c-scart",
                channelName: "scart",
                messagesFetched: 120,
                messagesStored: 120,
                exhausted: false,
                queryHint: "musica do scart",
            },
        } as any);
        const generateTextSpy = vi
            .spyOn(ModelGateway, "generateText")
            .mockResolvedValue("Escolhe a música X do canal scart.");

        const result = await AgentOrchestrator.answerQuestion({
            question: "escolha uma musica legal no canal scart",
            user: { id: "u1" } as any,
            guild: { id: "g1", name: "Oz Synthesis" } as any,
            currentChannelId: "c-now",
        });

        expect(result.answer).toBe("Escolhe a música X do canal scart.");
        expect(searchMessagesSpy).toHaveBeenCalledTimes(2);
        expect(crawlSpy).toHaveBeenCalledWith(
            expect.anything(),
            "c-scart",
            1000,
            "escolha uma musica legal no canal scart"
        );
        expect(generateTextSpy).toHaveBeenCalledTimes(1);
    });

    it("prioritizes a mentioned channel before member lookup", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        const searchMessagesSpy = vi
            .spyOn(DiscordToolService, "searchMessages")
            .mockResolvedValueOnce({
                tool: "search_messages",
                summary: "No relevant stored messages found.",
                data: [],
            } as any)
            .mockResolvedValueOnce({
                tool: "search_messages",
                summary: "Found 1 relevant message chunks.",
                data: [
                    {
                        totalScore: 0.72,
                        channelName: "scart",
                        authorName: "Ana",
                        jumpLink: "https://example.com/track",
                        content: "A melhor track aqui é X.",
                    },
                ],
            } as any);
        const crawlSpy = vi.spyOn(DiscordToolService, "crawlChannelMessages").mockResolvedValue({
            tool: "crawl_channel_messages",
            summary: "Fetched 40 messages from scart and stored 40.",
            data: {
                channelId: "12345",
                channelName: "scart",
                messagesFetched: 40,
                messagesStored: 40,
                exhausted: false,
                queryHint: "escolha uma musica legal ai do <#12345>",
            },
        } as any);
        const getMemberProfileSpy = vi.spyOn(DiscordToolService, "getMemberProfile");
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Escolhe a música X.");

        const result = await AgentOrchestrator.answerQuestion({
            question: "escolha uma musica legal ai do <#12345>",
            user: { id: "u1" } as any,
            guild: { id: "g1", name: "Oz Synthesis" } as any,
            currentChannelId: "c-now",
        });

        expect(result.answer).toBe("Escolhe a música X.");
        expect(searchMessagesSpy).toHaveBeenNthCalledWith(
            1,
            "escolha uma musica legal ai do <#12345>",
            expect.anything(),
            expect.objectContaining({
                channelIds: ["12345"],
            })
        );
        expect(crawlSpy).toHaveBeenCalledWith(
            expect.anything(),
            "12345",
            1000,
            "escolha uma musica legal ai do <#12345>"
        );
        expect(getMemberProfileSpy).not.toHaveBeenCalled();
    });
});
