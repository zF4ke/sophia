import { Collection, TextChannel } from "discord.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AgentOrchestrator } from "@/agent/AgentOrchestrator";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import { RequestClassifier } from "@/agent/RequestClassifier";
import { ModelGateway } from "@/ai/ModelGateway";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { MemoryDatabase } from "@/memory/MemoryDatabase";

function createReadableChannel(id: string, name: string) {
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

describe("AgentOrchestrator debug reporting", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
        process.env.SOPHIA_MEMORY_DB_PATH = ":memory:";
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";
        MemoryDatabase.reset();
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
                    limit: 30,
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
                "Limite: até 30 resultados",
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
        }, "heuristic", "insufficient");
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
        }, "heuristic", "confident");
        expect(debugSession.setGenerating).toHaveBeenCalledTimes(1);
        expect(debugSession.finishSuccess).toHaveBeenCalledWith("Resposta concluída.");
        expect(ModelGateway.generateJson).toHaveBeenCalled();
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
        expect(generateJsonSpy).toHaveBeenCalled();
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
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            intent: "channel_target",
            targetText: "scart",
            confidence: 0.9,
            reason: "This looks like a channel-focused request.",
        } as any);
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
            250,
            "scart",
            expect.any(Function)
        );
        expect(generateTextSpy).toHaveBeenCalledTimes(1);
    });

    it("forces a scoped indexed search before allowing the first crawl", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({
                questionIntent: "person_messages",
                nextAction: "crawl_channel_messages",
                targetText: "One Person",
                authorId: "u-open",
                authorQuery: "oneperson",
                topicText: "silksong",
                channelHintText: "silksong",
                channelIds: ["c-silk"],
                searchQuery: "silksong",
                needsMessageEvidence: true,
                answerConfidence: "best_effort",
                confidence: 0.91,
                reason: "Go straight to the channel crawl.",
            } as any)
            .mockResolvedValueOnce({
                questionIntent: "person_messages",
                nextAction: "crawl_channel_messages",
                targetText: "One Person",
                authorId: "u-open",
                authorQuery: "oneperson",
                topicText: "silksong",
                channelHintText: "silksong",
                channelIds: ["c-silk"],
                searchQuery: "silksong",
                needsMessageEvidence: true,
                answerConfidence: "best_effort",
                confidence: 0.91,
                reason: "Now crawl the hinted channel.",
            } as any)
            .mockResolvedValueOnce({
                questionIntent: "person_messages",
                nextAction: "answer",
                targetText: "One Person",
                authorId: "u-open",
                authorQuery: "oneperson",
                topicText: "silksong",
                channelHintText: "silksong",
                channelIds: ["c-silk"],
                searchQuery: "silksong",
                needsMessageEvidence: true,
                answerConfidence: "best_effort",
                confidence: 0.91,
                reason: "The evidence is enough to answer.",
            } as any);
        const searchSpy = vi
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
                        totalScore: 0.63,
                        channelId: "c-silk",
                        channelName: "silksong",
                        authorId: "u-open",
                        authorName: "Openrosen",
                        jumpLink: "https://example.com/silk",
                        content: "Silksong está incrível.",
                    },
                ],
            } as any);
        const crawlSpy = vi
            .spyOn(DiscordToolService, "crawlChannelMessages")
            .mockResolvedValue({
                tool: "crawl_channel_messages",
                summary: "Fetched 40 messages from silksong; queued 40 for background indexing.",
                data: {
                    channelId: "c-silk",
                    channelName: "silksong",
                    messagesFetched: 40,
                    messagesStored: 40,
                    exhausted: false,
                    queryHint: "silksong",
                    backgroundIngestQueued: true,
                    previewMessages: [],
                },
            } as any);
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "One Person disse que Silksong está incrível."
        );

        const result = await AgentOrchestrator.answerQuestion({
            question: "me conte a experencia de One Person jogando silksong baseado no que ele falou no canal de silksong",
            user: { id: "u1" } as any,
            guild: {
                id: "g1",
                name: "Oz Synthesis",
                channels: {
                    cache: new Collection([
                        ["c-silk", createReadableChannel("c-silk", "silksong")],
                    ]),
                },
            } as any,
            currentChannelId: "c-now",
        });

        expect(result.answer).toContain("Silksong");
        expect(searchSpy).toHaveBeenNthCalledWith(
            1,
            "silksong",
            expect.anything(),
            expect.objectContaining({
                authorId: "u-open",
                channelIds: ["c-silk"],
            })
        );
        expect(searchSpy).toHaveBeenCalled();
        if (crawlSpy.mock.calls.length > 0) {
            expect(searchSpy.mock.invocationCallOrder[0]).toBeLessThan(
                crawlSpy.mock.invocationCallOrder[0]
            );
        }
    });

    it("accepts moderate post-crawl channel results as sufficient grounding", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        vi.spyOn(ModelGateway, "generateJson").mockResolvedValue({
            intent: "channel_target",
            targetText: "scart",
            confidence: 0.92,
            reason: "This looks like a channel-focused request.",
        } as any);
        vi.spyOn(DiscordToolService, "searchMessages")
            .mockResolvedValueOnce({
                tool: "search_messages",
                summary: "No relevant stored messages found.",
                data: [],
            } as any)
            .mockResolvedValueOnce({
                tool: "search_messages",
                summary: "Found 8 relevant message chunks.",
                data: [
                    {
                        totalScore: 0.41,
                        channelName: "scart",
                        authorName: "F4zke",
                        jumpLink: "https://example.com/1",
                        content: "1. A realidade se transforma\n2. O Caminho Inverso",
                    },
                ],
            } as any);
        vi.spyOn(DiscordToolService, "listRelevantChannels").mockResolvedValue({
            tool: "list_relevant_channels",
            summary: "Found 1 potentially relevant channels.",
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
        vi.spyOn(DiscordToolService, "crawlChannelMessages").mockResolvedValue({
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
        } as any);
        const generateTextSpy = vi
            .spyOn(ModelGateway, "generateText")
            .mockResolvedValue("As músicas do canal scart incluem A realidade se transforma e O Caminho Inverso.");

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
            setRouting: vi.fn().mockResolvedValue(undefined),
        };

        const result = await AgentOrchestrator.answerQuestion({
            question: "pega as musicas do canal scart",
            user: { id: "u1" } as any,
            guild: { id: "g1", name: "Oz Synthesis" } as any,
            currentChannelId: "c-now",
            debugSession,
        });

        expect(result.answer).toContain("A realidade se transforma");
        expect(debugSession.setGroundingSummary).toHaveBeenCalledWith({
            messageEvidenceCount: 1,
            liveEvidenceCount: 0,
            sufficient: true,
        }, "heuristic", "best_effort");
        expect(debugSession.setGenerating).toHaveBeenCalledTimes(1);
        expect(debugSession.finishSuccess).toHaveBeenCalledWith("Resposta concluída.");
        expect(generateTextSpy).toHaveBeenCalledTimes(1);
        expect(
            vi.mocked(ModelGateway.generateJson).mock.calls.some(
                ([, , options]) =>
                    options?.traceContext?.traceLabel === "discord_retrieval_controller"
            )
        ).toBe(true);
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
            250,
            "escolha uma musica legal ai do <#12345>",
            expect.any(Function)
        );
        expect(getMemberProfileSpy).not.toHaveBeenCalled();
    });

    it("reuses a strong guild-wide grounded context without new tool calls", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-old",
            channelScopeKey: "c-old",
            questionFingerprint: "que servidor e esse",
            routeIntent: "server_context",
            evidenceText: "Tool: get_guild_context\nSummary: Oz Synthesis: 17 membros e 68 canais.\nServer name: Oz Synthesis",
            citations: [],
            toolRuns: [
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
            ],
            sufficient: true,
            groundingDecisionMode: "judge",
            createdTimestamp: Date.now(),
            expiryTimestamp: Date.now() + 60_000,
            createdResponseOrdinal: 1,
        });
        const getGuildContextSpy = vi.spyOn(DiscordToolService, "getGuildContext");
        const generateTextSpy = vi
            .spyOn(ModelGateway, "generateText")
            .mockResolvedValue("Você está no servidor Oz Synthesis.");

        const debugSession = {
            setClassifying: vi.fn().mockResolvedValue(undefined),
            setClassification: vi.fn().mockResolvedValue(undefined),
            setRouting: vi.fn().mockResolvedValue(undefined),
            setContextCacheStatus: vi.fn().mockResolvedValue(undefined),
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
            currentChannelId: "c-now",
            debugSession,
        });

        expect(result.answer).toBe("Você está no servidor Oz Synthesis.");
        expect(getGuildContextSpy).not.toHaveBeenCalled();
        expect(debugSession.setContextCacheStatus).toHaveBeenCalledWith("reused");
        expect(debugSession.setGroundingSummary).toHaveBeenCalledWith(
            {
                messageEvidenceCount: 0,
                liveEvidenceCount: 1,
                sufficient: true,
            },
            "reused",
            "confident"
        );
        expect(generateTextSpy).toHaveBeenCalledTimes(1);
    });

    it("seeds the tool loop from a prior insufficient grounded context", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-old",
            channelScopeKey: "c-old",
            questionFingerprint: "roadmap release",
            routeIntent: "broad_search",
            evidenceText: "Tool: search_messages\nSummary: No relevant stored messages found.",
            citations: [],
            toolRuns: [
                {
                    tool: "search_messages",
                    summary: "No relevant stored messages found.",
                    data: [],
                },
            ],
            sufficient: false,
            groundingDecisionMode: "heuristic",
            createdTimestamp: Date.now(),
            expiryTimestamp: Date.now() + 60_000,
            createdResponseOrdinal: 1,
        });
        const searchMessagesSpy = vi
            .spyOn(DiscordToolService, "searchMessages")
            .mockResolvedValue({
                tool: "search_messages",
                summary: "Found 1 relevant message chunks.",
                data: [
                    {
                        totalScore: 0.52,
                        channelName: "produto",
                        authorName: "Ana",
                        jumpLink: "https://example.com/roadmap",
                        content: "Roadmap release confirmed.",
                    },
                ],
            } as any);
        const listRelevantChannelsSpy = vi
            .spyOn(DiscordToolService, "listRelevantChannels")
            .mockResolvedValue({
                tool: "list_relevant_channels",
                summary: "Found 1 potentially relevant channels.",
                data: [
                    {
                        channelId: "c-prod",
                        channelName: "produto",
                        hitCount: 0,
                        isIndexed: false,
                        matchSource: "live_name",
                        lastIndexedTimestamp: null,
                    },
                ],
            } as any);
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({
                action: "crawl_channel_messages",
                arguments: {
                    channelId: "c-prod",
                    limit: 1000,
                    queryHint: "roadmap release",
                },
                reason: "continue from discovered channel",
            } as any)
            .mockResolvedValueOnce({
                sufficient: false,
                reason: "Still not enough evidence.",
                missingInformation: "Need actual roadmap details.",
            } as any);
        vi.spyOn(DiscordToolService, "crawlChannelMessages").mockResolvedValue({
            tool: "crawl_channel_messages",
            summary: "Fetched 20 messages from produto and stored 20.",
            data: {
                channelId: "c-prod",
                channelName: "produto",
                messagesFetched: 20,
                messagesStored: 20,
                exhausted: false,
                queryHint: "roadmap release",
            },
        } as any);
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "Roadmap release confirmed."
        );

        await AgentOrchestrator.answerQuestion({
            question: "roadmap release",
            user: { id: "u1" } as any,
            guild: { id: "g1", name: "Oz Synthesis" } as any,
            currentChannelId: "c-now",
            debugSession: {
                setClassifying: vi.fn().mockResolvedValue(undefined),
                setClassification: vi.fn().mockResolvedValue(undefined),
                setRouting: vi.fn().mockResolvedValue(undefined),
                setContextCacheStatus: vi.fn().mockResolvedValue(undefined),
                setPlanning: vi.fn().mockResolvedValue(undefined),
                setToolRunning: vi.fn().mockResolvedValue(undefined),
                setToolResult: vi.fn().mockResolvedValue(undefined),
                setGroundingSummary: vi.fn().mockResolvedValue(undefined),
                setGenerating: vi.fn().mockResolvedValue(undefined),
                finishSuccess: vi.fn().mockResolvedValue(undefined),
                finishError: vi.fn().mockResolvedValue(undefined),
            },
        });

        expect(searchMessagesSpy).toHaveBeenCalledTimes(1);
        expect(listRelevantChannelsSpy).toHaveBeenCalledTimes(1);
    });

    it("reuses resolved person context for follow-up person-topic questions", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        DiscordMemoryService.saveConversationResolutionContext({
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
                        totalScore: 0.71,
                        channelName: "silksong",
                        authorName: "Openrosen",
                        jumpLink: "https://example.com/silksong",
                        content: "Silksong ainda vai sair.",
                    },
                ],
            } as any);
        const crawlSpy = vi
            .spyOn(DiscordToolService, "crawlChannelMessages")
            .mockResolvedValue({
                tool: "crawl_channel_messages",
                summary: "Fetched 40 messages from silksong and stored 40.",
                data: {
                    channelId: "c-silk",
                    channelName: "silksong",
                    messagesFetched: 40,
                    messagesStored: 40,
                    exhausted: false,
                    queryHint: "silksong",
                },
            } as any);
        const listMembersSpy = vi.spyOn(DiscordToolService, "listMembers");
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "Openrosen disse que Silksong ainda vai sair."
        );

        const result = await AgentOrchestrator.answerQuestion({
            question: "o que é que ele falou sobre silksong",
            user: { id: "u1" } as any,
            guild: {
                id: "g1",
                name: "Oz Synthesis",
                channels: {
                    cache: new Collection([
                        [
                            "c-silk",
                            createReadableChannel("c-silk", "silksong"),
                        ],
                    ]),
                },
            } as any,
            currentChannelId: "c-now",
        });

        expect(result.answer).toContain("Silksong");
        expect(listMembersSpy).not.toHaveBeenCalled();
        expect(searchMessagesSpy).toHaveBeenNthCalledWith(
            1,
            "silksong",
            expect.anything(),
            expect.objectContaining({
                authorId: "u-open",
                channelIds: ["c-silk"],
            })
        );
        expect(crawlSpy).toHaveBeenCalledWith(
            expect.anything(),
            "c-silk",
            250,
            "silksong",
            expect.any(Function)
        );
    });

    it("seeds follow-up runs from the most recent grounded context in the conversation", async () => {
        vi.spyOn(RequestClassifier, "classify").mockResolvedValue({
            mode: "discord_grounded",
            reason: "needs discord evidence",
        });
        DiscordMemoryService.saveConversationResolutionContext({
            guildId: "g1",
            channelId: "c-now",
            routeIntent: "person_target",
            targetText: "One Person",
            authorId: "u-open",
            authorQuery: "oneperson",
            channelIds: ["c-silk"],
            topicText: "silksong",
            channelHintText: "silksong",
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
        });
        DiscordMemoryService.saveReusableGroundedContext({
            guildId: "g1",
            channelId: "c-now",
            channelScopeKey: "c-now",
            questionFingerprint: "me conte a experiencia de one person jogando silksong baseado no que ele falou no canal de silksong",
            routeIntent: "person_target",
            evidenceText: "Tool: search_messages\nSummary: Found 2 relevant message chunks.",
            citations: [],
            toolRuns: [
                {
                    tool: "search_messages",
                    summary: "Found 2 relevant message chunks.",
                    data: [
                        {
                            totalScore: 0.72,
                            channelName: "silksong",
                            authorName: "Openrosen",
                            jumpLink: "https://example.com/1",
                            content: "A jogabilidade é uma obra-prima.",
                        },
                        {
                            totalScore: 0.64,
                            channelName: "silksong",
                            authorName: "Openrosen",
                            jumpLink: "https://example.com/2",
                            content: "O começo parece fácil para quem já ficou bom em Hollow Knight.",
                        },
                    ],
                },
            ],
            sufficient: true,
            groundingDecisionMode: "heuristic",
            createdTimestamp: Date.now(),
            expiryTimestamp: Date.now() + 60_000,
            createdResponseOrdinal: 1,
        });
        vi.spyOn(ModelGateway, "generateJson")
            .mockResolvedValueOnce({
                questionIntent: "person_messages",
                nextAction: "search_messages",
                targetText: "One Person",
                authorId: "u-open",
                authorQuery: "oneperson",
                topicText: "silksong",
                channelHintText: "silksong",
                channelIds: ["c-silk"],
                searchQuery: "silksong",
                needsMessageEvidence: true,
                answerConfidence: "best_effort",
                confidence: 0.9,
                reason: "Reuse the existing person/topic context.",
            } as any)
            .mockResolvedValue({
                questionIntent: "person_messages",
                nextAction: "answer",
                targetText: "One Person",
                authorId: "u-open",
                authorQuery: "oneperson",
                topicText: "silksong",
                channelHintText: "silksong",
                channelIds: ["c-silk"],
                searchQuery: "silksong",
                needsMessageEvidence: true,
                answerConfidence: "confident",
                confidence: 0.9,
                reason: "The seeded evidence already answers the follow-up.",
            } as any);
        const crawlSpy = vi.spyOn(DiscordToolService, "crawlChannelMessages");
        const searchSpy = vi.spyOn(DiscordToolService, "searchMessages");
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue(
            "Ele também disse que a jogabilidade é uma obra-prima e que o começo parece fácil para veteranos de Hollow Knight."
        );

        const debugSession = {
            setClassifying: vi.fn().mockResolvedValue(undefined),
            setClassification: vi.fn().mockResolvedValue(undefined),
            setRouting: vi.fn().mockResolvedValue(undefined),
            setContextCacheStatus: vi.fn().mockResolvedValue(undefined),
            setPlanning: vi.fn().mockResolvedValue(undefined),
            setToolRunning: vi.fn().mockResolvedValue(undefined),
            setToolResult: vi.fn().mockResolvedValue(undefined),
            setGroundingSummary: vi.fn().mockResolvedValue(undefined),
            setGenerating: vi.fn().mockResolvedValue(undefined),
            finishSuccess: vi.fn().mockResolvedValue(undefined),
            finishError: vi.fn().mockResolvedValue(undefined),
        };

        const result = await AgentOrchestrator.answerQuestion({
            question: "que mais ele disse?",
            user: { id: "u1" } as any,
            guild: {
                id: "g1",
                name: "Oz Synthesis",
                channels: {
                    cache: new Collection([
                        ["c-silk", createReadableChannel("c-silk", "silksong")],
                    ]),
                },
            } as any,
            currentChannelId: "c-now",
            debugSession,
        });

        expect(result.answer).toContain("obra-prima");
        expect(debugSession.setContextCacheStatus).toHaveBeenCalledWith("seeded");
        expect(searchSpy).not.toHaveBeenCalled();
        expect(crawlSpy).not.toHaveBeenCalled();
    });
});
