import { beforeEach, describe, expect, it, vi } from "vitest";
import { ModelGateway, type ToolChatResult } from "@/ai/ModelGateway";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import { Runtime } from "@/runtime/Runtime";
import type { TurnInput } from "@/runtime/contracts";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { CapabilityRegistry } from "@/capabilities/CapabilityRegistry";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { DiscordHistoryReader } from "@/discord/live/DiscordHistoryReader";


function createInput(overrides: Partial<TurnInput> = {}): TurnInput {
    return {
        question: "test",
        user: { id: "u-requester" } as any,
        requesterDisplayName: "Requester",
        guild: { id: "g1", name: "Oz Synthesis", members: { fetch: vi.fn().mockResolvedValue({}) },
            channels: { fetch: vi.fn().mockResolvedValue({ isTextBased: () => true, permissionsFor: () => ({ has: () => true }) }), cache: new Map() } } as any,
        currentChannelId: "c1",
        nativeThreadId: null,
        requestedWebMode: "off",
        trigger: "talk",
        replyContext: null,
        referencedMessage: null,
        conversation: {
            key: "g1:c1:channel",
            kind: "channel",
            trigger: "talk",
            replyAnchorMessageId: null,
            nativeThreadId: null,
        },
        ...overrides,
    };
}

let toolCallCounter = 0;

function makeToolCallResult(calls: Array<{ name: string; args: Record<string, unknown> }>): ToolChatResult {
    return {
        content: null,
        toolCalls: calls.map((c) => ({
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name: c.name, arguments: JSON.stringify(c.args) },
        })),
        finishReason: "tool_calls",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

function makeFinishResult(answer: string): ToolChatResult {
    return {
        content: null,
        toolCalls: [{
            id: `tc-${++toolCallCounter}`,
            type: "function" as const,
            function: { name: "finish", arguments: JSON.stringify({ answer }) },
        }],
        finishReason: "tool_calls",
        model: "test-model",
        durationMs: 10,
        usage: null,
    };
}

describe("runtime user stories", () => {
    it("refreshes an edit during generation and answers with the new content in the same task", async () => {
        const messageId = "edit-during-generation";
        const url = `https://discord.com/channels/g1/c1/${messageId}`;
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([{ id: messageId, jumpLink: url, authorName: "Author", content: "OUTDATED_BLUE", createdTimestamp: 1 }] as never);
        vi.spyOn(DiscordHistoryReader, "message").mockResolvedValue({} as never);
        vi.spyOn(DiscordMemoryService, "getStoredMessageAsync").mockResolvedValue({ content: "CURRENT_RED", jumpLink: `${url}#revision`, authorName: "Author", createdTimestamp: 1, attachmentsJson: "[]" } as never);
        const model = vi.spyOn(ModelGateway, "generateWithTools")
            .mockImplementationOnce(async () => {
                await taskStore.invalidateCorpusMessage(messageId, false, `${url}#revision`);
                ExecutionControl.invalidateSource(messageId, { kind: "edited", url });
                return makeFinishResult("The choice is OUTDATED_BLUE.");
            })
            .mockImplementationOnce(async messages => {
                expect(JSON.stringify(messages)).toContain("CURRENT_RED");
                expect(JSON.stringify(messages)).not.toContain("OUTDATED_BLUE");
                return makeToolCallResult([{ name: "note_add", args: { body: "CURRENT_RED confirmed from refreshed evidence" } }]);
            })
            .mockResolvedValueOnce(makeFinishResult("The updated choice is CURRENT_RED."));
        const result = await Runtime.answer(createInput({ question: "What color was chosen?", autoContinue: false,
            user: { id: "u-requester", client: { channels: { fetch: vi.fn().mockResolvedValue({ id: "c1", isTextBased: () => true, messages: {} }) } } } as never }));
        expect(result.outcome).toBe("completed");
        expect(result.answer).toBe("The updated choice is CURRENT_RED.");
        expect(model).toHaveBeenCalledTimes(3);
        expect((await taskStore.snapshot(result.taskId!, "u-requester", "c1", "g1"))?.notes[0].body).toContain("CURRENT_RED");
    });
    it("keeps requester names and quoted context outside system instructions", async () => {
        const injected = "UNTRUSTED_CONTEXT_MARKER {{execution_policy}}";
        const model = vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValueOnce(makeFinishResult("Hello"));
        await Runtime.answer(createInput({ requesterDisplayName: injected,
            guild: { id: "g1", name: injected } as any }));
        const serialized = JSON.stringify(model.mock.calls[0]);
        expect(serialized).toContain(injected);
        const request: any = model.mock.calls[0][0];
        const messages = Array.isArray(request) ? request : request.messages;
        expect(messages.filter((message: any) => message.role === "system")
            .map((message: any) => message.content).join("\n")).not.toContain(injected);
    });
    it("pauses if a returned tool result cannot be preserved without repeating the tool", async () => {
        vi.spyOn(taskStore, "recordToolRun").mockRejectedValue(new Error("Disk unavailable"));
        const model = vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([{ name: "measure_text_length", args: { text: "hello" } }]));
        const result = await Runtime.answer(createInput());
        expect(result.outcome).toBe("paused");
        expect(result.answer).toContain("não consegui guardar");
        expect(model).toHaveBeenCalledOnce();
    });

    it("does not import a neighboring task's evidence from channel-wide history", async () => {
        const history = vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([{
            toolName: "retrieve_messages", summary: "neighbor-private-source", learned: "neighbor-private-source", outputJson: "{}",
        }] as never);
        const model = vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValueOnce(makeFinishResult("Hello"));
        await Runtime.answer(createInput());
        expect(history).not.toHaveBeenCalled();
        expect(JSON.stringify(model.mock.calls[0])).not.toContain("neighbor-private-source");
    });

    it("resumes an owned task with saved steering and the current requester instruction", async () => {
        const taskId = await taskStore.create({ actorId: "u-requester", guildId: "g1", channelId: "c1", conversationId: "g1:c1:channel", objective: "Explain the deployment" });
        await taskStore.appendSteering(taskId, "u-requester", "c1", "g1", "Keep the answer in Portuguese");
        await taskStore.finish(taskId, "u-requester", "paused", "Interrupted");
        const model = vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValueOnce(makeFinishResult("Retomado."));
        const result = await Runtime.answer(createInput({ resumeTaskId: taskId, question: "Focus on the database" }));
        expect(result.taskId).toBe(taskId);
        expect(result.outcome).toBe("completed");
        const messages = JSON.stringify(model.mock.calls[0]);
        expect(messages).toContain("Explain the deployment");
        expect(messages).toContain("Focus on the database");
        expect(messages).toContain("Keep the answer in Portuguese");
    });

    it("reloads saved task context without promoting its contents into system instructions", async () => {
        const id = await taskStore.create({ actorId: "u-requester", guildId: "g1", channelId: "c1", conversationId: "g1:c1:channel", objective: "Continue" });
        const workspace = await taskStore.workspace(id, "u-requester", "c1", "g1");
        await workspace.upsertRequestPlan({ requestId: "previous-leg", threadId: "g1:c1:channel", body: "SAVED-CONTEXT-MARKER" });
        vi.spyOn(ModelGateway, "generateWithTools").mockImplementation(async messages => {
            expect(messages.some(message => message.role === "assistant" && message.content?.includes("SAVED-CONTEXT-MARKER"))).toBe(true);
            expect(messages.some(message => message.role === "system" && message.content?.includes("SAVED-CONTEXT-MARKER"))).toBe(false);
            return makeFinishResult("Working context restored.");
        });
        await Runtime.answer(createInput({ taskId: id, trigger: "auto_continue", autoContinue: false }));
        await taskStore.finish(id, "u-requester", "completed", "test end");
    });
    it("does not continue another task's open goals in the same channel", async () => {
        const id = await taskStore.create({ actorId: "u-requester", guildId: "g1", channelId: "c1", conversationId: "g1:c1:channel", objective: "Unrelated work" });
        const workspace = await taskStore.workspace(id, "u-requester", "c1", "g1");
        await workspace.addRequestGoal({ requestId: "unrelated", threadId: "g1:c1:channel", label: null, body: "This must not trigger continuation" });
        const model = vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValue(makeFinishResult("A separate reply."));
        const result = await Runtime.answer(createInput());
        expect(result.outcome).toBe("completed");
        expect(model).toHaveBeenCalledOnce();
        expect((await taskStore.snapshot(id, "u-requester", "c1", "g1"))?.goals[0].status).toBe("open");
        await taskStore.finish(id, "u-requester", "paused", "test end");
    });

    it("marks the task paused when a finished leg leaves an open goal", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([{ name: "goal_open", args: { body: "Inspect the remaining sources" } }]))
            .mockResolvedValueOnce(makeFinishResult("The first source is summarized; the others remain unread."));
        const result = await Runtime.answer(createInput({ autoContinue: false }));
        expect(result.outcome).toBe("paused");
        expect((await taskStore.list("u-requester", "c1", "g1")).find(task => task.id === result.taskId)?.status).toBe("paused");
    });
    it("keeps a shared task running until its outer continuation chain finishes", async () => {
        const id = await taskStore.create({ actorId: "u-requester", guildId: "g1", channelId: "c1", conversationId: "g1:c1:channel", objective: "shared work" });
        vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValue(makeFinishResult("This leg is complete."));
        const leg = await Runtime.answer(createInput({ taskId: id, trigger: "auto_continue", autoContinue: false }));
        expect(leg.taskId).toBe(id);
        expect(await taskStore.ownsActive(id, "u-requester", "c1", "g1")).toBe(true);
        await taskStore.finish(id, "u-requester", "completed", leg.answer);
    });
    it("persists a task outcome and rejects a forged continuation owner", async () => {
        vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValue(makeFinishResult("The answer is ready."));
        const result = await Runtime.answer(createInput({ autoContinue: false }));
        expect(result.taskId).toBeTruthy();
        expect((await taskStore.list("u-requester", "c1", "g1")).find(task => task.id === result.taskId))
            .toMatchObject({ status: "completed", answer: "The answer is ready." });
        const activeId = await taskStore.create({ actorId: "owner", guildId: "g1", channelId: "c1", conversationId: "conversation", objective: "private task" });
        await expect(Runtime.answer(createInput({ taskId: activeId, trigger: "auto_continue" }))).rejects.toThrow("does not belong");
        await taskStore.finish(activeId, "owner", "cancelled", "test cleanup");
    });
    it("does not run an approved action whose instruction changed during approval", async () => {
        const execution = new ExecutionControl("u-requester", "c1");
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        const send = vi.fn();
        vi.spyOn(CapabilityRegistry, "get").mockImplementation(id => id === "send_message" ? { ...originalGet(id), run: send } : originalGet(id));
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([{ name: "send_message", args: { channel_id: "123456789012345678", content: "old text" } }]))
            .mockResolvedValueOnce(makeFinishResult("The draft is ready; nothing was sent."));
        const approvalGate = vi.fn(async () => {
            execution.steer("Keep this as a draft. Do not send it.");
            return { approved: true, decidedBy: "u-requester", decidedAt: Date.now() };
        });
        const result = await Runtime.answer(createInput({ execution, autoContinue: false, approvalGate,
            authorize: async effect => effect === "none" ? "allow" : "ask" }));
        expect(approvalGate).toHaveBeenCalledOnce();
        expect(send).not.toHaveBeenCalled();
        expect(result.answer).toContain("nothing was sent");
    });
    it("discards an in-flight finish when the requester changes direction", async () => {
        const execution = new ExecutionControl("u-requester", "c1");
        const model = vi.spyOn(ModelGateway, "generateWithTools")
            .mockImplementationOnce(async () => {
                execution.steer("Write the answer in Portuguese.");
                return makeFinishResult("Stale English answer.");
            }).mockImplementationOnce(async messages => {
                expect(messages.some(message => message.role === "user" && message.content?.includes("Write the answer in Portuguese."))).toBe(true);
                expect(JSON.stringify(messages)).not.toContain("Stale English answer.");
                return makeFinishResult("A resposta corrigida está em português.");
            });
        const result = await Runtime.answer(createInput({ execution, autoContinue: false }));
        expect(result.answer).toBe("A resposta corrigida está em português.");
        expect(model).toHaveBeenCalledTimes(2);
    });

    it("keeps completed results and skips remaining calls when steered during a tool", async () => {
        const execution = new ExecutionControl("u-requester", "c1");
        const originalGet = CapabilityRegistry.get.bind(CapabilityRegistry);
        const run = vi.fn(async () => {
            execution.steer("Stop calculating and explain the first result.");
            return { tool: "evaluate_math" as const, summary: "2", data: { result: 2 } };
        });
        vi.spyOn(CapabilityRegistry, "get").mockImplementation(id => id === "evaluate_math" ? { ...originalGet(id), run } : originalGet(id));
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(makeToolCallResult([
                { name: "evaluate_math", args: { expression: "1+1" } },
                { name: "evaluate_math", args: { expression: "2+2" } },
            ]))
            .mockImplementationOnce(async messages => {
                const assistant = messages.find(message => message.role === "assistant" && "tool_calls" in message && message.tool_calls?.length === 2);
                expect(assistant).toBeDefined();
                for (const call of (assistant as any).tool_calls) {
                    expect(messages.filter(message => message.role === "tool" && message.tool_call_id === call.id)).toHaveLength(1);
                }
                expect(JSON.stringify(messages)).toContain("Stop calculating and explain");
                return makeFinishResult("One plus one is two.");
            });
        const result = await Runtime.answer(createInput({ execution, autoContinue: false }));
        expect(run).toHaveBeenCalledTimes(1);
        expect(result.answer).toBe("One plus one is two.");
    });
    beforeEach(async () => {
        vi.restoreAllMocks();
        toolCallCounter = 0;
        process.env.DISCORD_TOKEN = "test-token";
        process.env.OPENROUTER_API_KEY = "test-key";

        vi.spyOn(DiscordMemoryService, "getRecentRuntimeRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getRecentChannelMessagesAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "getKnownChannelsAsync").mockResolvedValue([]);
        vi.spyOn(DiscordMemoryService, "recordToolRun").mockResolvedValue(undefined);
        vi.spyOn(DiscordMemoryService, "recordRuntimeRun").mockResolvedValue(undefined);
        vi.spyOn(ModelGateway, "generateText").mockResolvedValue("ok, continuando na tarefa");
    });

    it("honors cancellation received while a model call is in flight", async () => {
        const execution = new ExecutionControl("u-requester", "c1");
        vi.spyOn(ModelGateway, "generateWithTools").mockImplementation(async () => {
            execution.cancel();
            return makeFinishResult("This should not be delivered.");
        });
        const result = await Runtime.answer(createInput({ execution, autoContinue: false }));
        expect(result.outcome).toBe("cancelled");
        expect(result.answer).not.toContain("This should not be delivered");
    });

    it("pauses repeated invalid tool rounds without claiming completion", async () => {
        const model = vi.spyOn(ModelGateway, "generateWithTools").mockImplementation(async () =>
            makeToolCallResult([{ name: "nonexistent_tool", args: {} }]));
        const result = await Runtime.answer(createInput({ autoContinue: false }));
        expect(result.outcome).toBe("paused");
        expect(model).toHaveBeenCalledTimes(3);
    });

    it("finishes after more than 200 productive tool calls without declaring a long task", async () => {
        let count = 0;
        const model = vi.spyOn(ModelGateway, "generateWithTools").mockImplementation(async () => {
            if (count === 205) return makeFinishResult("Finished all 205 calculations.");
            count += 1;
            return makeToolCallResult([{ name: "evaluate_math", args: { expression: `${count}+1` } }]);
        });
        const result = await Runtime.answer(createInput({ question: "Complete these calculations.", autoContinue: false }));
        expect(result.answer).toBe("Finished all 205 calculations.");
        expect(result.outcome).toBe("completed");
        expect(result.toolRuns).toHaveLength(205);
        expect(model).toHaveBeenCalledTimes(206);
    }, 15000);

    it("answers a follow-up directly from reused prior retrieval evidence without rerunning tools", async () => {
        const followUpToolRun = {
            requestId: "req-previous",
            toolName: "retrieve_messages",
            argumentsJson: JSON.stringify({
                query: "9 de fevereiro openrosen youtube",
                channelIds: ["c-comandos"],
                authorId: "u-open",
            }),
            summary: "ordered history evidence; 2 history and 0 semantic result(s) from cached Discord history.",
            learned: "Rosewind mencionou que parecia Brackel.",
            outputJson: JSON.stringify({
                tool: "retrieve_messages",
                summary: "ordered history evidence",
                data: {
                    mode: "history",
                    sourceOrigin: "cache",
                    targetAuthorId: "u-open",
                    targetChannelIds: ["c-comandos"],
                    historyMessages: [
                        {
                            messageId: "m1",
                            channelId: "c-comandos",
                            channelName: "comandos",
                            guildId: "g1",
                            authorId: "u-open",
                            authorName: "Rosewind",
                            content: "A foto parecia o Brackel nessa thumb.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/c-comandos/m1",
                            lexicalScore: 4,
                            semanticScore: 0,
                            recencyScore: 0,
                            totalScore: 4,
                        },
                    ],
                    semanticMatches: [],
                    combinedResults: [
                        {
                            messageId: "m1",
                            channelId: "c-comandos",
                            channelName: "comandos",
                            guildId: "g1",
                            authorId: "u-open",
                            authorName: "Rosewind",
                            content: "A foto parecia o Brackel nessa thumb.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/c-comandos/m1",
                            lexicalScore: 4,
                            semanticScore: 0,
                            recencyScore: 0,
                            totalScore: 4,
                        },
                    ],
                    continuation: {
                        history: {
                            perChannelOldestMessageId: { "c-comandos": "m1" },
                            continuationAvailable: true,
                        },
                        semantic: {
                            cursor: null,
                            continuationAvailable: false,
                        },
                        continuationAvailable: true,
                    },
                },
            }),
            createdTimestamp: Date.now() - 5_000,
        };

        const taskId = await taskStore.create({ actorId: "u-requester", guildId: "g1", channelId: "c1", conversationId: "g1:c1:channel", objective: "Follow-up" });
        await taskStore.recordToolRun({ taskId, actorId: "u-requester", guildId: "g1", channelId: "c1", requestId: "earlier-leg", invocationId: "earlier-call",
            record: { tool: "retrieve_messages", arguments: {}, summary: followUpToolRun.summary, learned: followUpToolRun.learned,
                output: JSON.parse(followUpToolRun.outputJson), confidenceImproved: true, durationMs: 1 } });

        // Model sees prior evidence in the system prompt and answers directly via finish
        vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValue(
            makeFinishResult("Sim, ele comparou a foto com o Brackel.")
        );

        const result = await Runtime.answer(
            createInput({
                question: "ele comentou com qual artista parecia?",
                taskId, trigger: "auto_continue",
            })
        );

        expect(result.answer).toBe("Sim, ele comparou a foto com o Brackel.");
        expect(result.toolRuns).toEqual([]);
        expect(JSON.stringify(vi.mocked(ModelGateway.generateWithTools).mock.calls[0])).toContain("Brackel");
    });

    it("drops stale scoped targets when a new question explicitly retargets a different channel group", async () => {
        const priorMemberRun = {
            requestId: "req-member",
            toolName: "resolve_member_identity",
            argumentsJson: JSON.stringify({ query: "Faymon" }),
            summary: "resolved Faymon",
            learned: "Resolved the author target.",
            outputJson: JSON.stringify({
                tool: "resolve_member_identity",
                summary: "resolved Faymon",
                data: {
                    query: "Faymon",
                    resolvedId: "222222222222222222",
                    displayName: "Faymon",
                    username: "Faymon",
                    globalName: null,
                    nickname: null,
                    isBot: true,
                    isCurrentGuildMember: true,
                    source: "live_exact",
                    confidence: "exact",
                    roles: [],
                },
            }),
            createdTimestamp: Date.now() - 10_000,
        };

        const priorRetrievalRun = {
            requestId: "req-retrieval",
            toolName: "retrieve_messages",
            argumentsJson: JSON.stringify({
                query: "markov",
                channelIds: ["333333333333333333"],
                authorId: "222222222222222222",
            }),
            summary: "ordered history evidence",
            learned: "Retrieved prior scoped discussion messages.",
            outputJson: JSON.stringify({
                tool: "retrieve_messages",
                summary: "ordered history evidence",
                data: {
                    mode: "history",
                    sourceOrigin: "cache_after_refresh",
                    targetAuthorId: "222222222222222222",
                    targetChannelIds: ["333333333333333333"],
                    searchedChannelIds: ["333333333333333333"],
                    historyMessages: [
                        {
                            messageId: "m-disc-1",
                            channelId: "333333333333333333",
                            channelName: "discussão",
                            guildId: "g1",
                            authorId: "222222222222222222",
                            authorName: "Faymon",
                            authorUsername: "Faymon",
                            content: "Eu não sou eu.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/333333333333333333/m-disc-1",
                            lexicalScore: 4,
                        },
                    ],
                    semanticMatches: [],
                    combinedResults: [
                        {
                            messageId: "m-disc-1",
                            channelId: "333333333333333333",
                            channelName: "discussão",
                            guildId: "g1",
                            authorId: "222222222222222222",
                            authorName: "Faymon",
                            authorUsername: "Faymon",
                            content: "Eu não sou eu.",
                            createdTimestamp: 1707510000000,
                            jumpLink: "https://discord.com/channels/g1/333333333333333333/m-disc-1",
                            lexicalScore: 4,
                        },
                    ],
                    continuation: {
                        history: {
                            perChannelOldestMessageId: { "333333333333333333": "m-disc-1" },
                            continuationAvailable: true,
                        },
                        semantic: {
                            cursor: null,
                            continuationAvailable: false,
                        },
                        continuationAvailable: true,
                    },
                },
            }),
            createdTimestamp: Date.now() - 9_000,
        };

        vi.spyOn(DiscordMemoryService, "getRecentToolRunsAsync").mockResolvedValue([
            priorMemberRun as any,
            priorRetrievalRun as any,
        ]);

        const previewSpy = vi.fn().mockResolvedValue(undefined);

        // Model sees stale evidence but decides to answer directly about a new topic
        vi.spyOn(ModelGateway, "generateWithTools").mockResolvedValue(
            makeFinishResult("Os canais de Serviços incluem bot-commands e automation.")
        );

        const result = await Runtime.answer(
            createInput({
                question:
                    "estou com pressa e precisava de um resumo do que tem nos canais do Serviços. quero uma descrição de cada serviço",
                trigger: "mention",
                debugSession: {
                    setClassifying: vi.fn().mockResolvedValue(undefined),
                    setClassification: vi.fn().mockResolvedValue(undefined),
                    setPlanning: vi.fn().mockResolvedValue(undefined),
                    setToolRunning: vi.fn().mockResolvedValue(undefined),
                    setToolResult: vi.fn().mockResolvedValue(undefined),
                    setEvidenceSummary: vi.fn().mockResolvedValue(undefined),
                    setGenerating: vi.fn().mockResolvedValue(undefined),
                    finishSuccess: vi.fn().mockResolvedValue(undefined),
                    finishError: vi.fn().mockResolvedValue(undefined),
                    setTraceEvent: previewSpy,
                },
            })
        );

        expect(result.answer.length).toBeGreaterThan(0);
        // Model answered directly via finish — no tools were called
        expect(result.toolRuns).toEqual([]);
    });

    it("answers requester identity naturally after resolving the requester exactly", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_member_identity", args: { queries: ["u-requester"] } }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("Tu és o Requester aqui no servidor.")
            );
        vi.spyOn(DiscordLiveService, "resolveMemberIdentity").mockResolvedValue({
            query: "u-requester",
            resolvedId: "u-requester",
            displayName: "Requester",
            username: "requester",
            globalName: null,
            nickname: null,
            isBot: false,
            isCurrentGuildMember: true,
            source: "live_id",
            confidence: "exact",
            roles: ["Admin"],
        });

        const result = await Runtime.answer(
            createInput({
                question: "Quem sou eu?",
            })
        );

        expect(result.answer).toBe("Tu és o Requester aqui no servidor.");
        expect(result.toolRuns.map((run) => run.tool)).toEqual(["resolve_member_identity"]);
        expect(DiscordLiveService.resolveMemberIdentity).toHaveBeenCalledWith(
            expect.objectContaining({ id: "g1" }),
            "u-requester"
        );
    });

    it("combines member resolution, channel resolution, and message retrieval for a grounded explanation", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_member_identity", args: { queries: ["Riverside"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_channel_targets", args: { targets: ["reflexoes"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "list_guild_structure", args: { targetText: "reflexoes" } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "retrieve_messages", args: { query: "do que o Riverside está falando?", authorId: "u-one", channelIds: ["c-reflexoes"] } }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("O Riverside estava falando sobre a imagem e dizendo que ela era mais simbólica do que literal.")
            );
        vi.spyOn(DiscordLiveService, "resolveMemberIdentity").mockResolvedValue({
            query: "Riverside",
            resolvedId: "u-one",
            displayName: "Riverside",
            username: "riverside",
            globalName: null,
            nickname: "Riverside",
            isBot: false,
            isCurrentGuildMember: true,
            source: "live_search",
            confidence: "high",
            roles: [],
        });
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargetsBatch").mockResolvedValue([{
            query: "reflexoes",
            resolvedIds: ["c-reflexoes"],
            entries: [
                {
                    id: "c-reflexoes",
                    guildId: "g1",
                    name: "reflexoes",
                    type: "0",
                    position: null,
                    parentCategoryId: "cat-1",
                    parentCategoryName: "Text",
                    isReadable: true,
                    isViewable: true,
                    isIndexed: true,
                    source: "live",
                    missingOrDeletedPossible: false,
                },
            ],
            exactIdMatch: false,
            confidence: "high",
        }]);
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-1",
                guildId: "g1",
                name: "Text",
                type: "4",
                position: null,
                parentCategoryId: null,
                parentCategoryName: null,
                isReadable: false,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "c-reflexoes",
                guildId: "g1",
                name: "reflexoes",
                type: "0",
                position: null,
                parentCategoryId: "cat-1",
                parentCategoryName: "Text",
                isReadable: true,
                isViewable: true,
                isIndexed: true,
                source: "live",
                missingOrDeletedPossible: false,
            },
        ]);
        vi.spyOn(UnifiedMessageRetrieval, "retrieve").mockResolvedValue({
            query: "do que o Riverside está falando?",
            mode: "mixed",
            historyMessages: [
                {
                    messageId: "m1",
                    channelId: "c-reflexoes",
                    channelName: "reflexoes",
                    guildId: "g1",
                    authorId: "u-one",
                    authorName: "Riverside",
                    content: "A imagem era mais simbólica do que literal.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-reflexoes/m1",
                    lexicalScore: 3,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 3,
                },
            ],
            semanticMatches: [],
            combinedResults: [
                {
                    messageId: "m1",
                    channelId: "c-reflexoes",
                    channelName: "reflexoes",
                    guildId: "g1",
                    authorId: "u-one",
                    authorName: "Riverside",
                    content: "A imagem era mais simbólica do que literal.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-reflexoes/m1",
                    lexicalScore: 3,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 3,
                },
            ],
            cacheHit: true,
            liveEscalated: false,
            searchedChannelIds: ["c-reflexoes"],
            fetchedChannelIds: [],
            cacheEnriched: false,
            evidenceSufficient: true,
            strongResultCount: 1,
            weakResultCount: 0,
            historyMessageCount: 1,
            semanticMatchCount: 0,
            sourceOrigin: "cache",
            targetAuthorId: "u-one",
            targetChannelIds: ["c-reflexoes"],
            continuation: {
                history: {
                    perChannelOldestMessageId: { "c-reflexoes": "m1" },
                    continuationAvailable: true,
                },
                semantic: {
                    cursor: null,
                    continuationAvailable: false,
                },
                perChannelOldestMessageId: { "c-reflexoes": "m1" },
                continuationAvailable: true,
            },
            exhaustion: {
                historyExhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                exhaustedChannelIds: [],
                exhausted: false,
            },
            accumulatedWindow: {
                beforeTimestamp: null,
                afterTimestamp: null,
            },
            accumulatedUniqueCount: 1,
            beforeTimestamp: null,
            afterTimestamp: null,
            excludedMessageIds: [],
        });

        const result = await Runtime.answer(
            createInput({
                question: "do que o Riverside está falando em #reflexoes?",
            })
        );

        expect(result.answer).toBe(
            "O Riverside estava falando sobre a imagem e dizendo que ela era mais simbólica do que literal."
        );
        expect(result.toolRuns.map((run) => run.tool)).toEqual([
            "resolve_member_identity",
            "resolve_channel_targets",
            "list_guild_structure",
            "retrieve_messages",
        ]);
        expect(UnifiedMessageRetrieval.retrieve).toHaveBeenCalledWith(
            expect.objectContaining({
                authorId: "u-one",
                channelIds: ["c-reflexoes"],
            })
        );
    });

    it("can combine channel target resolution with guild structure discovery to answer structure questions", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_channel_targets", args: { targets: ["123456789012345678"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "list_guild_structure", args: {} }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("Esse id corresponde ao canal #ideas, dentro da categoria Projects.")
            );
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargetsBatch").mockResolvedValue([{
            query: "123456789012345678",
            resolvedIds: ["123456789012345678"],
            entries: [
                {
                    id: "123456789012345678",
                    guildId: "g1",
                    name: "ideas",
                    type: "0",
                    position: null,
                    parentCategoryId: "cat-projects",
                    parentCategoryName: "Projects",
                    isReadable: true,
                    isViewable: true,
                    isIndexed: true,
                    source: "live",
                    missingOrDeletedPossible: false,
                },
            ],
            exactIdMatch: true,
            confidence: "exact",
        }]);
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-projects",
                guildId: "g1",
                name: "Projects",
                type: "4",
                position: null,
                parentCategoryId: null,
                parentCategoryName: null,
                isReadable: false,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "123456789012345678",
                guildId: "g1",
                name: "ideas",
                type: "0",
                position: null,
                parentCategoryId: "cat-projects",
                parentCategoryName: "Projects",
                isReadable: true,
                isViewable: true,
                isIndexed: true,
                source: "live",
                missingOrDeletedPossible: false,
            },
        ]);

        const result = await Runtime.answer(
            createInput({
                question: "what channel is 123456789012345678?",
            })
        );

        expect(result.answer).toBe(
            "Esse id corresponde ao canal #ideas, dentro da categoria Projects."
        );
        expect(result.toolRuns.map((run) => run.tool)).toEqual([
            "resolve_channel_targets",
            "list_guild_structure",
        ]);
    });

    it("inspects a matched category and then retrieves scoped messages before describing available services, even when one target channel is not indexed", async () => {
        vi.spyOn(ModelGateway, "generateWithTools")
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "resolve_channel_targets", args: { targets: ["serviços"] } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "list_guild_structure", args: { targetText: "serviços" } }])
            )
            .mockResolvedValueOnce(
                makeToolCallResult([{ name: "retrieve_messages", args: { query: "que serviços estão disponíveis nesse servidor?", channelIds: ["c-bot-commands", "c-automation"] } }])
            )
            .mockResolvedValueOnce(
                makeFinishResult("Na categoria Serviços, vocês têm pelo menos o #bot-commands para comandos e o #automation para automações e integrações.")
            );
        vi.spyOn(DiscordGuildDiscoveryService, "resolveChannelTargetsBatch").mockResolvedValue([{
            query: "serviços",
            resolvedIds: ["c-bot-commands", "c-automation"],
            entries: [
                {
                    id: "cat-services",
                    guildId: "g1",
                    name: "Serviços",
                    type: "4",
                    position: null,
                    parentCategoryId: null,
                    parentCategoryName: null,
                    isReadable: false,
                    isViewable: true,
                    isIndexed: false,
                    source: "live",
                    missingOrDeletedPossible: false,
                },
            ],
            exactIdMatch: false,
            confidence: "high",
        }]);
        vi.spyOn(DiscordGuildDiscoveryService, "listGuildStructure").mockResolvedValue([
            {
                id: "cat-services",
                guildId: "g1",
                name: "Serviços",
                type: "4",
                position: null,
                parentCategoryId: null,
                parentCategoryName: null,
                isReadable: false,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "c-bot-commands",
                guildId: "g1",
                name: "bot-commands",
                type: "0",
                position: null,
                parentCategoryId: "cat-services",
                parentCategoryName: "Serviços",
                isReadable: true,
                isViewable: true,
                isIndexed: true,
                source: "live",
                missingOrDeletedPossible: false,
            },
            {
                id: "c-automation",
                guildId: "g1",
                name: "automation",
                type: "0",
                position: null,
                parentCategoryId: "cat-services",
                parentCategoryName: "Serviços",
                isReadable: true,
                isViewable: true,
                isIndexed: false,
                source: "live",
                missingOrDeletedPossible: false,
            },
        ]);
        vi.spyOn(UnifiedMessageRetrieval, "retrieve").mockResolvedValue({
            query: "que serviços estão disponíveis nesse servidor?",
            mode: "mixed",
            historyMessages: [
                {
                    messageId: "m1",
                    channelId: "c-bot-commands",
                    channelName: "bot-commands",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Service Bot",
                    content: "Use este canal para comandos e utilidades do bot.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-bot-commands/m1",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
                {
                    messageId: "m2",
                    channelId: "c-automation",
                    channelName: "automation",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Automation Bot",
                    content: "Este canal centraliza automações e integrações.",
                    createdTimestamp: 1700000001000,
                    jumpLink: "https://discord.com/channels/g1/c-automation/m2",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
            ],
            semanticMatches: [],
            combinedResults: [
                {
                    messageId: "m1",
                    channelId: "c-bot-commands",
                    channelName: "bot-commands",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Service Bot",
                    content: "Use este canal para comandos e utilidades do bot.",
                    createdTimestamp: 1700000000000,
                    jumpLink: "https://discord.com/channels/g1/c-bot-commands/m1",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
                {
                    messageId: "m2",
                    channelId: "c-automation",
                    channelName: "automation",
                    guildId: "g1",
                    authorId: "u-bot",
                    authorName: "Automation Bot",
                    content: "Este canal centraliza automações e integrações.",
                    createdTimestamp: 1700000001000,
                    jumpLink: "https://discord.com/channels/g1/c-automation/m2",
                    lexicalScore: 4,
                    semanticScore: 0,
                    recencyScore: 0,
                    totalScore: 4,
                },
            ],
            cacheHit: false,
            liveEscalated: true,
            searchedChannelIds: ["c-bot-commands", "c-automation"],
            fetchedChannelIds: ["c-automation"],
            cacheEnriched: true,
            evidenceSufficient: true,
            strongResultCount: 2,
            weakResultCount: 0,
            historyMessageCount: 2,
            semanticMatchCount: 0,
            sourceOrigin: "live_refresh",
            targetAuthorId: null,
            targetChannelIds: ["c-bot-commands", "c-automation"],
            continuation: {
                history: {
                    perChannelOldestMessageId: {
                        "c-bot-commands": "m1",
                        "c-automation": "m2",
                    },
                    continuationAvailable: true,
                },
                semantic: {
                    cursor: null,
                    continuationAvailable: false,
                },
                perChannelOldestMessageId: {
                    "c-bot-commands": "m1",
                    "c-automation": "m2",
                },
                continuationAvailable: true,
            },
            exhaustion: {
                historyExhaustedChannelIds: [],
                historyExhausted: false,
                semanticExhausted: true,
                exhaustedChannelIds: [],
                exhausted: false,
            },
            accumulatedWindow: {
                beforeTimestamp: null,
                afterTimestamp: null,
            },
            accumulatedUniqueCount: 2,
            beforeTimestamp: null,
            afterTimestamp: null,
            excludedMessageIds: [],
        });

        const result = await Runtime.answer(
            createInput({
                question: "que serviços estão disponíveis nesse servidor?",
            })
        );

        expect(result.answer).toBe(
            "Na categoria Serviços, vocês têm pelo menos o #bot-commands para comandos e o #automation para automações e integrações."
        );
        expect(result.toolRuns.map((run) => run.tool)).toEqual([
            "resolve_channel_targets",
            "list_guild_structure",
            "retrieve_messages",
        ]);
        expect(UnifiedMessageRetrieval.retrieve).toHaveBeenCalledWith(
            expect.objectContaining({
                channelIds: ["c-bot-commands", "c-automation"],
            })
        );
        expect(result.toolRuns.find((run) => run.tool === "retrieve_messages")?.data).toEqual(
            expect.objectContaining({
                liveEscalated: true,
                fetchedChannelIds: ["c-automation"],
                cacheEnriched: true,
            })
        );
    });

});
