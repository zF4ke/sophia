import type { Guild, User } from "discord.js";
import { EmptyModelOutputError } from "@/ai/EmptyModelOutputError";
import type { DebugSessionReporter } from "@/discord/debug/types";
import { RequestClassifier } from "@/agent/RequestClassifier";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type {
    DiscordToolResult,
    RequestClassification,
} from "@/shared/appTypes";
import {
    EMPTY_OUTPUT_FALLBACK,
    generateDirectAnswer,
    generateGroundedAnswer,
} from "@/agent/orchestration/answerFlow";
import {
    findConversationResolutionContext,
    saveConversationResolutionContext,
} from "@/agent/orchestration/conversationContext";
import { decideRetrievalAction } from "@/agent/orchestration/retrievalController";
import {
    findRecentConversationGroundedContext,
    findReusableGroundedContext,
    saveReusableGroundedContext,
} from "@/agent/orchestration/reuseCache";
import { runWithRequestCacheContext } from "@/agent/orchestration/requestCacheContext";
import { runDiscordToolLoop } from "@/agent/orchestration/toolLoop";
import { assessGrounding } from "@/agent/orchestration/grounding";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import { isReferentialFollowUp } from "@/agent/orchestration/questionAnalysis";
import type { SearchContext } from "@/agent/orchestration/types";

export class AgentOrchestrator {
    public static async answerQuestion(options: {
        question: string;
        user: User;
        guild: Guild | null;
        currentChannelId?: string | null;
        debugSession?: DebugSessionReporter | null;
    }): Promise<{
        answer: string;
        citations: Array<{ label: string; jumpLink: string }>;
        classification: RequestClassification;
        toolRuns: DiscordToolResult[];
    }> {
        const responseOrdinal = DiscordMemoryService.nextGuildResponseOrdinal(
            options.guild?.id || null
        );

        return runWithRequestCacheContext(
            {
                guildId: options.guild?.id || null,
                responseOrdinal,
            },
            async () => {
                try {
                    await options.debugSession?.setClassifying();

                    const classification = await RequestClassifier.classify(options.question);
                    await options.debugSession?.setClassification(classification.mode);

                    if (classification.mode === "direct_answer") {
                        return await this.answerDirectQuestion(
                            options.question,
                            classification,
                            options.debugSession
                        );
                    }

                    return await this.answerGroundedQuestion(options, classification);
                } catch (error) {
                    if (error instanceof EmptyModelOutputError) {
                        await options.debugSession?.finishSuccess(
                            "Modelo devolveu resposta vazia; usei fallback."
                        );
                        return {
                            answer: EMPTY_OUTPUT_FALLBACK,
                            citations: [],
                            classification: {
                                mode: "direct_answer",
                                reason: "Fallback after empty model output.",
                            },
                            toolRuns: [],
                        };
                    }

                    await options.debugSession?.finishError(error);
                    throw error;
                }
            }
        );
    }

    private static async answerDirectQuestion(
        question: string,
        classification: RequestClassification,
        debugSession?: DebugSessionReporter | null
    ) {
        await debugSession?.setGenerating();
        const answer = await generateDirectAnswer(question);
        await debugSession?.finishSuccess("Resposta direta concluída.");

        return {
            answer,
            citations: [],
            classification,
            toolRuns: [],
        };
    }

    private static async answerGroundedQuestion(
        options: {
            question: string;
            user: User;
            guild: Guild | null;
            currentChannelId?: string | null;
            debugSession?: DebugSessionReporter | null;
        },
        classification: RequestClassification
    ) {
        const priorConversationContext = findConversationResolutionContext({
            guildId: options.guild?.id || null,
            currentChannelId: options.currentChannelId || null,
        });
        const initialControllerContext: SearchContext = {
            crawledChannelIds: new Set<string>(),
            seededFromContext: false,
            initialToolRuns: [],
            latestControllerDecision: null,
        };
        const initialDecision = await decideRetrievalAction({
            question: options.question,
            guild: options.guild,
            currentChannelId: options.currentChannelId,
            toolRuns: [],
            priorContext: priorConversationContext,
            context: initialControllerContext,
        });
        await options.debugSession?.setRouting?.(initialDecision);
        await options.debugSession?.setContextCacheStatus?.("none");

        const reusableContext = findReusableGroundedContext({
            guildId: options.guild?.id || null,
            currentChannelId: options.currentChannelId || null,
            question: options.question,
            routeIntent: initialDecision.routeIntent,
            requireSufficient: true,
        });

        if (reusableContext) {
            await options.debugSession?.setContextCacheStatus?.("reused");
            const cachedGrounding = assessGrounding(options.question, reusableContext.toolRuns);
            const cachedDecision = await decideRetrievalAction({
                question: options.question,
                guild: options.guild,
                currentChannelId: options.currentChannelId,
                toolRuns: reusableContext.toolRuns,
                priorContext: priorConversationContext,
                context: {
                    ...initialControllerContext,
                    seededFromContext: true,
                    initialToolRuns: [...reusableContext.toolRuns],
                },
                forcedFinal: true,
            });
            await options.debugSession?.setRouting?.(cachedDecision);
            await options.debugSession?.setGroundingSummary(
                {
                    ...cachedGrounding.summary,
                    sufficient: cachedDecision.answerConfidence !== "insufficient",
                },
                "reused",
                cachedDecision.answerConfidence
            );

            if (cachedDecision.answerConfidence === "insufficient") {
                await options.debugSession?.finishSuccess("Concluído sem evidência suficiente.");
                return {
                    answer: PromptRegistry.load("guards/insufficient_evidence"),
                    citations: [],
                    classification,
                    toolRuns: reusableContext.toolRuns,
                };
            }

            await options.debugSession?.setGenerating();
            const answer = await generateGroundedAnswer(
                options.question,
                reusableContext.evidenceText,
                cachedDecision.answerConfidence
            );
            await options.debugSession?.finishSuccess("Resposta concluída.");

            return {
                answer,
                citations: [],
                classification,
                toolRuns: reusableContext.toolRuns,
            };
        }

        const seedContext = findReusableGroundedContext({
            guildId: options.guild?.id || null,
            currentChannelId: options.currentChannelId || null,
            question: options.question,
            routeIntent: initialDecision.routeIntent,
            requireSufficient: false,
        });
        const followUpSeedContext =
            !seedContext &&
            priorConversationContext &&
            isReferentialFollowUp(options.question)
                ? findRecentConversationGroundedContext({
                      guildId: options.guild?.id || null,
                      currentChannelId: options.currentChannelId || null,
                      routeIntent: priorConversationContext.routeIntent,
                      requireSufficient: true,
                  })
                : null;
        const effectiveSeedContext = seedContext ?? followUpSeedContext;
        const initialToolRuns = effectiveSeedContext?.toolRuns?.length
            ? effectiveSeedContext.toolRuns
            : [];
        if (initialToolRuns.length) {
            await options.debugSession?.setContextCacheStatus?.("seeded");
        }

        const toolLoopResult = await runDiscordToolLoop({
            ...options,
            initialToolRuns,
            seededFromContext: initialToolRuns.length > 0,
            priorContext: priorConversationContext,
        });
        const toolRuns = toolLoopResult.toolRuns;
        const grounding = assessGrounding(options.question, toolRuns);
        const finalDecision = toolLoopResult.finalDecision;
        const groundingSummary = {
            ...grounding.summary,
            sufficient: finalDecision.answerConfidence !== "insufficient",
        };
        await options.debugSession?.setGroundingSummary(
            groundingSummary,
            effectiveSeedContext ? "reused" : "heuristic",
            finalDecision.answerConfidence
        );

        saveReusableGroundedContext({
            guildId: options.guild?.id || null,
            currentChannelId: options.currentChannelId || null,
            question: options.question,
            controllerDecision: finalDecision,
            grounding,
            groundingDecision: {
                sufficient: finalDecision.answerConfidence !== "insufficient",
                mode: effectiveSeedContext ? "reused" : "heuristic",
                answerMode: finalDecision.answerConfidence,
                reason: finalDecision.reason,
                missingInformation:
                    finalDecision.answerConfidence === "insufficient"
                        ? "More Discord evidence is needed."
                        : null,
            },
            toolRuns,
        });
        saveConversationResolutionContext({
            guildId: options.guild?.id || null,
            currentChannelId: options.currentChannelId || null,
            controllerDecision: finalDecision,
            toolRuns,
        });

        if (finalDecision.answerConfidence === "insufficient") {
            await options.debugSession?.finishSuccess("Concluído sem evidência suficiente.");
            return {
                answer: PromptRegistry.load("guards/insufficient_evidence"),
                citations: [],
                classification,
                toolRuns,
            };
        }

        await options.debugSession?.setGenerating();
        const answer = await generateGroundedAnswer(
            options.question,
            grounding.evidence,
            finalDecision.answerConfidence
        );
        this.recordToolRuns(options, toolRuns.slice(initialToolRuns.length));
        await options.debugSession?.finishSuccess("Resposta concluída.");

        return {
            answer,
            citations: [],
            classification,
            toolRuns,
        };
    }

    private static recordToolRuns(
        options: {
            question: string;
            user: User;
            guild: Guild | null;
            currentChannelId?: string | null;
        },
        toolRuns: DiscordToolResult[]
    ): void {
        toolRuns.forEach((run) => {
            DiscordMemoryService.recordToolRun(
                options.guild?.id || null,
                options.currentChannelId || null,
                options.user.id,
                options.question,
                run.tool,
                run.summary
            );
        });
    }
}
