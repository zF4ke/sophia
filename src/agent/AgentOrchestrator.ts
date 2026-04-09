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
import { runDiscordToolLoop } from "@/agent/orchestration/toolLoop";
import { assessGrounding } from "@/agent/orchestration/grounding";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";

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
        const toolRuns = await runDiscordToolLoop(options);
        const grounding = assessGrounding(options.question, toolRuns);
        await options.debugSession?.setGroundingSummary(grounding.summary);

        if (!grounding.summary.sufficient) {
            await options.debugSession?.finishSuccess("Concluído sem evidência suficiente.");
            return {
                answer: PromptRegistry.load("guards/insufficient_evidence"),
                citations: [],
                classification,
                toolRuns,
            };
        }

        await options.debugSession?.setGenerating();
        const answer = await generateGroundedAnswer(options.question, grounding.evidence);
        this.recordToolRuns(options, toolRuns);
        await options.debugSession?.finishSuccess("Resposta concluída.");

        return {
            answer,
            citations: grounding.citations,
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
