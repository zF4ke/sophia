import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import type { WebMode, WebStatus } from "@/shared/appTypes";

export const EMPTY_OUTPUT_FALLBACK =
    "Não consegui gerar uma resposta válida agora. Tente novamente.";

type AnswerFlowOptions = {
    webMode?: WebMode;
    webContext?: string;
    onComplete?: (meta: {
        webStatus: WebStatus;
        webSearchRequests: number;
    }) => void | Promise<void>;
};

type AnswerMessage = {
    role: "system" | "user";
    content: string;
};

export async function generateDirectAnswer(
    question: string,
    options: AnswerFlowOptions = {}
): Promise<string> {
    const messages: AnswerMessage[] = [
        { role: "system" as const, content: PromptRegistry.load("system/base") },
    ];

    if ((options.webMode || "off") !== "off") {
        messages.push({
            role: "system" as const,
            content:
                "If web search is used, answer directly without appending source lists, citations, or raw URLs unless the user explicitly asks for them.",
        });
    }

    messages.push({ role: "user" as const, content: question });

    return ModelGateway.generateText(
        messages,
        {
            webMode: options.webMode || "off",
            onComplete: options.onComplete,
            traceContext: {
                traceLabel: "direct_answer_generation",
                questionPreview: question,
                webMode: options.webMode || "off",
                webContext: options.webContext,
            },
        }
    );
}

export async function generateGroundedAnswer(
    question: string,
    evidence: string,
    answerMode: "confident" | "best_effort" | "insufficient" = "confident",
    options: AnswerFlowOptions = {}
): Promise<string> {
    const messages: AnswerMessage[] = [
        { role: "system" as const, content: PromptRegistry.load("system/base") },
        { role: "system" as const, content: PromptRegistry.load("system/grounded") },
    ];

    if ((options.webMode || "off") !== "off") {
        messages.push({
            role: "system" as const,
            content:
                "If web search is used, answer directly without appending source lists, citations, or raw URLs unless the user explicitly asks for them.",
        });
    }

    messages.push({
        role: "user" as const,
        content: PromptRegistry.render("tasks/synthesize_answer", {
            question,
            evidence,
            answer_mode: answerMode,
        }),
    });

    return ModelGateway.generateText(
        messages,
        {
            webMode: options.webMode || "off",
            onComplete: options.onComplete,
            traceContext: {
                traceLabel: "grounded_answer_generation",
                questionPreview: question,
                webMode: options.webMode || "off",
                webContext: options.webContext,
            },
        }
    );
}
