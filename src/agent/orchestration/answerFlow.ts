import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";

export const EMPTY_OUTPUT_FALLBACK =
    "Não consegui gerar uma resposta válida agora. Tente novamente.";

export async function generateDirectAnswer(question: string): Promise<string> {
    return ModelGateway.generateText(
        [
            { role: "system", content: PromptRegistry.load("system/base") },
            { role: "user", content: question },
        ],
        {
            traceContext: {
                traceLabel: "direct_answer_generation",
                questionPreview: question,
            },
        }
    );
}

export async function generateGroundedAnswer(
    question: string,
    evidence: string
): Promise<string> {
    return ModelGateway.generateText(
        [
            { role: "system", content: PromptRegistry.load("system/base") },
            { role: "system", content: PromptRegistry.load("system/grounded") },
            {
                role: "user",
                content: PromptRegistry.render("tasks/synthesize_answer", {
                    question,
                    evidence,
                }),
            },
        ],
        {
            traceContext: {
                traceLabel: "grounded_answer_generation",
                questionPreview: question,
            },
        }
    );
}
