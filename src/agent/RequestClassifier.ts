import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import type { RequestClassification } from "@/shared/appTypes";

const DISCORD_SIGNAL_WORDS = [
    "server",
    "servidor",
    "discord",
    "channel",
    "canal",
    "mensagem",
    "message",
    "member",
    "membro",
    "role",
    "cargo",
    "guild",
    "esse servidor",
    "este servidor",
    "this server",
    "staff",
    "who said",
    "quem disse",
    "what happened",
    "o que aconteceu",
    "when did",
    "quando",
    "decision",
    "decisão",
];

export class RequestClassifier {
    public static async classify(question: string): Promise<RequestClassification> {
        const heuristic = this.classifyHeuristically(question);
        const prompt = PromptRegistry.render("tasks/classify_request", {
            question,
        });

        const modelResult = await ModelGateway.generateJson<RequestClassification>(
            [
                { role: "system", content: "Return strict JSON only." },
                { role: "user", content: prompt },
            ],
            heuristic,
            {
                traceContext: {
                    traceLabel: "request_classification",
                    questionPreview: question,
                },
            }
        );

        if (modelResult.mode === "direct_answer" || modelResult.mode === "discord_grounded") {
            return modelResult;
        }

        return heuristic;
    }

    public static classifyHeuristically(question: string): RequestClassification {
        const normalized = question.toLowerCase();
        const needsDiscord = DISCORD_SIGNAL_WORDS.some((term) => normalized.includes(term));

        return {
            mode: needsDiscord ? "discord_grounded" : "direct_answer",
            reason: needsDiscord
                ? "Question appears to depend on Discord-specific evidence."
                : "Question can be answered directly without Discord retrieval.",
        };
    }
}
