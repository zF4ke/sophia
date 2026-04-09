import { ModelGateway } from "@/services/ai/ModelGateway";
import { PromptRegistry } from "@/services/prompt/PromptRegistry";
import { AIBaseService } from "@/services/ai/AIBaseService";

export class ResponseGenerationService extends AIBaseService {
    public static async generateCustomResponse(
        promptTemplate: string,
        params: Record<string, string>,
        _temperature = 0.2
    ): Promise<string> {
        let prompt = promptTemplate;
        for (const [key, value] of Object.entries(params)) {
            prompt = prompt.split(`{{${key}}}`).join(value);
        }
        return ModelGateway.generateText([{ role: "user", content: prompt }]);
    }

    public static async generateConversationResponse(
        question: string,
        context: string,
        additionalInstructions = ""
    ): Promise<string> {
        try {
            return await ModelGateway.generateText([
                { role: "system", content: PromptRegistry.load("system/base") },
                {
                    role: "user",
                    content: `${additionalInstructions}\n\nContexto:\n${context}\n\nPergunta:\n${question}`.trim(),
                },
            ]);
        } catch (error) {
            return this.handleError(error, "generateConversationResponse");
        }
    }

    public static async generateContextualResponse(
        question: string,
        context: string,
        _options: {
            additionalInstructions?: string;
            extremelyLongAnswer?: boolean;
        } = {}
    ): Promise<string> {
        return this.generateConversationResponse(question, context);
    }

    public static async generateWebSearchResponse(
        question: string,
        context = "",
        additionalInstructions = ""
    ): Promise<string> {
        return this.generateConversationResponse(question, context, additionalInstructions);
    }
}
