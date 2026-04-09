import type { ChatInputCommandInteraction, TextChannel } from "discord.js";
import { AgentOrchestrator } from "@/services/agent/AgentOrchestrator";

export class ChattingService {
    public static async generateChatResponse(
        channel: TextChannel,
        prompt: string,
        options: {
            userName?: string;
            interaction?: ChatInputCommandInteraction;
            additionalContext?: string;
        } = {}
    ): Promise<string> {
        const result = await AgentOrchestrator.answerQuestion({
            question: options.additionalContext
                ? `${prompt}\n\n${options.additionalContext}`
                : prompt,
            user: options.interaction?.user || channel.client.user!,
            guild: channel.guild,
            currentChannelId: channel.id,
        });

        return result.answer;
    }
}
