import {
    ChatInputCommandInteraction,
    Message,
    MessageFlags,
    TextChannel,
    ThreadChannel,
} from "discord.js";
import { DebugModeService } from "@/discord/debug/DebugModeService";
import { DebugSession } from "@/discord/debug/DebugSession";
import { renderDebugTrace } from "@/discord/debug/renderDebugTrace";
import type { DebugSessionReporter, DebugTraceState } from "@/discord/debug/types";

function buildInitialState(question: string): DebugTraceState {
    return {
        questionPreview: question.replace(/\s+/g, " ").trim().slice(0, 140) || "Sem texto.",
        status: "running",
        stage: "Iniciando",
        mode: null,
        controllerDecision: null,
        toolNames: [],
        groundingSummary: null,
        groundingDecisionMode: null,
        groundedAnswerMode: null,
        contextCacheStatus: "none",
        recentEvents: ["Iniciado"],
        startedAt: Date.now(),
    };
}

async function sendDebugMessage(
    channel: TextChannel | ThreadChannel,
    question: string
): Promise<Message | null> {
    try {
        return await channel.send({
            components: [renderDebugTrace(buildInitialState(question))],
            flags: MessageFlags.IsComponentsV2,
        });
    } catch (error) {
        console.error("Error creating debug trace message:", error);
        return null;
    }
}

export class DebugService {
    public static isEnabled(): boolean {
        return DebugModeService.isEnabled();
    }

    public static async startForInteraction(
        interaction: ChatInputCommandInteraction,
        question: string
    ): Promise<DebugSessionReporter | null> {
        const channel = interaction.channel;
        if (!DebugModeService.isEnabled()) {
            return null;
        }
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) {
            return null;
        }

        const message = await sendDebugMessage(channel, question);
        return message ? new DebugSession(message, question) : null;
    }

    public static async startForMessage(
        message: Message,
        question: string
    ): Promise<DebugSessionReporter | null> {
        const channel = message.channel;
        if (!DebugModeService.isEnabled()) {
            return null;
        }
        if (!(channel instanceof TextChannel) && !(channel instanceof ThreadChannel)) {
            return null;
        }

        const debugMessage = await sendDebugMessage(channel, question);
        return debugMessage ? new DebugSession(debugMessage, question) : null;
    }
}
