import { AccessPolicy } from "@/security/AccessPolicy";
import {
    ChatInputCommandInteraction,
    MessageFlags,
    SlashCommandBuilder,
} from "discord.js";
import { DebugService } from "@/discord/debug/DebugService";
import { ProgressStatusService, type ProgressStatus } from "@/discord/responding/ProgressStatus";
import { ResponseActivityService } from "@/discord/responding/ResponseActivityIndicator";
import { UIService } from "@/discord/ui/UIService";
import { EMOJIS } from "@/discord/constants";
import { ConversationAdapter } from "@/discord/conversation/ConversationAdapter";
import { Runtime } from "@/runtime/Runtime";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";

export = {
    data: new SlashCommandBuilder()
        .setName("talk")
        .setDescription("Conversa com Sophia")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0, 1)
        .addStringOption((option) =>
            option
                .setName("message")
                .setDescription("Mensagem para Sophia")
                .setRequired(false)
        )
        .addStringOption(option => option.setName("task_id").setDescription("Retoma um dos teus pedidos pausados neste canal"))
        .addBooleanOption(option => option.setName("handoff").setDescription("Transfere um pedido pausado de outro local; exige ephemeral:true"))
        .addStringOption(option => option.setName("approval_mode").setDescription("Política de aprovação para este pedido").addChoices(
            { name: "Pedir aprovação para alterações", value: "ask" }, { name: "Usar a minha autorização configurada", value: "inherit" }))
        .addAttachmentOption(option => option.setName("attachment").setDescription("Imagem ou ficheiro para Sophia analisar"))
        .addBooleanOption((option) =>
            option
                .setName("ephemeral")
                .setDescription("Somente você pode ver a resposta")
                .setRequired(false)
        ),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        const releaseRequest = ActiveRequestTracker.begin();
        let debugSession = null;
        let activityIndicator = null;
        let progressStatus: ProgressStatus | null = null;

        try {
            if (await AccessPolicy.decide(interaction.user.id, interaction.guild, "none") === "deny") {
                await interaction.reply({
                    content: `${EMOJIS.error} Não tens acesso à Sophia neste local.`,
                    flags: MessageFlags.Ephemeral,
                });
                return;
            }

            const message = interaction.options.getString("message") || (interaction.options.getAttachment("attachment") ? "Analisa o anexo." : "Olá.");
            const ephemeral = interaction.options.getBoolean("ephemeral") ?? false;
            await interaction.deferReply({
                flags: ephemeral ? MessageFlags.Ephemeral : undefined,
            });
            activityIndicator = await ResponseActivityService.startForInteraction(interaction);
            await activityIndicator.startThinking();
            debugSession = await DebugService.startForInteraction(interaction, message);
            progressStatus = ephemeral ? ProgressStatusService.startForInteraction(interaction)
                : interaction.channel ? ProgressStatusService.startForChannel(interaction.channel) : null;

            const input = await ConversationAdapter.fromInteraction({
                interaction,
                question: message,
                debugSession,
                progressNotifier: progressStatus
                    ? (summary: string) => progressStatus!.notify(summary)
                    : null,
            });
            input.resumeTaskId = interaction.options.getString("task_id") ?? undefined;
            input.handoffTask = interaction.options.getBoolean("handoff") ?? false;
            input.approvalMode = (interaction.options.getString("approval_mode") ?? undefined) as "ask" | "inherit" | undefined;
            input.onTaskBound = taskId => progressStatus?.bindTask?.(taskId);
            const result = await Runtime.answer(input);
            await activityIndicator.startTyping();
            const formatted = UIService.formatAnswer(result.answer, result.citations);
            const sentMessages = await UIService.sendLongResponse(
                interaction,
                "",
                formatted.trim() ? formatted : "Não consegui gerar uma resposta desta vez. Tenta de novo.",
                ephemeral
            );
            await ConversationAdapter.bindResponseMessages({
                input,
                result,
                sentMessages,
            });
        } catch (error) {
            console.error("Error in talk command:", error);
            await debugSession?.finishError(error);
            await interaction.editReply(
                UIService.formatStatusMessage(
                    EMOJIS.error,
                    error instanceof Error && error.message.startsWith("Não foi possível retomar") ? error.message : "Ocorreu um erro ao conversar.",
                    false
                )
            );
        } finally {
            await progressStatus?.finalize();
            await activityIndicator?.stop();
            releaseRequest();
        }
    },
};
