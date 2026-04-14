import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonInteraction,
    ButtonStyle,
    ContainerBuilder,
    MessageFlags,
    TextDisplayBuilder,
    type Interaction,
} from "discord.js";
import {
    APPROVAL_APPROVE_PREFIX,
    APPROVAL_DENY_PREFIX,
    buildResolvedContainer,
    getPendingApproval,
    resolvePendingApproval,
} from "@/discord/approval/ApprovalGate";
import { getToolDisplay } from "@/tools/registry";
import { SecurityService } from "@/security/SecurityService";
import type { ApprovalRequest } from "@/runtime/contracts";

export const APPROVAL_CONFIRM_APPROVE_PREFIX = "approval:confirm_approve:";
export const APPROVAL_CONFIRM_CANCEL_PREFIX = "approval:confirm_cancel:";

function isUnknownInteractionError(error: unknown): boolean {
    return Boolean(
        error &&
            typeof error === "object" &&
            "code" in error &&
            (error as { code?: unknown }).code === 10062
    );
}

export async function handleApprovalInteraction(
    interaction: Interaction
): Promise<boolean> {
    if (!interaction.isButton()) return false;

    const { customId } = interaction;
    const isApprove = customId.startsWith(APPROVAL_APPROVE_PREFIX);
    const isDeny = customId.startsWith(APPROVAL_DENY_PREFIX);
    const isConfirmApprove = customId.startsWith(APPROVAL_CONFIRM_APPROVE_PREFIX);
    const isConfirmCancel = customId.startsWith(APPROVAL_CONFIRM_CANCEL_PREFIX);
    if (!isApprove && !isDeny && !isConfirmApprove && !isConfirmCancel) return false;

    const requestId = isApprove
        ? customId.slice(APPROVAL_APPROVE_PREFIX.length)
        : isDeny
            ? customId.slice(APPROVAL_DENY_PREFIX.length)
            : isConfirmApprove
                ? customId.slice(APPROVAL_CONFIRM_APPROVE_PREFIX.length)
                : customId.slice(APPROVAL_CONFIRM_CANCEL_PREFIX.length);

    const pending = getPendingApproval(requestId);
    if (!pending) {
        await safeReply(interaction, "⏳ Esta aprovação já expirou ou foi resolvida.");
        return true;
    }

    await SecurityService.initialize();
    if (!SecurityService.isAdmin(interaction.user.id)) {
        await safeReply(interaction, "❌ Apenas administradores podem aceitar ou recusar ações.");
        return true;
    }

    if (isConfirmCancel) {
        const { request } = pending;
        resolvePendingApproval(requestId, {
            approved: false,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        await updateMessage(interaction, request, false);
        return true;
    }

    if (isConfirmApprove) {
        const { request } = pending;
        resolvePendingApproval(requestId, {
            approved: true,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        await updateMessage(interaction, request, true);
        return true;
    }

    if (isDeny) {
        const { request } = pending;
        resolvePendingApproval(requestId, {
            approved: false,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        await updateMessage(interaction, request, false);
        return true;
    }

    // Approve path
    if (pending.request.sideEffectLevel === "destructive") {
        await showDestructiveConfirmButtons(interaction, pending.request, requestId);
        return true;
    }

    // Write (non-destructive) — approve immediately
    const { request } = pending;
    resolvePendingApproval(requestId, {
        approved: true,
        decidedBy: interaction.user.id,
        decidedAt: Date.now(),
    });
    await updateMessage(interaction, request, true);
    return true;
}

async function showDestructiveConfirmButtons(
    interaction: ButtonInteraction,
    request: ApprovalRequest,
    requestId: string,
): Promise<void> {
    const tool = getToolDisplay(request.toolName);
    const container = new ContainerBuilder()
        .setAccentColor(0xed4245)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`### ⚠️ Confirmar ${tool.labelPt}`),
            new TextDisplayBuilder().setContent(request.description),
            new TextDisplayBuilder().setContent("Esta ação é destrutiva. Tens a certeza?"),
        );

    const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(`${APPROVAL_CONFIRM_CANCEL_PREFIX}${requestId}`)
            .setLabel("Cancelar")
            .setStyle(ButtonStyle.Secondary),
        new ButtonBuilder()
            .setCustomId(`${APPROVAL_CONFIRM_APPROVE_PREFIX}${requestId}`)
            .setLabel("Confirmar")
            .setStyle(ButtonStyle.Danger),
    );

    try {
        await interaction.update({ components: [container, row] });
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
}

async function updateMessage(interaction: ButtonInteraction, request: ApprovalRequest, approved: boolean): Promise<void> {
    const container = buildResolvedContainer(
        request,
        approved ? "approved" : "denied",
        interaction.user.id,
    );
    try {
        await interaction.update({ components: [container] });
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
}

async function safeReply(interaction: ButtonInteraction, content: string): Promise<void> {
    try {
        await interaction.reply({ content, flags: MessageFlags.Ephemeral });
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
}
