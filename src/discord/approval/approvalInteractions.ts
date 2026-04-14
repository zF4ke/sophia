import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonInteraction,
    ButtonStyle,
    ContainerBuilder,
    MessageFlags,
    ModalBuilder,
    ModalSubmitInteraction,
    StringSelectMenuInteraction,
    TextDisplayBuilder,
    TextInputBuilder,
    TextInputStyle,
    type Interaction,
} from "discord.js";
import {
    APPROVAL_APPROVE_PREFIX,
    APPROVAL_CORRECT_PREFIX,
    APPROVAL_DENY_PREFIX,
    APPROVAL_STOP_PREFIX,
    BATCH_APPROVE_ALL_PREFIX,
    BATCH_CATEGORY_PREFIX,
    BATCH_CORRECT_PREFIX,
    BATCH_DENY_ALL_PREFIX,
    BATCH_MODAL_PREFIX,
    BATCH_STOP_PREFIX,
    buildResolvedBatchContainer,
    buildResolvedContainer,
    getPendingApproval,
    getPendingBatchApproval,
    resolvePendingApproval,
    resolvePendingBatchApproval,
} from "@/discord/approval/ApprovalGate";
import { getToolDisplay } from "@/tools/registry";
import { SecurityService } from "@/security/SecurityService";
import type { ApprovalRequest, BatchItemDecision } from "@/runtime/contracts";

export const APPROVAL_CONFIRM_APPROVE_PREFIX = "approval:confirm_approve:";
export const APPROVAL_CONFIRM_CANCEL_PREFIX = "approval:confirm_cancel:";
export const APPROVAL_MODAL_PREFIX = "approval:modal:";

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
    // ── Modal submission path (single + batch) ──
    if (interaction.isModalSubmit()) {
        const { customId } = interaction;

        if (customId.startsWith(BATCH_MODAL_PREFIX)) {
            return handleBatchModalSubmit(interaction);
        }

        if (!customId.startsWith(APPROVAL_MODAL_PREFIX)) return false;

        const requestId = customId.slice(APPROVAL_MODAL_PREFIX.length);
        const pending = getPendingApproval(requestId);
        if (!pending) {
            await safeModalReply(interaction, "⏳ Esta aprovação já expirou ou foi resolvida.");
            return true;
        }

        const correctionText = interaction.fields.getTextInputValue("correction_text").trim();
        const { request } = pending;
        resolvePendingApproval(requestId, {
            approved: false,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
            correction: correctionText || undefined,
        });

        const resolvedContainer = buildResolvedContainer(
            request,
            "corrected",
            interaction.user.id,
            undefined,
            correctionText || undefined,
        );
        try {
            if (interaction.isFromMessage()) {
                await interaction.update({ components: [resolvedContainer] });
            } else {
                await interaction.reply({ content: "✏️ Correção registada.", flags: MessageFlags.Ephemeral });
            }
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    // ── Batch category select menu ──
    if (interaction.isStringSelectMenu()) {
        if (!interaction.customId.startsWith(BATCH_CATEGORY_PREFIX)) return false;
        return handleBatchCategorySelect(interaction);
    }

    if (!interaction.isButton()) return false;

    const { customId } = interaction;

    // ── Batch button handling ──
    if (
        customId.startsWith(BATCH_APPROVE_ALL_PREFIX) ||
        customId.startsWith(BATCH_DENY_ALL_PREFIX) ||
        customId.startsWith(BATCH_STOP_PREFIX) ||
        customId.startsWith(BATCH_CORRECT_PREFIX)
    ) {
        return handleBatchButton(interaction);
    }

    const isApprove = customId.startsWith(APPROVAL_APPROVE_PREFIX);
    const isDeny = customId.startsWith(APPROVAL_DENY_PREFIX);
    const isStop = customId.startsWith(APPROVAL_STOP_PREFIX);
    const isCorrect = customId.startsWith(APPROVAL_CORRECT_PREFIX);
    const isConfirmApprove = customId.startsWith(APPROVAL_CONFIRM_APPROVE_PREFIX);
    const isConfirmCancel = customId.startsWith(APPROVAL_CONFIRM_CANCEL_PREFIX);
    if (!isApprove && !isDeny && !isStop && !isCorrect && !isConfirmApprove && !isConfirmCancel) return false;

    const requestId = isApprove
        ? customId.slice(APPROVAL_APPROVE_PREFIX.length)
        : isDeny
            ? customId.slice(APPROVAL_DENY_PREFIX.length)
            : isStop
                ? customId.slice(APPROVAL_STOP_PREFIX.length)
            : isCorrect
                ? customId.slice(APPROVAL_CORRECT_PREFIX.length)
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
        await updateMessage(interaction, request, "denied");
        return true;
    }

    if (isConfirmApprove) {
        const { request } = pending;
        resolvePendingApproval(requestId, {
            approved: true,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        await updateMessage(interaction, request, "approved");
        return true;
    }

    if (isStop) {
        const { request } = pending;
        resolvePendingApproval(requestId, {
            approved: false,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
            haltExecution: true,
        });
        await updateMessage(interaction, request, "stopped");
        return true;
    }

    if (isDeny) {
        const { request } = pending;
        resolvePendingApproval(requestId, {
            approved: false,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        await updateMessage(interaction, request, "denied");
        return true;
    }

    if (isCorrect) {
        await showCorrectionModal(interaction, requestId);
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
    await updateMessage(interaction, request, "approved");
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
        new ButtonBuilder()
            .setCustomId(`${APPROVAL_STOP_PREFIX}${requestId}`)
            .setLabel("Parar execução")
            .setStyle(ButtonStyle.Secondary),
    );

    try {
        await interaction.update({ components: [container, row] });
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
}

async function updateMessage(
    interaction: ButtonInteraction,
    request: ApprovalRequest,
    status: "approved" | "denied" | "stopped",
): Promise<void> {
    const container = buildResolvedContainer(
        request,
        status,
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

async function safeModalReply(interaction: ModalSubmitInteraction, content: string): Promise<void> {
    try {
        await interaction.reply({ content, flags: MessageFlags.Ephemeral });
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
}

async function showCorrectionModal(
    interaction: ButtonInteraction,
    requestId: string,
): Promise<void> {
    const modal = new ModalBuilder()
        .setCustomId(`${APPROVAL_MODAL_PREFIX}${requestId}`)
        .setTitle("Recusar e corrigir");

    const textInput = new TextInputBuilder()
        .setCustomId("correction_text")
        .setLabel("O que deve ser feito de diferente?")
        .setStyle(TextInputStyle.Paragraph)
        .setPlaceholder("Descreve a correção ou instrução alternativa…")
        .setRequired(true)
        .setMaxLength(1000);

    modal.addComponents(
        new ActionRowBuilder<TextInputBuilder>().addComponents(textInput),
    );

    try {
        await interaction.showModal(modal);
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
}

// ── Batch handlers ──

async function handleBatchButton(interaction: ButtonInteraction): Promise<boolean> {
    const { customId } = interaction;

    const isApproveAll = customId.startsWith(BATCH_APPROVE_ALL_PREFIX);
    const isDenyAll = customId.startsWith(BATCH_DENY_ALL_PREFIX);
    const isStop = customId.startsWith(BATCH_STOP_PREFIX);
    const isCorrect = customId.startsWith(BATCH_CORRECT_PREFIX);

    const batchId = isApproveAll
        ? customId.slice(BATCH_APPROVE_ALL_PREFIX.length)
        : isDenyAll
            ? customId.slice(BATCH_DENY_ALL_PREFIX.length)
            : isStop
                ? customId.slice(BATCH_STOP_PREFIX.length)
                : customId.slice(BATCH_CORRECT_PREFIX.length);

    const pending = getPendingBatchApproval(batchId);
    if (!pending) {
        await safeReply(interaction, "⏳ Esta aprovação em lote já expirou ou foi resolvida.");
        return true;
    }

    await SecurityService.initialize();
    if (!SecurityService.isAdmin(interaction.user.id)) {
        await safeReply(interaction, "❌ Apenas administradores podem aceitar ou recusar ações.");
        return true;
    }

    if (isCorrect) {
        const modal = new ModalBuilder()
            .setCustomId(`${BATCH_MODAL_PREFIX}${batchId}`)
            .setTitle("Recusar lote e corrigir");

        const textInput = new TextInputBuilder()
            .setCustomId("correction_text")
            .setLabel("O que deve ser feito de diferente?")
            .setStyle(TextInputStyle.Paragraph)
            .setPlaceholder("Descreve a correção ou instrução alternativa…")
            .setRequired(true)
            .setMaxLength(1000);

        modal.addComponents(
            new ActionRowBuilder<TextInputBuilder>().addComponents(textInput),
        );

        try {
            await interaction.showModal(modal);
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    const allDecisions: Record<string, BatchItemDecision> = {};

    if (isApproveAll) {
        for (const item of pending.request.items) {
            allDecisions[item.toolCallId] = "approved";
        }
        resolvePendingBatchApproval(batchId, {
            decisions: allDecisions,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        try {
            await interaction.update({
                components: [buildResolvedBatchContainer(pending.request, "approved")],
            });
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    if (isDenyAll) {
        for (const item of pending.request.items) {
            allDecisions[item.toolCallId] = "denied";
        }
        resolvePendingBatchApproval(batchId, {
            decisions: allDecisions,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        try {
            await interaction.update({
                components: [buildResolvedBatchContainer(pending.request, "denied")],
            });
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    if (isStop) {
        for (const item of pending.request.items) {
            allDecisions[item.toolCallId] = "denied";
        }
        resolvePendingBatchApproval(batchId, {
            decisions: allDecisions,
            haltExecution: true,
            decidedBy: interaction.user.id,
            decidedAt: Date.now(),
        });
        try {
            await interaction.update({
                components: [buildResolvedBatchContainer(pending.request, "stopped")],
            });
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    return false;
}

async function handleBatchCategorySelect(interaction: StringSelectMenuInteraction): Promise<boolean> {
    const batchId = interaction.customId.slice(BATCH_CATEGORY_PREFIX.length);
    const pending = getPendingBatchApproval(batchId);
    if (!pending) {
        try {
            await interaction.reply({ content: "⏳ Esta aprovação em lote já expirou ou foi resolvida.", flags: MessageFlags.Ephemeral });
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    await SecurityService.initialize();
    if (!SecurityService.isAdmin(interaction.user.id)) {
        try {
            await interaction.reply({ content: "❌ Apenas administradores podem aceitar ou recusar ações.", flags: MessageFlags.Ephemeral });
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    const selectedCategory = interaction.values[0];
    const decisions: Record<string, BatchItemDecision> = {};
    for (const item of pending.request.items) {
        decisions[item.toolCallId] = item.category === selectedCategory ? "approved" : "denied";
    }

    const CATEGORY_LABELS: Record<string, string> = {
        messages: "📨 Mensagens",
        channels: "📁 Canais",
        roles: "🎭 Roles",
        other: "🛠️ Outro",
    };

    resolvePendingBatchApproval(batchId, {
        decisions,
        decidedBy: interaction.user.id,
        decidedAt: Date.now(),
    });

    try {
        await interaction.update({
            components: [buildResolvedBatchContainer(pending.request, "partial", {
                categoryLabel: CATEGORY_LABELS[selectedCategory] || selectedCategory,
            })],
        });
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
    return true;
}

async function handleBatchModalSubmit(interaction: ModalSubmitInteraction): Promise<boolean> {
    const batchId = interaction.customId.slice(BATCH_MODAL_PREFIX.length);
    const pending = getPendingBatchApproval(batchId);
    if (!pending) {
        await safeModalReply(interaction, "⏳ Esta aprovação em lote já expirou ou foi resolvida.");
        return true;
    }

    const correctionText = interaction.fields.getTextInputValue("correction_text").trim();
    const allDenied: Record<string, BatchItemDecision> = {};
    for (const item of pending.request.items) {
        allDenied[item.toolCallId] = "denied";
    }

    resolvePendingBatchApproval(batchId, {
        decisions: allDenied,
        correction: correctionText || undefined,
        decidedBy: interaction.user.id,
        decidedAt: Date.now(),
    });

    try {
        if (interaction.isFromMessage()) {
            await interaction.update({
                components: [buildResolvedBatchContainer(pending.request, "corrected", {
                    correctionText: correctionText || undefined,
                })],
            });
        } else {
            await interaction.reply({ content: "✏️ Correção registada.", flags: MessageFlags.Ephemeral });
        }
    } catch (error) {
        if (!isUnknownInteractionError(error)) throw error;
    }
    return true;
}
