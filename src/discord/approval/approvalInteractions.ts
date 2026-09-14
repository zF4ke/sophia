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
import { DISCORD_TOOL_NAMES, type DiscordToolName } from "@/shared/discordTools";
import { AccessPolicy } from "@/security/AccessPolicy";
import type { ApprovalRequest, BatchApprovalRequest, BatchItemDecision } from "@/runtime/contracts";

export const APPROVAL_CONFIRM_APPROVE_PREFIX = "approval:confirm_approve:";
export const APPROVAL_CONFIRM_CANCEL_PREFIX = "approval:confirm_cancel:";
export const APPROVAL_MODAL_PREFIX = "approval:modal:";

export const BATCH_CONFIRM_APPROVE_PREFIX = "batch:confirm_approve:";
export const BATCH_CONFIRM_CANCEL_PREFIX = "batch:confirm_cancel:";

async function canDecide(interaction: Interaction, request?: ApprovalRequest | BatchApprovalRequest): Promise<boolean> {
    await SecurityService.initialize();
    if (!request || request.requesterId !== interaction.user.id) return false;
    if ("items" in request) {
        for (const item of request.items) if (!DISCORD_TOOL_NAMES.includes(item.toolName as DiscordToolName) || await AccessPolicy.decide(interaction.user.id, interaction.guild, "destructive", item.toolName as DiscordToolName) === "deny") return false;
        return true;
    }
    return DISCORD_TOOL_NAMES.includes(request.toolName as DiscordToolName) && await AccessPolicy.decide(interaction.user.id, interaction.guild, request.sideEffectLevel, request.toolName as DiscordToolName) !== "deny";
}

function buildBatchDecisions(
    request: BatchApprovalRequest,
    decision: BatchItemDecision
): Record<string, BatchItemDecision> {
    const out: Record<string, BatchItemDecision> = {};
    for (const item of request.items) {
        out[item.toolCallId] = decision;
    }
    return out;
}

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

        if (customId.startsWith(BATCH_MODAL_PREFIX) || customId.startsWith(APPROVAL_MODAL_PREFIX)) {
            const request = customId.startsWith(BATCH_MODAL_PREFIX)
                ? getPendingBatchApproval(customId.slice(BATCH_MODAL_PREFIX.length))?.request
                : getPendingApproval(customId.slice(APPROVAL_MODAL_PREFIX.length))?.request;
            if (!await canDecide(interaction, request)) {
                await safeModalReply(interaction, "Não tens permissão para alterar esta aprovação.");
                return true;
            }
        }

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
        customId.startsWith(BATCH_CORRECT_PREFIX) ||
        customId.startsWith(BATCH_CONFIRM_APPROVE_PREFIX) ||
        customId.startsWith(BATCH_CONFIRM_CANCEL_PREFIX)
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
    if (!await canDecide(interaction, pending.request)) {
        await safeReply(interaction, "Não tens permissão para decidir esta aprovação.");
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

async function showBatchDestructiveConfirm(
    interaction: ButtonInteraction,
    request: BatchApprovalRequest,
    batchId: string,
): Promise<void> {
    const itemList = request.items
        .map((item, i) => {
            const tool = getToolDisplay(item.toolName);
            return `${i + 1}. ${tool.icon} ${item.description}`;
        })
        .join("\n");

    const container = new ContainerBuilder()
        .setAccentColor(0xed4245)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`### ⚠️ Confirmar ${request.items.length} ações destrutivas`),
            new TextDisplayBuilder().setContent(itemList),
            new TextDisplayBuilder().setContent("Estas ações são destrutivas e irreversíveis. Tens a certeza?"),
        );

    const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(`${BATCH_CONFIRM_CANCEL_PREFIX}${batchId}`)
            .setLabel("Cancelar")
            .setStyle(ButtonStyle.Secondary),
        new ButtonBuilder()
            .setCustomId(`${BATCH_CONFIRM_APPROVE_PREFIX}${batchId}`)
            .setLabel(`Confirmar ${request.items.length} ações`)
            .setStyle(ButtonStyle.Danger),
        new ButtonBuilder()
            .setCustomId(`${BATCH_STOP_PREFIX}${batchId}`)
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
    const isConfirmApprove = customId.startsWith(BATCH_CONFIRM_APPROVE_PREFIX);
    const isConfirmCancel = customId.startsWith(BATCH_CONFIRM_CANCEL_PREFIX);

    const batchId = isApproveAll
        ? customId.slice(BATCH_APPROVE_ALL_PREFIX.length)
        : isDenyAll
            ? customId.slice(BATCH_DENY_ALL_PREFIX.length)
            : isStop
                ? customId.slice(BATCH_STOP_PREFIX.length)
                : isCorrect
                    ? customId.slice(BATCH_CORRECT_PREFIX.length)
                    : isConfirmApprove
                        ? customId.slice(BATCH_CONFIRM_APPROVE_PREFIX.length)
                        : customId.slice(BATCH_CONFIRM_CANCEL_PREFIX.length);

    const pending = getPendingBatchApproval(batchId);
    if (!pending) {
        await safeReply(interaction, "⏳ Esta aprovação em massa já expirou ou foi resolvida.");
        return true;
    }

    await SecurityService.initialize();
    if (!await canDecide(interaction, pending.request)) {
        await safeReply(interaction, "Não tens permissão para decidir esta aprovação.");
        return true;
    }

    // ── Batch confirm approve (after destructive confirmation) ──
    if (isConfirmApprove) {
        const allDecisions = buildBatchDecisions(pending.request, "approved");
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

    // ── Batch confirm cancel (back out of destructive confirmation) ──
    if (isConfirmCancel) {
        const allDecisions = buildBatchDecisions(pending.request, "denied");
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

    if (isCorrect) {
        const modal = new ModalBuilder()
            .setCustomId(`${BATCH_MODAL_PREFIX}${batchId}`)
            .setTitle("Recusar em massa e corrigir");

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

    if (isApproveAll) {
        // Show destructive confirmation before batch-approving
        await showBatchDestructiveConfirm(interaction, pending.request, batchId);
        return true;
    }

    if (isDenyAll) {
        const allDecisions = buildBatchDecisions(pending.request, "denied");
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
        const allDecisions = buildBatchDecisions(pending.request, "denied");
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
            await interaction.reply({ content: "⏳ Esta aprovação em massa já expirou ou foi resolvida.", flags: MessageFlags.Ephemeral });
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    await SecurityService.initialize();
    if (!await canDecide(interaction, pending.request)) {
        try {
            await interaction.reply({ content: "Não tens permissão para decidir esta aprovação.", flags: MessageFlags.Ephemeral });
        } catch (error) {
            if (!isUnknownInteractionError(error)) throw error;
        }
        return true;
    }

    const selectedCategoryId = interaction.values[0];
    const decisions: Record<string, BatchItemDecision> = {};
    let categoryName = selectedCategoryId;
    for (const item of pending.request.items) {
        if (item.targetCategory?.id === selectedCategoryId) {
            decisions[item.toolCallId] = "approved";
            categoryName = item.targetCategory.name;
        } else {
            decisions[item.toolCallId] = "denied";
        }
    }

    resolvePendingBatchApproval(batchId, {
        decisions,
        decidedBy: interaction.user.id,
        decidedAt: Date.now(),
    });

    try {
        await interaction.update({
            components: [buildResolvedBatchContainer(pending.request, "partial", {
                categoryLabel: `📁 ${categoryName}`,
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
        await safeModalReply(interaction, "⏳ Esta aprovação em massa já expirou ou foi resolvida.");
        return true;
    }

    const correctionText = interaction.fields.getTextInputValue("correction_text").trim();
    const allDenied = buildBatchDecisions(pending.request, "denied");

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
