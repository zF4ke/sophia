import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    MessageFlags,
    SeparatorBuilder,
    SeparatorSpacingSize,
    StringSelectMenuBuilder,
    TextDisplayBuilder,
    type SendableChannels,
} from "discord.js";
import type {
    ApprovalRequest,
    ApprovalResult,
    BatchApprovalRequest,
    BatchApprovalResult,
    BatchItemDecision,
} from "@/runtime/contracts";
import { SettingsService } from "@/app/SettingsService";
import { getToolDisplay } from "@/tools/registry";

// ── Single-item prefixes ──
export const APPROVAL_APPROVE_PREFIX = "approval:approve:";
export const APPROVAL_DENY_PREFIX = "approval:deny:";
export const APPROVAL_STOP_PREFIX = "approval:stop:";
export const APPROVAL_CORRECT_PREFIX = "approval:correct:";

// ── Batch prefixes ──
export const BATCH_APPROVE_ALL_PREFIX = "batch:approve_all:";
export const BATCH_DENY_ALL_PREFIX = "batch:deny_all:";
export const BATCH_STOP_PREFIX = "batch:stop:";
export const BATCH_CORRECT_PREFIX = "batch:correct:";
export const BATCH_CATEGORY_PREFIX = "batch:category:";
export const BATCH_MODAL_PREFIX = "batch:modal:";

interface PendingApproval {
    request: ApprovalRequest;
    resolve: (result: ApprovalResult) => void;
    timer: ReturnType<typeof setTimeout>;
}

const pendingApprovals = new Map<string, PendingApproval>();

export function getPendingApproval(requestId: string): PendingApproval | undefined {
    return pendingApprovals.get(requestId);
}

export function resolvePendingApproval(requestId: string, result: ApprovalResult): void {
    const entry = pendingApprovals.get(requestId);
    if (!entry) return;
    clearTimeout(entry.timer);
    pendingApprovals.delete(requestId);
    entry.resolve(result);
}

export function buildResolvedContainer(
    request: ApprovalRequest,
    status: "approved" | "denied" | "timeout" | "stopped" | "corrected",
    decidedBy?: string,
    timeoutSec?: number,
    correctionText?: string,
): ContainerBuilder {
    const tool = getToolDisplay(request.toolName);

    let statusLine: string | null = null;
    let accentColor: number;
    let statusIcon: string;
    switch (status) {
        case "approved":
            accentColor = 0x57f287;
            statusIcon = "✅";
            break;
        case "denied":
            accentColor = 0xed4245;
            statusIcon = "❌";
            break;
        case "corrected":
            accentColor = 0xfee75c;
            statusIcon = "✏️";
            statusLine = correctionText
                ? `\`✏️ Recusado com correção:\` ${correctionText}`
                : "`✏️ Recusado com correção`";
            break;
        case "timeout":
            statusLine = `\`⏳ Recusado automaticamente após ${timeoutSec}s\``;
            accentColor = 0x95a5a6;
            statusIcon = "⏳";
            break;
        case "stopped":
            statusLine = "`🛑 Execução interrompida manualmente; a Sophia não fará mais tool calls neste pedido.`";
            accentColor = 0x95a5a6;
            statusIcon = "🛑";
            break;
    }

    const container = new ContainerBuilder()
        .setAccentColor(accentColor)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`### ${statusIcon} ${tool.labelPt}`),
            new TextDisplayBuilder().setContent(request.description),
        );

    if (!statusLine) {
        return container;
    }

    return container
        .addSeparatorComponents(new SeparatorBuilder().setDivider(false).setSpacing(SeparatorSpacingSize.Small))
        .addTextDisplayComponents(new TextDisplayBuilder().setContent(statusLine));
}

export function createApprovalGate(channel: SendableChannels) {
    return async (request: ApprovalRequest): Promise<ApprovalResult> => {
        const settings = SettingsService.load();
        const timeoutMs = settings.runtime.approvalTimeoutMs;
        const tool = getToolDisplay(request.toolName);

        const accentColor = request.sideEffectLevel === "destructive" ? 0xed4245 : 0xfee75c;

        const row = new ActionRowBuilder<ButtonBuilder>().addComponents(
            new ButtonBuilder()
                .setCustomId(`${APPROVAL_APPROVE_PREFIX}${request.requestId}`)
                .setLabel("Aceitar")
                .setStyle(ButtonStyle.Success),
            new ButtonBuilder()
                .setCustomId(`${APPROVAL_DENY_PREFIX}${request.requestId}`)
                .setLabel("Recusar")
                .setStyle(ButtonStyle.Danger),
            new ButtonBuilder()
                .setCustomId(`${APPROVAL_CORRECT_PREFIX}${request.requestId}`)
                .setLabel("Recusar e corrigir")
                .setStyle(ButtonStyle.Primary),
            new ButtonBuilder()
                .setCustomId(`${APPROVAL_STOP_PREFIX}${request.requestId}`)
                .setLabel("Parar execução")
                .setStyle(ButtonStyle.Secondary),
        );

        const container = new ContainerBuilder()
            .setAccentColor(accentColor)
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent(`### ${tool.icon} ${tool.labelPt}`),
                new TextDisplayBuilder().setContent(request.description),
            );

        const sentMessage = await channel.send({
            components: [container, row],
            flags: MessageFlags.IsComponentsV2,
        });

        return new Promise<ApprovalResult>((resolve) => {
            const timer = setTimeout(async () => {
                pendingApprovals.delete(request.requestId);
                resolve({ approved: false, decidedBy: "timeout", decidedAt: Date.now() });

                const timeoutSec = Math.round(timeoutMs / 1000);
                try {
                    await sentMessage.edit({
                        components: [buildResolvedContainer(request, "timeout", undefined, timeoutSec)],
                    });
                } catch { /* message may have been deleted */ }
            }, timeoutMs);

            pendingApprovals.set(request.requestId, { request, resolve, timer });
        });
    };
}

// ── Batch destructive approval ──

export interface PendingBatchApproval {
    request: BatchApprovalRequest;
    resolve: (result: BatchApprovalResult) => void;
    timer: ReturnType<typeof setTimeout>;
}

const pendingBatchApprovals = new Map<string, PendingBatchApproval>();

export function getPendingBatchApproval(batchId: string): PendingBatchApproval | undefined {
    return pendingBatchApprovals.get(batchId);
}

export function resolvePendingBatchApproval(batchId: string, result: BatchApprovalResult): void {
    const entry = pendingBatchApprovals.get(batchId);
    if (!entry) return;
    clearTimeout(entry.timer);
    pendingBatchApprovals.delete(batchId);
    entry.resolve(result);
}

function buildBatchItemList(items: BatchApprovalRequest["items"]): string {
    return items
        .map((item, i) => {
            const tool = getToolDisplay(item.toolName);
            return `${i + 1}. ${tool.icon} ${item.description}`;
        })
        .join("\n");
}

export function buildResolvedBatchContainer(
    request: BatchApprovalRequest,
    status: "approved" | "denied" | "partial" | "timeout" | "stopped" | "corrected",
    options?: { decidedBy?: string; timeoutSec?: number; correctionText?: string; categoryLabel?: string },
): ContainerBuilder {
    let statusLine: string;
    let accentColor: number;
    let statusIcon: string;
    switch (status) {
        case "approved":
            accentColor = 0x57f287;
            statusIcon = "✅";
            statusLine = "`✅ Batch aprovado`";
            break;
        case "partial":
            accentColor = 0xfee75c;
            statusIcon = "✅";
            statusLine = options?.categoryLabel
                ? `\`✅ Categoria aprovada: ${options.categoryLabel}\``
                : "`✅ Aprovação parcial`";
            break;
        case "denied":
            accentColor = 0xed4245;
            statusIcon = "❌";
            statusLine = "`❌ Batch recusado`";
            break;
        case "corrected":
            accentColor = 0xfee75c;
            statusIcon = "✏️";
            statusLine = options?.correctionText
                ? `\`✏️ Recusado com correção:\` ${options.correctionText}`
                : "`✏️ Recusado com correção`";
            break;
        case "timeout":
            accentColor = 0x95a5a6;
            statusIcon = "⏳";
            statusLine = `\`⏳ Recusado automaticamente após ${options?.timeoutSec ?? "?"}s\``;
            break;
        case "stopped":
            accentColor = 0x95a5a6;
            statusIcon = "🛑";
            statusLine = "`🛑 Execução interrompida manualmente`";
            break;
    }

    return new ContainerBuilder()
        .setAccentColor(accentColor)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`### ${statusIcon} Aprovação batch (${request.items.length} ações)`),
            new TextDisplayBuilder().setContent(buildBatchItemList(request.items)),
        )
        .addSeparatorComponents(new SeparatorBuilder().setDivider(false).setSpacing(SeparatorSpacingSize.Small))
        .addTextDisplayComponents(new TextDisplayBuilder().setContent(statusLine));
}

export function createBatchApprovalGate(channel: SendableChannels) {
    return async (request: BatchApprovalRequest): Promise<BatchApprovalResult> => {
        const settings = SettingsService.load();
        const timeoutMs = settings.runtime.approvalTimeoutMs;

        const itemList = buildBatchItemList(request.items);
        const container = new ContainerBuilder()
            .setAccentColor(0xed4245)
            .addTextDisplayComponents(
                new TextDisplayBuilder().setContent(`### 🗑️ Aprovação batch (${request.items.length} ações)`),
                new TextDisplayBuilder().setContent(itemList),
            );

        const buttonRow = new ActionRowBuilder<ButtonBuilder>().addComponents(
            new ButtonBuilder()
                .setCustomId(`${BATCH_APPROVE_ALL_PREFIX}${request.batchId}`)
                .setLabel("Aprovar tudo")
                .setStyle(ButtonStyle.Success),
            new ButtonBuilder()
                .setCustomId(`${BATCH_DENY_ALL_PREFIX}${request.batchId}`)
                .setLabel("Recusar tudo")
                .setStyle(ButtonStyle.Danger),
            new ButtonBuilder()
                .setCustomId(`${BATCH_CORRECT_PREFIX}${request.batchId}`)
                .setLabel("Recusar e corrigir")
                .setStyle(ButtonStyle.Primary),
            new ButtonBuilder()
                .setCustomId(`${BATCH_STOP_PREFIX}${request.batchId}`)
                .setLabel("Parar execução")
                .setStyle(ButtonStyle.Secondary),
        );

        // Only show category select if there are 2+ distinct Discord Categories
        const categoryMap = new Map<string, string>(); // id → name
        for (const item of request.items) {
            if (item.targetCategory) {
                categoryMap.set(item.targetCategory.id, item.targetCategory.name);
            }
        }
        const components: (ContainerBuilder | ActionRowBuilder<ButtonBuilder | StringSelectMenuBuilder>)[] = [container, buttonRow];
        if (categoryMap.size > 1) {
            const categoryRow = new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                new StringSelectMenuBuilder()
                    .setCustomId(`${BATCH_CATEGORY_PREFIX}${request.batchId}`)
                    .setPlaceholder("Aprovar por categoria…")
                    .addOptions(
                        [...categoryMap.entries()].map(([catId, catName]) => ({
                            label: `📁 ${catName}`,
                            value: catId,
                            description: `Aprovar apenas ações em ${catName}`,
                        })),
                    ),
            );
            components.push(categoryRow);
        }

        const sentMessage = await channel.send({
            components,
            flags: MessageFlags.IsComponentsV2,
        });

        return new Promise<BatchApprovalResult>((resolve) => {
            const allDenied: Record<string, BatchItemDecision> = {};
            for (const item of request.items) {
                allDenied[item.toolCallId] = "denied";
            }

            const timer = setTimeout(async () => {
                pendingBatchApprovals.delete(request.batchId);
                resolve({
                    decisions: allDenied,
                    decidedBy: "timeout",
                    decidedAt: Date.now(),
                });
                const timeoutSec = Math.round(timeoutMs / 1000);
                try {
                    await sentMessage.edit({
                        components: [buildResolvedBatchContainer(request, "timeout", { timeoutSec })],
                    });
                } catch { /* message may have been deleted */ }
            }, timeoutMs);

            pendingBatchApprovals.set(request.batchId, { request, resolve, timer });
        });
    };
}
