import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    MessageFlags,
    SeparatorBuilder,
    SeparatorSpacingSize,
    TextDisplayBuilder,
    type SendableChannels,
} from "discord.js";
import type { ApprovalRequest, ApprovalResult } from "@/runtime/contracts";
import { SettingsService } from "@/app/SettingsService";
import { getToolDisplay } from "@/tools/registry";

export const APPROVAL_APPROVE_PREFIX = "approval:approve:";
export const APPROVAL_DENY_PREFIX = "approval:deny:";

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
    status: "approved" | "denied" | "timeout",
    decidedBy?: string,
    timeoutSec?: number,
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
        case "timeout":
            statusLine = `\`⏳ Recusado automaticamente após ${timeoutSec}s\``;
            accentColor = 0x95a5a6;
            statusIcon = "⏳";
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
