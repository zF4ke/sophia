import {
    ActionRowBuilder,
    ButtonInteraction,
    ModalBuilder,
    TextInputBuilder,
    TextInputStyle,
} from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import { getAccessLimitsModalId } from "@/discord/commands/system/access/panelIds";
import { refreshPanel } from "@/discord/commands/system/access/handlers/panelRefresh";
import type { AccessHandlerContext } from "@/discord/commands/system/access/handlers/types";
import { isAccessPolicyTarget } from "@/security/policyTargets";

function formatTargetLabel(commandName: string): string {
    const labels = SecurityService.getPolicyTargetLabels();
    return isAccessPolicyTarget(commandName) ? labels[commandName] : `/${commandName}`;
}

function buildLimitsModal(
    commandName: string,
    defaults: { default: number; moderator: number; admin: number }
): ModalBuilder {
    return new ModalBuilder()
        .setCustomId(getAccessLimitsModalId(commandName))
        .setTitle(`Limites: ${formatTargetLabel(commandName)}`)
        .addComponents(
            new ActionRowBuilder<TextInputBuilder>().addComponents(
                new TextInputBuilder()
                    .setCustomId("default_limit")
                    .setLabel("Limite padrão")
                    .setStyle(TextInputStyle.Short)
                    .setRequired(true)
                    .setValue(String(defaults.default))
            ),
            new ActionRowBuilder<TextInputBuilder>().addComponents(
                new TextInputBuilder()
                    .setCustomId("moderator_limit")
                    .setLabel("Limite de moderador")
                    .setStyle(TextInputStyle.Short)
                    .setRequired(true)
                    .setValue(String(defaults.moderator))
            ),
            new ActionRowBuilder<TextInputBuilder>().addComponents(
                new TextInputBuilder()
                    .setCustomId("admin_limit")
                    .setLabel("Limite de admin")
                    .setStyle(TextInputStyle.Short)
                    .setRequired(true)
                    .setValue(String(defaults.admin))
            )
        );
}

export async function handleAccessButton(
    interaction: ButtonInteraction,
    context: AccessHandlerContext
): Promise<boolean> {
    const parts = interaction.customId.split(":");

    if (parts[1] === "view") {
        const view = ["overview", "admins", "moderators", "commands", "grants"].includes(parts[2]) ? parts[2] : "overview";
        await refreshPanel(interaction, context, {
            view: view as import("../panelTypes").AccessPanelView,
            page: Number(parts[3]) || 0,
        });
        return true;
    }

    if (parts[1] === "commands" && parts[2] === "visibility") {
        const commandName = parts[3];
        const isPublic = parts[4] === "public";
        await SecurityService.setCommandVisibility(commandName, isPublic);
        await refreshPanel(interaction, context, {
            view: "commands",
            selectedCommand: commandName,
            notice: `${formatTargetLabel(commandName)} agora está ${isPublic ? "permitido" : "bloqueado"}.`,
        });
        return true;
    }

    if (parts[1] === "commands" && parts[2] === "limits") {
        const commandName = parts[3];
        const configs = await SecurityService.getCommandConfigs();
        const config = configs.get(commandName);
        await interaction.showModal(
            buildLimitsModal(commandName, {
                default: config?.rateLimits.default ?? 5,
                moderator: config?.rateLimits.moderator ?? 7,
                admin: config?.rateLimits.admin ?? 10,
            })
        );
        return true;
    }

    return false;
}
