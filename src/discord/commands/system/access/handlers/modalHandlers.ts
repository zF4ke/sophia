import { ModalSubmitInteraction } from "discord.js";
import { SecurityService } from "@/security/SecurityService";
import { refreshPanel } from "@/discord/commands/system/access/handlers/panelRefresh";
import type { AccessHandlerContext } from "@/discord/commands/system/access/handlers/types";
import { isAccessPolicyTarget } from "@/security/policyTargets";

function formatTargetLabel(commandName: string): string {
    const labels = SecurityService.getPolicyTargetLabels();
    return isAccessPolicyTarget(commandName) ? labels[commandName] : `/${commandName}`;
}

export async function handleAccessModal(
    interaction: ModalSubmitInteraction,
    context: AccessHandlerContext
): Promise<boolean> {
    const parts = interaction.customId.split(":");
    if (!(parts[1] === "commands" && parts[2] === "limits_modal")) {
        return false;
    }

    const commandName = parts[3];
    const defaultLimit = Number(interaction.fields.getTextInputValue("default_limit"));
    const moderatorLimit = Number(interaction.fields.getTextInputValue("moderator_limit"));
    const adminLimit = Number(interaction.fields.getTextInputValue("admin_limit"));

    if (
        !Number.isInteger(defaultLimit) ||
        !Number.isInteger(moderatorLimit) ||
        !Number.isInteger(adminLimit) ||
        defaultLimit < 0 ||
        moderatorLimit < 0 ||
        adminLimit < 0
    ) {
        await interaction.reply({
            content: "❌ Os limites precisam ser números inteiros maiores ou iguais a 0.",
            ephemeral: true,
        });
        return true;
    }

    await SecurityService.setCommandRateLimit(
        commandName,
        defaultLimit,
        adminLimit,
        moderatorLimit
    );
    await refreshPanel(interaction, context, {
        view: "commands",
        selectedCommand: commandName,
        notice: `Limites de ${formatTargetLabel(commandName)} atualizados.`,
    });
    return true;
}
