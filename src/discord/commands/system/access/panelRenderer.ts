import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    SeparatorBuilder,
    SeparatorSpacingSize,
    StringSelectMenuBuilder,
    StringSelectMenuOptionBuilder,
    TextDisplayBuilder,
    UserSelectMenuBuilder,
} from "discord.js";
import {
    ACCESS_ADMIN_ADD_ID,
    ACCESS_ADMIN_REMOVE_ID,
    ACCESS_COMMAND_SELECT_ID,
    ACCESS_MODERATOR_ADD_ID,
    ACCESS_MODERATOR_REMOVE_ID,
    ACCESS_VIEW_BUTTONS,
    getAccessLimitsButtonId,
    getAccessVisibilityId,
} from "./panelIds";
import { fetchUserLabel, type AccessPanelData } from "./panelData";
import type { AccessPanelState, AccessPanelView } from "./panelTypes";
import { ACCESS_POLICY_TARGET_DESCRIPTIONS, type AccessPolicyTarget } from "@/security/policyTargets";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";
import type { CommandConfig } from "@/security/types";

type AccessRow = ActionRowBuilder<
    ButtonBuilder | StringSelectMenuBuilder | UserSelectMenuBuilder
>;

function buildViewButtonsRow(currentView: AccessPanelView): ActionRowBuilder<ButtonBuilder> {
    return new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(ACCESS_VIEW_BUTTONS.overview)
            .setLabel("Visão geral")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(currentView === "overview"),
        new ButtonBuilder()
            .setCustomId(ACCESS_VIEW_BUTTONS.admins)
            .setLabel("Admins")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(currentView === "admins"),
        new ButtonBuilder()
            .setCustomId(ACCESS_VIEW_BUTTONS.moderators)
            .setLabel("Moderadores")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(currentView === "moderators"),
        new ButtonBuilder()
            .setCustomId(ACCESS_VIEW_BUTTONS.commands)
            .setLabel("Comandos")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(currentView === "commands")
    );
}

function getCommandConfig(
    panelData: AccessPanelData,
    commandName: string
): CommandConfig {
    return (
        panelData.commandConfigs.get(commandName) ?? {
            isPublic: false,
            rateLimits: {
                default: 5,
                moderator: 7,
                admin: 10,
            },
        }
    );
}

async function buildOverviewBody(
    client: BotClient,
    panelData: AccessPanelData
): Promise<string> {
    const publicCommands = panelData.commandNames.filter((commandName) =>
        getCommandConfig(panelData, commandName).isPublic
    ).length;
    const enabledTriggers = panelData.triggerNames.filter((triggerName) =>
        getCommandConfig(panelData, triggerName).isPublic
    ).length;
    const sampleAdmins = await Promise.all(
        panelData.admins.slice(0, 3).map((admin) => fetchUserLabel(client, admin.userId))
    );

    return [
        `**Administradores:** ${panelData.admins.length}`,
        `**Moderadores:** ${panelData.moderators.length}`,
        `**Comandos públicos:** ${publicCommands}/${panelData.commandNames.length}`,
        `**Triggers permitidos:** ${enabledTriggers}/${panelData.triggerNames.length}`,
        `**Admins recentes:** ${sampleAdmins.length ? sampleAdmins.join(", ") : "nenhum"}`,
    ].join("\n");
}

async function buildUserListBody(
    client: BotClient,
    items: { userId: string; addedBy: string }[],
    emptyMessage: string
): Promise<string> {
    if (!items.length) {
        return emptyMessage;
    }

    const lines = await Promise.all(
        items.slice(0, 10).map(async (item) => {
            const label = await fetchUserLabel(client, item.userId);
            return `- ${label}${item.addedBy === "system" ? " · fixo" : ""}`;
        })
    );

    return lines.join("\n");
}

function buildCommandBody(
    panelData: AccessPanelData,
    commandName: string
): string {
    const config = getCommandConfig(panelData, commandName);
    return [
        `**Comando:** \`/${commandName}\``,
        `**Visibilidade:** ${config.isPublic ? "Público" : "Privado"}`,
        `**Limites:** padrão ${config.rateLimits.default} · moderador ${config.rateLimits.moderator} · admin ${config.rateLimits.admin}`,
    ].join("\n");
}

function buildTriggerBody(
    panelData: AccessPanelData,
    triggerName: AccessPolicyTarget,
): string {
    const config = getCommandConfig(panelData, triggerName);
    const labels = SecurityService.getPolicyTargetLabels();
    return [
        `**Trigger:** \`${labels[triggerName]}\``,
        `**Estado:** ${config.isPublic ? "Permitido" : "Bloqueado"}`,
        `**Descrição:** ${ACCESS_POLICY_TARGET_DESCRIPTIONS[triggerName]}`,
        `**Limites:** padrão ${config.rateLimits.default} · moderador ${config.rateLimits.moderator} · admin ${config.rateLimits.admin}`,
    ].join("\n");
}

function buildCommandSelectRow(
    panelData: AccessPanelData,
    selectedCommand: string
): ActionRowBuilder<StringSelectMenuBuilder> {
    const labels = SecurityService.getPolicyTargetLabels();
    const menu = new StringSelectMenuBuilder()
        .setCustomId(ACCESS_COMMAND_SELECT_ID)
        .setPlaceholder("Escolher comando ou trigger")
        .addOptions(
            [...panelData.triggerNames, ...panelData.commandNames].slice(0, 25).map((commandName) =>
                new StringSelectMenuOptionBuilder()
                    .setLabel(
                        panelData.triggerNames.includes(commandName as AccessPolicyTarget)
                            ? labels[commandName as AccessPolicyTarget]
                            : `/${commandName}`
                    )
                    .setValue(commandName)
                    .setDefault(commandName === selectedCommand)
            )
        );

    return new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(menu);
}

function buildCommandActionsRow(
    panelData: AccessPanelData,
    selectedCommand: string
): ActionRowBuilder<ButtonBuilder> {
    const config = getCommandConfig(panelData, selectedCommand);

    return new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(getAccessVisibilityId(selectedCommand, true))
            .setLabel("Tornar público")
            .setStyle(ButtonStyle.Success)
            .setDisabled(config.isPublic),
        new ButtonBuilder()
            .setCustomId(getAccessVisibilityId(selectedCommand, false))
            .setLabel("Tornar privado")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(!config.isPublic),
        new ButtonBuilder()
            .setCustomId(getAccessLimitsButtonId(selectedCommand))
            .setLabel("Editar limites")
            .setStyle(ButtonStyle.Primary)
    );
}

function buildUserSelectRow(
    customId: string,
    placeholder: string
): ActionRowBuilder<UserSelectMenuBuilder> {
    return new ActionRowBuilder<UserSelectMenuBuilder>().addComponents(
        new UserSelectMenuBuilder()
            .setCustomId(customId)
            .setPlaceholder(placeholder)
            .setMinValues(1)
            .setMaxValues(1)
    );
}

async function buildRemovalRow(
    client: BotClient,
    customId: string,
    placeholder: string,
    items: { userId: string; addedBy: string }[]
): Promise<ActionRowBuilder<StringSelectMenuBuilder>> {
    const removableItems = items.filter((item) => item.addedBy !== "system").slice(0, 25);
    const menu = new StringSelectMenuBuilder()
        .setCustomId(customId)
        .setPlaceholder(placeholder)
        .setMinValues(1)
        .setMaxValues(1)
        .setDisabled(removableItems.length === 0);

    if (removableItems.length === 0) {
        menu.addOptions(
            new StringSelectMenuOptionBuilder()
                .setLabel("Nenhum usuário removível")
                .setValue("none")
                .setDefault(true)
        );
    } else {
        const options = await Promise.all(
            removableItems.map(async (item) =>
                new StringSelectMenuOptionBuilder()
                    .setLabel(await fetchUserLabel(client, item.userId))
                    .setValue(item.userId)
            )
        );

        menu.addOptions(options);
    }

    return new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(menu);
}

export async function buildAccessPanel(
    client: BotClient,
    panelData: AccessPanelData,
    state: AccessPanelState
): Promise<{ components: [ContainerBuilder, ...AccessRow[]] }> {
    const policyNames = [...panelData.triggerNames, ...panelData.commandNames];
    const selectedCommand =
        state.selectedCommand && policyNames.includes(state.selectedCommand)
            ? state.selectedCommand
            : policyNames[0];
    const titleByView: Record<AccessPanelView, string> = {
        overview: "## Painel de acesso",
        admins: "## Administradores",
        moderators: "## Moderadores",
        commands: "## Comandos",
    };
    const rows: AccessRow[] = [buildViewButtonsRow(state.view)];
    let body = "";

    if (state.view === "overview") {
        body = await buildOverviewBody(client, panelData);
    }

    if (state.view === "admins") {
        body = await buildUserListBody(
            client,
            panelData.admins,
            "Nenhum administrador cadastrado."
        );
        rows.push(buildUserSelectRow(ACCESS_ADMIN_ADD_ID, "Adicionar administrador"));
        rows.push(
            await buildRemovalRow(
                client,
                ACCESS_ADMIN_REMOVE_ID,
                "Remover administrador",
                panelData.admins
            )
        );
    }

    if (state.view === "moderators") {
        body = await buildUserListBody(
            client,
            panelData.moderators,
            "Nenhum moderador cadastrado."
        );
        rows.push(buildUserSelectRow(ACCESS_MODERATOR_ADD_ID, "Adicionar moderador"));
        rows.push(
            await buildRemovalRow(
                client,
                ACCESS_MODERATOR_REMOVE_ID,
                "Remover moderador",
                panelData.moderators
            )
        );
    }

    if (state.view === "commands") {
        body = selectedCommand
            ? panelData.triggerNames.includes(selectedCommand as AccessPolicyTarget)
                ? buildTriggerBody(panelData, selectedCommand as AccessPolicyTarget)
                : buildCommandBody(panelData, selectedCommand)
            : "Nenhum comando carregado.";

        if (selectedCommand) {
            rows.push(buildCommandSelectRow(panelData, selectedCommand));
            rows.push(buildCommandActionsRow(panelData, selectedCommand));
        }
    }

    const container = new ContainerBuilder()
        // .setAccentColor(0x9aa7ff)
        // change to a red
        .setAccentColor(0xff6961)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(titleByView[state.view]),
            new TextDisplayBuilder().setContent(
                "Gerencie acesso e políticas do bot"
            )
        );

    if (state.notice) {
        container.addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`**Atualização:** ${state.notice}`)
        );
    }

    container
        .addSeparatorComponents(
            new SeparatorBuilder()
                .setDivider(true)
                .setSpacing(SeparatorSpacingSize.Small)
        )
        .addTextDisplayComponents(new TextDisplayBuilder().setContent(body));

    return {
        components: [container, ...rows],
    };
}
