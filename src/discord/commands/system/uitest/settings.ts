import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    MessageFlags,
    SlashCommandSubcommandBuilder,
    StringSelectMenuBuilder,
    TextDisplayBuilder,
    type ButtonInteraction,
    type ChatInputCommandInteraction,
    type StringSelectMenuInteraction,
} from "discord.js";
import { readModelProfiles, resolveModelProfileName } from "@/app/modelProfiles";
import { SettingsService, type BotSettings } from "@/app/SettingsService";
import type { BotClient } from "@/shared/appTypes";
import { guardUiTestAdmin } from "@/discord/commands/system/uitest/shared";
import type { UiTestSubcommandModule } from "@/discord/commands/system/uitest/types";

const UITEST_SETTINGS_PREFIX = "uitest:settings";
const UITEST_CONTAINER_ACCENT = 0xd1d5db;

type MockSettingsTab = "model" | "runtime";
type MockRuntimeKey =
    | "maxPriorTurns"
    | "maxChannelMessages"
    | "maxToolRunsContext"
    | "maxEvidenceSlice"
    | "retrievalHistoryLimit"
    | "retrievalContextWindow"
    | "maxToolCalls"
    | "maxRepeatedCallSignature"
    | "maxLatencyBudgetMs"
    | "escalationFetchLimit"
    | "approvalTimeoutMs";

type UiTestSettingsState = {
    ownerUserId: string;
    settings: BotSettings;
    tab: MockSettingsTab;
};

type MockRuntimeMeta = {
    label: string;
    shortDescription: string;
    longDescription: string;
    presets: number[];
};

const uiTestSettingsStates = new Map<string, UiTestSettingsState>();

const MOCK_RUNTIME_META: Record<MockRuntimeKey, MockRuntimeMeta> = {
    maxPriorTurns: {
        label: "🧠 Turnos recentes",
        shortDescription: "Perguntas e respostas recentes no contexto.",
        longDescription: "Número de turnos recentes reaproveitados no prompt do turno seguinte.",
        presets: [3, 5, 8, 10],
    },
    maxChannelMessages: {
        label: "💬 Mensagens do canal",
        shortDescription: "Contexto recente do canal atual.",
        longDescription: "Quantidade de mensagens recentes do canal atual usadas como contexto local.",
        presets: [10, 15, 25, 40],
    },
    maxToolRunsContext: {
        label: "🧩 Histórico de ferramentas",
        shortDescription: "Ferramentas anteriores para recuperar contexto.",
        longDescription: "Número de runs persistidos usados para reconstruir evidência em follow-ups.",
        presets: [6, 12, 18, 24],
    },
    maxEvidenceSlice: {
        label: "📚 Evidência reaproveitada",
        shortDescription: "Quantidade de evidência levada para o próximo turno.",
        longDescription: "Máximo de itens de evidência reaproveitados de runs anteriores.",
        presets: [16, 32, 48, 64],
    },
    retrievalHistoryLimit: {
        label: "🔎 Histórico por pesquisa",
        shortDescription: "Itens por chamada de retrieve_messages.",
        longDescription: "Tamanho de página padrão quando o modelo não define limite.",
        presets: [25, 50, 75, 100, 150, 250, 400, 600, 1000],
    },
    retrievalContextWindow: {
        label: "🪟 Janela à volta da mensagem",
        shortDescription: "Mensagens vizinhas ao redor do resultado.",
        longDescription: "Quantidade de mensagens vizinhas carregadas ao usar aroundMessageId.",
        presets: [8, 15, 25, 40],
    },
    maxToolCalls: {
        label: "🛠️ Máx. de ferramentas",
        shortDescription: "Limite de ferramentas por turno.",
        longDescription: "Limite rígido de execuções de ferramentas por turno.",
        presets: [2, 4, 6, 8, 10, 12, 15, 20, 25, 30],
    },
    maxRepeatedCallSignature: {
        label: "🔁 Limite de repetição",
        shortDescription: "Quantas vezes o mesmo pedido pode repetir.",
        longDescription: "Tentativas permitidas para a mesma assinatura antes de bloquear loop.",
        presets: [1, 2, 3],
    },
    maxLatencyBudgetMs: {
        label: "⏱️ Tempo máximo por turno",
        shortDescription: "Tempo total de execução por turno.",
        longDescription: "Tempo máximo de execução do loop antes de encerrar por orçamento.",
        presets: [10000, 15000, 20000, 30000, 45000, 60000, 90000, 120000, 180000, 240000, 300000],
    },
    escalationFetchLimit: {
        label: "🚀 Limite de refresh ao vivo",
        shortDescription: "Máximo de fetch durante refresh ao vivo.",
        longDescription: "Máximo de mensagens num refresh ao vivo durante retries de retrieve.",
        presets: [50, 100, 150, 250, 400, 600, 800, 1000],
    },
    approvalTimeoutMs: {
        label: "✅ Tempo de aprovação",
        shortDescription: "Tempo máximo à espera de admin.",
        longDescription: "Tempo máximo de espera por aprovação antes de auto-recusa.",
        presets: [30000, 60000, 120000, 300000],
    },
};

function cloneSettings(settings: BotSettings): BotSettings {
    return {
        ...settings,
        runtime: { ...settings.runtime },
        protectedChannelIds: [...settings.protectedChannelIds],
    };
}

function formatNumber(value: number): string {
    return value.toLocaleString("pt-PT");
}

function formatUsd(value?: number): string {
    if (value == null) return "n/d";
    const hasThreeDecimals = Math.abs(value * 100 - Math.round(value * 100)) > Number.EPSILON;
    return `$${value.toFixed(hasThreeDecimals ? 3 : 2)}`;
}

function getSortedModelEntries() {
    const profiles = readModelProfiles().profiles;
    return Object.entries(profiles).sort(([, left], [, right]) => {
        const leftInput = left.pricing?.inputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        const rightInput = right.pricing?.inputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        if (leftInput !== rightInput) {
            return leftInput - rightInput;
        }

        const leftOutput = left.pricing?.outputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        const rightOutput = right.pricing?.outputPerMillionUsd ?? Number.POSITIVE_INFINITY;
        if (leftOutput !== rightOutput) {
            return leftOutput - rightOutput;
        }

        return (left.label || "").localeCompare(right.label || "");
    });
}

function buildMockModelOptions(settings: BotSettings) {
    const selectedProfileName = resolveModelProfileName(settings.modelProfile, readModelProfiles());
    return getSortedModelEntries().map(([profileName, profile]) => {
        const inputPrice = profile.pricing?.inputPerMillionUsd;
        const outputPrice = profile.pricing?.outputPerMillionUsd;
        return {
            label: (profile.label || profileName).slice(0, 100),
            value: profileName,
            description: [
                `${formatNumber(profile.contextWindow)} contexto`,
                inputPrice != null && outputPrice != null
                    ? `in ${formatUsd(inputPrice)}/1M · out ${formatUsd(outputPrice)}/1M`
                    : "sem preço disponível.",
            ].join(" · ").slice(0, 100),
            default: profileName === selectedProfileName,
        };
    });
}

function buildMockPriceLines(profileName: string): string[] {
    const profile = readModelProfiles().profiles[profileName];
    const pricing = profile?.pricing;
    if (!pricing) {
        return ["Sem preço disponível."];
    }

    const lines = [
        `Entrada: \`${formatUsd(pricing.inputPerMillionUsd)} / 1M tokens\``,
        `Saída: \`${formatUsd(pricing.outputPerMillionUsd)} / 1M tokens\``,
    ];
    if (pricing.cacheReadPerMillionUsd != null) {
        lines.push(`Leitura de cache: \`${formatUsd(pricing.cacheReadPerMillionUsd)} / 1M tokens\``);
    }
    if (pricing.webSearchPerCallUsd != null) {
        lines.push(`Pesquisa web: \`${formatUsd(pricing.webSearchPerCallUsd)} / pedido\``);
    }
    return lines;
}

function getPlainRuntimeLabel(key: MockRuntimeKey): string {
    return MOCK_RUNTIME_META[key].label.replace(/^[^\p{L}\p{N}]+/u, "").trim();
}

function buildMockRuntimeLines(state: UiTestSettingsState): string[] {
    return (Object.keys(MOCK_RUNTIME_META) as MockRuntimeKey[]).map(
        (key) => `${getPlainRuntimeLabel(key)}: \`${formatNumber(state.settings.runtime[key])}\``
    );
}

function renderMockSettingsPanel(state: UiTestSettingsState) {
    const profiles = readModelProfiles();
    const selectedProfileName = resolveModelProfileName(state.settings.modelProfile, profiles);
    const selectedProfile = profiles.profiles[selectedProfileName];
    const selectedLabel = selectedProfile?.label || selectedProfileName;

    const components: Array<
        ContainerBuilder | ActionRowBuilder<ButtonBuilder> | ActionRowBuilder<StringSelectMenuBuilder>
    > = [];

    function spaceHack(size: number) {
        return "‎ ".repeat(size)
    }
    const sizeee = 12;

    const tabs = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(`${UITEST_SETTINGS_PREFIX}:tab:model`)
            .setLabel(spaceHack(sizeee) + "Modelo" + spaceHack(sizeee)) // espaçamento hack para deixar o botão maior
            .setStyle(state.tab === "model" ? ButtonStyle.Primary : ButtonStyle.Secondary)
            .setDisabled(state.tab === "model"),
        new ButtonBuilder()
            .setCustomId(`${UITEST_SETTINGS_PREFIX}:tab:runtime`)
            .setLabel(spaceHack(sizeee) + "Runtime" + spaceHack(sizeee)) // espaçamento hack para deixar o botão maior
            .setStyle(state.tab === "runtime" ? ButtonStyle.Primary : ButtonStyle.Secondary)
            .setDisabled(state.tab === "runtime"),
    );

    if (state.tab === "model") {
        components.push(
            new ContainerBuilder()
                .setAccentColor(UITEST_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("## Configurações"),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(
                        [
                            `Nome: \`${selectedLabel} ☑️\``,
                            selectedProfile ? `ID: \`${selectedProfile.chatModel}\`` : "ID OpenRouter: n/d",                            selectedProfile ? `Contexto: \`${formatNumber(selectedProfile.contextWindow)} tokens\`` : "- Contexto: n/d",
                            selectedProfile ? `Output máximo: \`${formatNumber(selectedProfile.maxOutputTokens)} tokens\`` : "- Output máximo: n/d",
                            ...buildMockPriceLines(selectedProfileName),
                        ].join("\n")
                    )
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${UITEST_SETTINGS_PREFIX}:model:select`)
                            .setPlaceholder("Escolher modelo")
                            .addOptions(buildMockModelOptions(state.settings))
                    )
                )
        );
        components.push(tabs);
    } else {
        const runtimeLines = buildMockRuntimeLines(state);

        components.push(
            new ContainerBuilder()
                .setAccentColor(UITEST_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("## Configurações"),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(
                        runtimeLines.join("\n")
                    )
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${UITEST_SETTINGS_PREFIX}:runtime`)
                            .setPlaceholder("Ajustar parâmetro runtime")
                            .addOptions(
                                (Object.keys(MOCK_RUNTIME_META) as MockRuntimeKey[]).map((key) => ({
                                    label: MOCK_RUNTIME_META[key].label.slice(0, 100),
                                    value: key,
                                    description: `${MOCK_RUNTIME_META[key].shortDescription} Atual: ${formatNumber(state.settings.runtime[key])}`.slice(0, 100),
                                }))
                            )
                    )
                )
                .addActionRowComponents(
                    new ActionRowBuilder<ButtonBuilder>().addComponents(
                        new ButtonBuilder()
                            .setCustomId(`${UITEST_SETTINGS_PREFIX}:auto-approve-writes:toggle`)
                            .setLabel(state.settings.runtime.autoApproveWrites ? "Desativar autoaprovação" : "Ativar autoaprovação")
                            .setStyle(state.settings.runtime.autoApproveWrites ? ButtonStyle.Secondary : ButtonStyle.Success),
                        new ButtonBuilder()
                            .setCustomId(`${UITEST_SETTINGS_PREFIX}:reset`)
                            .setLabel("Repor padrões")
                            .setStyle(ButtonStyle.Danger)
                    )
                )
        );
        components.push(tabs);
    }

    return {
        components,
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

function renderMockRuntimePicker(state: UiTestSettingsState, key: MockRuntimeKey) {
    const meta = MOCK_RUNTIME_META[key];
    const currentValue = state.settings.runtime[key];
    const values = meta.presets.includes(currentValue)
        ? meta.presets
        : [...meta.presets, currentValue].sort((a, b) => a - b);

    return {
        components: [
            new ContainerBuilder()
                .setAccentColor(UITEST_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(`## ${meta.label}`),
                    new TextDisplayBuilder().setContent(
                        [
                            `**Descrição:** ${meta.longDescription}`,
                            `**Valor atual:** ${formatNumber(currentValue)}`,
                        ].join("\n")
                    )
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${UITEST_SETTINGS_PREFIX}:runtime:set:${key}`)
                            .setPlaceholder("Escolher valor")
                            .addOptions(
                                values.map((value) => ({
                                    label: formatNumber(value),
                                    value: String(value),
                                    default: value === currentValue,
                                }))
                            )
                    )
                ),
            new ActionRowBuilder<ButtonBuilder>().addComponents(
                new ButtonBuilder()
                    .setCustomId(`${UITEST_SETTINGS_PREFIX}:tab:runtime`)
                    .setLabel("Voltar ao runtime")
                    .setStyle(ButtonStyle.Secondary)
            ),
        ],
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

function parseMockSettingsInteraction(customId: string) {
    if (customId === `${UITEST_SETTINGS_PREFIX}:tab:model`) return { type: "tab_model" as const };
    if (customId === `${UITEST_SETTINGS_PREFIX}:tab:runtime`) return { type: "tab_runtime" as const };
    if (customId === `${UITEST_SETTINGS_PREFIX}:model:select`) return { type: "model_select" as const };
    if (customId === `${UITEST_SETTINGS_PREFIX}:runtime`) return { type: "runtime_pick" as const };
    if (customId === `${UITEST_SETTINGS_PREFIX}:auto-approve-writes:toggle`) return { type: "auto_approve_writes_toggle" as const };
    if (customId === `${UITEST_SETTINGS_PREFIX}:reset`) return { type: "reset" as const };
    if (customId.startsWith(`${UITEST_SETTINGS_PREFIX}:runtime:set:`)) {
        return {
            type: "runtime_set" as const,
            key: customId.slice(`${UITEST_SETTINGS_PREFIX}:runtime:set:`.length) as MockRuntimeKey,
        };
    }
    return { type: null };
}

const uiTestSettings: UiTestSubcommandModule = {
    name: "settings",
    description: "Abrir mock navegável do painel /settings",
    register(builder: SlashCommandSubcommandBuilder) {
        return builder.setName(this.name).setDescription(this.description);
    },
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        if (!(await guardUiTestAdmin(interaction))) {
            return;
        }

        const settings = cloneSettings(SettingsService.load());
        const state: UiTestSettingsState = {
            ownerUserId: interaction.user.id,
            settings,
            tab: "model",
        };

        await interaction.reply(renderMockSettingsPanel(state));
        const reply = await interaction.fetchReply();
        uiTestSettingsStates.set(reply.id, state);
    },
};

export async function handleUiTestSettingsInteraction(
    interaction: ButtonInteraction | StringSelectMenuInteraction
): Promise<boolean> {
    if (!interaction.customId.startsWith(`${UITEST_SETTINGS_PREFIX}:`)) {
        return false;
    }

    const state = uiTestSettingsStates.get(interaction.message.id);
    if (!state) {
        await interaction.reply({
            content: "❌ Este mock expirou.",
            flags: MessageFlags.Ephemeral as const,
        });
        return true;
    }

    if (interaction.user.id !== state.ownerUserId) {
        await interaction.reply({
            content: "❌ Só quem abriu este mock pode interagir com ele.",
            flags: MessageFlags.Ephemeral as const,
        });
        return true;
    }

    const parsed = parseMockSettingsInteraction(interaction.customId);

    switch (parsed.type) {
        case "tab_model":
            state.tab = "model";
            await interaction.update(renderMockSettingsPanel(state));
            return true;
        case "tab_runtime":
            state.tab = "runtime";
            await interaction.update(renderMockSettingsPanel(state));
            return true;
        case "model_select":
            if (!interaction.isStringSelectMenu()) return true;
            state.settings = {
                ...state.settings,
                modelProfile: interaction.values[0],
            };
            state.tab = "model";
            await interaction.update(renderMockSettingsPanel(state));
            return true;
        case "runtime_pick":
            if (!interaction.isStringSelectMenu()) return true;
            state.tab = "runtime";
            await interaction.update(renderMockRuntimePicker(state, interaction.values[0] as MockRuntimeKey));
            return true;
        case "runtime_set":
            if (!interaction.isStringSelectMenu() || !parsed.key) return true;
            state.settings = {
                ...state.settings,
                runtime: {
                    ...state.settings.runtime,
                    [parsed.key]: Number(interaction.values[0]),
                },
            };
            state.tab = "runtime";
            await interaction.update(renderMockSettingsPanel(state));
            return true;
        case "auto_approve_writes_toggle":
            state.settings = {
                ...state.settings,
                runtime: {
                    ...state.settings.runtime,
                    autoApproveWrites: !state.settings.runtime.autoApproveWrites,
                },
            };
            state.tab = "runtime";
            await interaction.update(renderMockSettingsPanel(state));
            return true;
        case "reset":
            state.settings = cloneSettings(SettingsService.getDefaults());
            state.tab = "model";
            await interaction.update(renderMockSettingsPanel(state));
            return true;
        default:
            return false;
    }
}

export default uiTestSettings;
