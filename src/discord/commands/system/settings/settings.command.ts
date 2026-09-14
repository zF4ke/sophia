import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ChatInputCommandInteraction,
    ContainerBuilder,
    MessageFlags,
    SlashCommandBuilder,
    StringSelectMenuBuilder,
    TextDisplayBuilder,
} from "discord.js";
import { SettingsService, type BotSettings } from "@/app/SettingsService";
import { listModelProfiles, readModelProfiles, resolveModelProfileName } from "@/app/modelProfiles";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient } from "@/shared/appTypes";

export type RuntimeSettingKey =
    | "toolCallLimit"
    | "modelConcurrency"
    | "maxRepeatedCallSignature"
    | "maxPriorTurns"
    | "maxChannelMessages"
    | "maxEvidenceSlice"
    | "maxToolRunsContext"
    | "escalationFetchLimit"
    | "retrievalHistoryLimit"
    | "retrievalContextWindow"
    | "approvalTimeoutMs"
    | "longTaskEvidenceSliceFloor";

const LONG_TASK_KEYS: RuntimeSettingKey[] = [
    "longTaskEvidenceSliceFloor",
];

function isLongTaskKey(key: RuntimeSettingKey): boolean {
    return LONG_TASK_KEYS.includes(key);
}

export function getRuntimeSettingValue(
    settings: BotSettings,
    key: RuntimeSettingKey,
): number {
    switch (key) {
        case "longTaskEvidenceSliceFloor":
            return settings.runtime.longTask.evidenceSliceFloor;
        default:
            return settings.runtime[key] as number;
    }
}

export function buildRuntimeSettingPatch(
    current: BotSettings,
    key: RuntimeSettingKey,
    value: number,
): Partial<BotSettings> {
    if (isLongTaskKey(key)) {
        const longTask = { ...current.runtime.longTask };
        switch (key) {
            case "longTaskEvidenceSliceFloor":
                longTask.evidenceSliceFloor = value;
                break;
        }
        return { runtime: { ...current.runtime, longTask } };
    }
    return { runtime: { ...current.runtime, [key]: value } };
}

export type SettingsTab = "model" | "runtime" | "compaction" | "longTask" | "voice" | "features";

type VoiceMode = BotSettings["voice"];

const VOICE_OPTIONS: Array<{
    value: VoiceMode;
    label: string;
    description: string;
}> = [
    {
        value: "balanced",
        label: "Equilibrada",
        description: "Calorosa e direta. A voz base.",
    },
    {
        value: "casual",
        label: "Descontraída",
        description: "Confiante, curiosa e com humor seco. Afiada sem ser fria.",
    },
    {
        value: "formal",
        label: "Formal",
        description: "Mais formal, com explicações diretas e completas.",
    },
];
type SettingsPanelOptions = {
    idPrefix?: string;
};

type RuntimeSettingMeta = {
    label: string;
    shortDescription: string;
    longDescription: string;
    presets: number[];
};

const RUNTIME_SETTING_META: Record<RuntimeSettingKey, RuntimeSettingMeta> = {
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
    toolCallLimit: {
        label: "Limite opcional de ferramentas",
        shortDescription: "0 permite continuar sem limite de chamadas.",
        longDescription: "Limite explícito por execução, incluindo continuações. 0 desativa. Ao atingir o limite, a tarefa fica incompleta.",
        presets: [0, 100, 500, 1000],
    },
    modelConcurrency: {
        label: "Pedidos simultâneos ao modelo",
        shortDescription: "Capacidade partilhada entre utilizadores.",
        longDescription: "Até este número de pedidos em simultâneo, com um por utilizador. Conversas têm prioridade sobre sonhos em espera. Não limita o trabalho total de uma tarefa.",
        presets: [1, 2, 4, 8],
    },
    maxRepeatedCallSignature: {
        label: "🔁 Limite de repetição",
        shortDescription: "Quantas vezes o mesmo pedido pode repetir.",
        longDescription: "Tentativas permitidas para a mesma assinatura antes de bloquear loop.",
        presets: [1, 2, 3],
    },
    escalationFetchLimit: {
        label: "🚀 Limite de refresh ao vivo",
        shortDescription: "Máximo de fetch durante refresh ao vivo.",
        longDescription: "Máximo de mensagens num refresh ao vivo durante retries de retrieve.",
        presets: [50, 100, 150, 250, 400, 600, 800, 1000],
    },
    approvalTimeoutMs: {
        label: "✅ Tempo de aprovação",
        shortDescription: "Tempo máximo à espera de aprovação.",
        longDescription: "Tempo máximo de espera por aprovação de ações write/destructive antes de auto-recusa.",
        presets: [30000, 60000, 120000, 300000],
    },
    longTaskEvidenceSliceFloor: {
        label: "📚 Long task: floor de evidência",
        shortDescription: "Piso da fatia de evidência em long-task.",
        longDescription: "Valor mínimo para maxEvidenceSlice ao iniciar long-task, garantindo que retrievals grandes não sejam cortados.",
        presets: [64, 96, 128, 192, 256],
    },
};

const SETTINGS_CONTAINER_ACCENT = 0xd1d5db;
const TAB_LABEL_PAD = 0;

function formatNumber(value: number): string {
    return value.toLocaleString("pt-PT");
}

function formatUsd(value?: number): string {
    if (value == null) return "n/d";
    const hasThreeDecimals =
        Math.abs(value * 100 - Math.round(value * 100)) > Number.EPSILON;
    return `$${value.toFixed(hasThreeDecimals ? 3 : 2)}`;
}

function padTabLabel(label: string): string {
    const spacer = "‎ ".repeat(TAB_LABEL_PAD);
    return `${spacer}${label}${spacer}`;
}

function getPlainRuntimeLabel(key: RuntimeSettingKey): string {
    return RUNTIME_SETTING_META[key].label.replace(/^[^\p{L}\p{N}]+/u, "").trim();
}

function buildPriceLines(profileName: string): string[] {
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

function buildModelOptions(settings: BotSettings) {
    const profileConfig = readModelProfiles();
    const selectedProfileName = resolveModelProfileName(settings.modelProfile, profileConfig);
    return listModelProfiles(profileConfig)
        .map(([profileName, profile]) => {
            const label = profile.label || profileName;
            const inputPrice = profile.pricing?.inputPerMillionUsd;
            const outputPrice = profile.pricing?.outputPerMillionUsd;
            const description = [
                `${formatNumber(profile.contextWindow)} contexto`,
                inputPrice != null && outputPrice != null
                    ? `in ${formatUsd(inputPrice)}/1M · out ${formatUsd(outputPrice)}/1M`
                    : "sem preço disponível.",
            ].join(" · ");

            return {
                label: label.slice(0, 100),
                value: profileName,
                description: description.slice(0, 100),
                default: profileName === selectedProfileName,
            };
        });
}

function buildTabsRow(tab: SettingsTab, idPrefix: string) {
    return new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(`${idPrefix}:tab:model`)
            .setLabel(padTabLabel("Modelo"))
            .setStyle(tab === "model" ? ButtonStyle.Primary : ButtonStyle.Secondary)
            .setDisabled(tab === "model"),
        new ButtonBuilder()
            .setCustomId(`${idPrefix}:tab:runtime`)
            .setLabel(padTabLabel("Runtime"))
            .setStyle(tab === "runtime" ? ButtonStyle.Primary : ButtonStyle.Secondary)
            .setDisabled(tab === "runtime"),
        new ButtonBuilder()
            .setCustomId(`${idPrefix}:tab:compaction`)
            .setLabel(padTabLabel("Compactação"))
            .setStyle(tab === "compaction" ? ButtonStyle.Primary : ButtonStyle.Secondary)
            .setDisabled(tab === "compaction"),
        new ButtonBuilder()
            .setCustomId(`${idPrefix}:tab:longTask`)
            .setLabel(padTabLabel("Long task"))
            .setStyle(tab === "longTask" ? ButtonStyle.Primary : ButtonStyle.Secondary)
            .setDisabled(tab === "longTask"),
        new ButtonBuilder()
            .setCustomId(`${idPrefix}:tab:voice`)
            .setLabel(padTabLabel("Voz"))
            .setStyle(tab === "voice" ? ButtonStyle.Primary : ButtonStyle.Secondary)
            .setDisabled(tab === "voice")
    );
}

export function buildSettingsPanel(
    settings: BotSettings,
    tab: SettingsTab = "model",
    options: SettingsPanelOptions = {}
) {
    const idPrefix = options.idPrefix ?? "settings";
    const profileConfig = readModelProfiles();
    const selectedProfileName = resolveModelProfileName(settings.modelProfile, profileConfig);
    const selectedProfile = profileConfig.profiles[selectedProfileName];
    const selectedLabel = selectedProfile?.label || selectedProfileName;

    const components: Array<
        ContainerBuilder | ActionRowBuilder<ButtonBuilder> | ActionRowBuilder<StringSelectMenuBuilder>
    > = [];

    if (tab === "model") {
        const modelInfoLines = [
            `Nome: \`${selectedLabel} ☑️\``,
            selectedProfile ? `ID: \`${selectedProfile.chatModel}\`` : "ID: n/d",
            selectedProfile
                ? `Contexto: \`${formatNumber(selectedProfile.contextWindow)} tokens\``
                : "Contexto: n/d",
            selectedProfile
                ? `Output máximo: \`${formatNumber(selectedProfile.maxOutputTokens)} tokens\``
                : "Output máximo: n/d",
            ...buildPriceLines(selectedProfileName),
        ];

        components.push(
            new ContainerBuilder()
                .setAccentColor(SETTINGS_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("## Configurações — Modelo"),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(modelInfoLines.join("\n"))
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${idPrefix}:model:select`)
                            .setPlaceholder("Escolher modelo")
                            .addOptions(buildModelOptions(settings))
                    )
                )
        );
    } else if (tab === "runtime") {
        const runtimeKeys = (Object.keys(RUNTIME_SETTING_META) as RuntimeSettingKey[]).filter(
            (key) => !isLongTaskKey(key)
        );
        const runtimeLines = runtimeKeys.map(
            (key) => `${getPlainRuntimeLabel(key)}: \`${formatNumber(getRuntimeSettingValue(settings, key))}\``
        );

        components.push(
            new ContainerBuilder()
                .setAccentColor(SETTINGS_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("## Configurações — Runtime"),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(runtimeLines.join("\n"))
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${idPrefix}:runtime`)
                            .setPlaceholder("Ajustar parâmetro runtime")
                            .addOptions(
                                runtimeKeys.map((key) => {
                                    const meta = RUNTIME_SETTING_META[key];
                                    return {
                                        label: meta.label.slice(0, 100),
                                        value: key,
                                        description: `${meta.shortDescription} Atual: ${formatNumber(getRuntimeSettingValue(settings, key))}`.slice(0, 100),
                                    };
                                })
                            )
                    )
                )
                .addActionRowComponents(
                    new ActionRowBuilder<ButtonBuilder>().addComponents(
                        new ButtonBuilder()
                            .setCustomId(`${idPrefix}:reset`)
                            .setLabel("Repor ajustes de execução")
                            .setStyle(ButtonStyle.Danger)
                    )
                )
        );
    } else if (tab === "compaction") {
        const compactionSettings = settings.compaction;
        const compactionProfileName = compactionSettings.summarizerModel;
        const compactionProfile = profileConfig.profiles[compactionProfileName];
        const compactionLabel = compactionProfile?.label || compactionProfileName;

        const compactionInfoLines = [
            `Modelo de compactação: \`${compactionLabel} ☑️\``,
            `Trigger: \`${Math.round(compactionSettings.triggerFraction * 100)}%\` do contexto`,
        ];

        const compactionModelOptions = listModelProfiles(profileConfig)
            .map(([name, p]) => ({
                label: (p.label || name).slice(0, 100),
                value: name,
                description: p.pricing
                    ? `in ${formatUsd(p.pricing.inputPerMillionUsd)}/1M · out ${formatUsd(p.pricing.outputPerMillionUsd)}/1M`.slice(0, 100)
                    : "sem preço",
                default: name === compactionProfileName,
            }));

        components.push(
            new ContainerBuilder()
                .setAccentColor(SETTINGS_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("## Configurações — Compactação"),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(compactionInfoLines.join("\n"))
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${idPrefix}:compaction:model`)
                            .setPlaceholder("Escolher modelo de compactação")
                            .addOptions(
                                compactionModelOptions.length > 0
                                    ? compactionModelOptions
                                    : [{ label: "Nenhum modelo elegível", value: "_none" }]
                            )
                    )
                )
        );
    }

    if (tab === "voice") {
        const current = settings.voice;
        const descriptionLines = VOICE_OPTIONS.map((opt) => {
            const mark = opt.value === current ? "☑️" : "▫️";
            return `${mark} **${opt.label}** — ${opt.description}`;
        });

        components.push(
            new ContainerBuilder()
                .setAccentColor(SETTINGS_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("## Configurações — Voz"),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(
                        "Controla o tom e a voz da Sophia. A escolha é aplicada em todos os turnos.",
                    ),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(descriptionLines.join("\n")),
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${idPrefix}:voice:select`)
                            .setPlaceholder("Escolher estilo")
                            .addOptions(
                                VOICE_OPTIONS.map((opt) => ({
                                    label: opt.label.slice(0, 100),
                                    value: opt.value,
                                    description: opt.description.slice(0, 100),
                                    default: opt.value === current,
                                })),
                            ),
                    ),
                ),
        );
    }

    if (tab === "longTask") {
        const longTaskLines = LONG_TASK_KEYS.map(
            (key) => `${getPlainRuntimeLabel(key)}: \`${formatNumber(getRuntimeSettingValue(settings, key))}\``
        );

        components.push(
            new ContainerBuilder()
                .setAccentColor(SETTINGS_CONTAINER_ACCENT)
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent("## Configurações — Long task"),
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(
                        "Contexto e recuperação de mensagens para investigações maiores. Estes ajustes não limitam a duração do pedido nem exigem estimativas do modelo."
                    )
                )
                .addTextDisplayComponents(
                    new TextDisplayBuilder().setContent(longTaskLines.join("\n"))
                )
                .addActionRowComponents(
                    new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
                        new StringSelectMenuBuilder()
                            .setCustomId(`${idPrefix}:runtime`)
                            .setPlaceholder("Ajustar parâmetro long-task")
                            .addOptions(
                                LONG_TASK_KEYS.map((key) => {
                                    const meta = RUNTIME_SETTING_META[key];
                                    return {
                                        label: meta.label.slice(0, 100),
                                        value: key,
                                        description: `${meta.shortDescription} Atual: ${formatNumber(getRuntimeSettingValue(settings, key))}`.slice(0, 100),
                                    };
                                })
                            )
                    )
                )
        );
    }

    if (tab === "features") {
        const entries = [
            { key: "dreaming", label: "Memória automática", enabled: settings.memory.dreamingEnabled, description: "Consolida conversas entregues e propõe procedimentos privados durante períodos livres." },
            { key: "sandbox", label: "Ficheiros e execução", enabled: settings.sandbox.enabled, description: "Permite trabalhar em ficheiros num contentor isolado. Requer Docker e a imagem local." },
            { key: "scheduling", label: "Agendamentos", enabled: settings.scheduling.enabled, description: "Executa os acompanhamentos guardados. Pausar este serviço preserva os agendamentos." },
        ];
        const container = new ContainerBuilder().setAccentColor(SETTINGS_CONTAINER_ACCENT)
            .addTextDisplayComponents(new TextDisplayBuilder().setContent("## Memória, ficheiros e agendamentos"));
        for (const entry of entries) container.addTextDisplayComponents(new TextDisplayBuilder().setContent(`**${entry.label}: ${entry.enabled ? "ativo" : "pausado"}**\n${entry.description}`))
            .addActionRowComponents(new ActionRowBuilder<ButtonBuilder>().addComponents(new ButtonBuilder().setCustomId(`${idPrefix}:${entry.key}:toggle`).setLabel(`${entry.enabled ? "Pausar" : "Ativar"} ${entry.label.toLowerCase()}`).setStyle(ButtonStyle.Secondary)));
        components.push(container);
    }
    components.push(buildTabsRow(tab, idPrefix));
    components.push(new ActionRowBuilder<ButtonBuilder>().addComponents(new ButtonBuilder().setCustomId(`${idPrefix}:tab:features`).setLabel("Memória e ferramentas").setStyle(tab === "features" ? ButtonStyle.Primary : ButtonStyle.Secondary).setDisabled(tab === "features"), new ButtonBuilder().setCustomId(`${idPrefix}:tab:costs`).setLabel("Custos").setStyle(ButtonStyle.Secondary)));

    return {
        components,
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

export function buildRuntimeValueSelect(
    key: RuntimeSettingKey,
    currentValue: number,
    options: SettingsPanelOptions = {}
) {
    const idPrefix = options.idPrefix ?? "settings";
    const meta = RUNTIME_SETTING_META[key];
    const presets = meta.presets.includes(currentValue)
        ? meta.presets
        : [...meta.presets, currentValue].sort((a, b) => a - b);

    return {
        components: [
            new ContainerBuilder()
                .setAccentColor(SETTINGS_CONTAINER_ACCENT)
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
                            .setCustomId(`${idPrefix}:runtime:set:${key}`)
                            .setPlaceholder("Escolher valor")
                            .addOptions(
                                presets.map((v) => ({
                                    label: formatNumber(v),
                                    value: String(v),
                                    default: v === currentValue,
                                }))
                            )
                    )
                ),
            new ActionRowBuilder<ButtonBuilder>().addComponents(
                new ButtonBuilder()
                    .setCustomId(`${idPrefix}:tab:${isLongTaskKey(key) ? "longTask" : "runtime"}`)
                    .setLabel(isLongTaskKey(key) ? "Voltar ao Long task" : "Voltar ao runtime")
                    .setStyle(ButtonStyle.Secondary)
            ),
        ],
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

export default {
    data: new SlashCommandBuilder()
        .setName("settings")
        .setDescription("Configurar modelo e runtime da Sophia")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDMPermission(false)
        .addBooleanOption(option => option.setName("ephemeral").setDescription("Mostrar este painel só para ti (por padrão aparece no canal)")),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ Apenas administradores podem usar este comando.",
                flags: MessageFlags.Ephemeral as const,
            });
            return;
        }

        const settings = SettingsService.load();
        await interaction.reply({ ...buildSettingsPanel(settings, "model"), flags: MessageFlags.IsComponentsV2 | (interaction.options.getBoolean("ephemeral") ? MessageFlags.Ephemeral : 0) });
    },
};

export function handleSettingsInteraction(customId: string): {
    type:
        | "tab_model"
        | "tab_runtime"
        | "tab_compaction"
        | "tab_long_task"
        | "tab_voice"
        | "model_select"
        | "compaction_model_select"
        | "runtime_pick"
        | "runtime_set"
        | "voice_select"
        | "dreaming_toggle"
        | "tab_features"
        | "sandbox_toggle"
        | "scheduling_toggle"
        | "reset"
        | null;
    key?: RuntimeSettingKey;
} {
    if (customId === "settings:tab:features") return { type: "tab_features" };
    if (customId === "settings:sandbox:toggle") return { type: "sandbox_toggle" };
    if (customId === "settings:scheduling:toggle") return { type: "scheduling_toggle" };
    if (customId === "settings:tab:model") return { type: "tab_model" };
    if (customId === "settings:tab:runtime") return { type: "tab_runtime" };
    if (customId === "settings:tab:compaction") return { type: "tab_compaction" };
    if (customId === "settings:tab:longTask") return { type: "tab_long_task" };
    if (customId === "settings:tab:voice") return { type: "tab_voice" };
    if (customId === "settings:voice:select") return { type: "voice_select" };
    if (customId === "settings:model:select") return { type: "model_select" };
    if (customId === "settings:compaction:model") return { type: "compaction_model_select" };
    if (customId === "settings:runtime") return { type: "runtime_pick" };
    if (customId === "settings:dreaming:toggle") return { type: "dreaming_toggle" };
    if (customId === "settings:reset") return { type: "reset" };
    if (customId.startsWith("settings:runtime:set:")) {
        return {
            type: "runtime_set",
            key: customId.slice("settings:runtime:set:".length) as RuntimeSettingKey,
        };
    }
    return { type: null };
}

export function parseSettingsInteraction(
    customId: string,
    idPrefix = "settings"
): {
    type:
        | "tab_model"
        | "tab_runtime"
        | "tab_compaction"
        | "tab_long_task"
        | "tab_voice"
        | "model_select"
        | "compaction_model_select"
        | "runtime_pick"
        | "runtime_set"
        | "voice_select"
        | "dreaming_toggle"
        | "tab_features"
        | "sandbox_toggle"
        | "scheduling_toggle"
        | "reset"
        | null;
    key?: RuntimeSettingKey;
} {
    if (customId === `${idPrefix}:tab:features`) return { type: "tab_features" };
    if (customId === `${idPrefix}:sandbox:toggle`) return { type: "sandbox_toggle" };
    if (customId === `${idPrefix}:scheduling:toggle`) return { type: "scheduling_toggle" };
    if (customId === `${idPrefix}:tab:model`) return { type: "tab_model" };
    if (customId === `${idPrefix}:tab:runtime`) return { type: "tab_runtime" };
    if (customId === `${idPrefix}:tab:compaction`) return { type: "tab_compaction" };
    if (customId === `${idPrefix}:tab:longTask`) return { type: "tab_long_task" };
    if (customId === `${idPrefix}:tab:voice`) return { type: "tab_voice" };
    if (customId === `${idPrefix}:voice:select`) return { type: "voice_select" };
    if (customId === `${idPrefix}:model:select`) return { type: "model_select" };
    if (customId === `${idPrefix}:compaction:model`) return { type: "compaction_model_select" };
    if (customId === `${idPrefix}:runtime`) return { type: "runtime_pick" };
    if (customId === `${idPrefix}:dreaming:toggle`) return { type: "dreaming_toggle" };
    if (customId === `${idPrefix}:reset`) return { type: "reset" };
    if (customId.startsWith(`${idPrefix}:runtime:set:`)) {
        return {
            type: "runtime_set",
            key: customId.slice(`${idPrefix}:runtime:set:`.length) as RuntimeSettingKey,
        };
    }
    return { type: null };
}
