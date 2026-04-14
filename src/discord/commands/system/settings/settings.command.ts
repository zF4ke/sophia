import fs from "fs";
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
import { AppPaths } from "@/app/AppPaths";
import { SettingsService, type BotSettings } from "@/app/SettingsService";
import { SecurityService } from "@/security/SecurityService";
import type { BotClient, ModelProfileConfig } from "@/shared/appTypes";

export type RuntimeSettingKey =
    | "maxToolCalls"
    | "maxRepeatedCallSignature"
    | "maxLatencyBudgetMs"
    | "maxPriorTurns"
    | "maxChannelMessages"
    | "maxEvidenceSlice"
    | "maxToolRunsContext"
    | "interactiveCrawlLimit"
    | "escalationFetchLimit"
    | "retrievalHistoryLimit"
    | "retrievalContextWindow"
    | "approvalTimeoutMs";

type RuntimeSettingMeta = {
    label: string;
    shortDescription: string;
    longDescription: string;
    presets: number[];
};

const MAX_TOOL_RESULT_CHARS = 150_000;
const CONTEXT_HEADROOM_RATIO = 0.80;

const RUNTIME_SETTING_META: Record<RuntimeSettingKey, RuntimeSettingMeta> = {
    maxPriorTurns: {
        label: "Recent Turns",
        shortDescription: "How many recent Q/A pairs go back into the prompt.",
        longDescription: "Number of recent conversation turns replayed into the next turn's prompt as compact dialogue context.",
        presets: [3, 5, 8, 10],
    },
    maxChannelMessages: {
        label: "Recent Channel Messages",
        shortDescription: "Ambient channel context loaded per turn.",
        longDescription: "How many recent ambient messages from the current channel are injected as local context before the loop starts.",
        presets: [10, 15, 25, 40],
    },
    maxToolRunsContext: {
        label: "Prior Tool Runs",
        shortDescription: "How many past tool runs are replayed for evidence reconstruction.",
        longDescription: "Number of persisted tool runs scanned when rebuilding prior evidence for follow-up turns.",
        presets: [6, 12, 18, 24],
    },
    maxEvidenceSlice: {
        label: "Prior Evidence Slice",
        shortDescription: "Reusable evidence carried into the next turn.",
        longDescription: "Maximum number of evidence items reconstructed from prior tool runs and carried into the next turn's system prompt.",
        presets: [16, 32, 48, 64],
    },
    retrievalHistoryLimit: {
        label: "Default Retrieval Page",
        shortDescription: "Default rows returned per retrieve_messages call.",
        longDescription: "Default page size for retrieve_messages when the model does not specify a limit. Higher values scan more history per call but also create larger tool outputs.",
        presets: [25, 50, 75, 100, 150],
    },
    retrievalContextWindow: {
        label: "Around-Message Window",
        shortDescription: "Neighbor messages loaded around a hit.",
        longDescription: "How many neighboring messages are loaded when retrieve_messages zooms into a specific message via aroundMessageId.",
        presets: [8, 15, 25, 40],
    },
    maxToolCalls: {
        label: "Max Tool Calls",
        shortDescription: "Hard cap on tool executions per turn.",
        longDescription: "Hard cap on how many tool executions Sophia can make in one turn before it must finish or fall back.",
        presets: [
            2,
            4,
            6,
            8,
            10,
            12,
            15,
            20,
            25,
            30,
        ],
    },
    maxRepeatedCallSignature: {
        label: "Repeated Call Guard",
        shortDescription: "Same tool + same args retry limit.",
        longDescription: "How many times the exact same tool call signature is allowed before the runtime blocks it as a loop.",
        presets: [1, 2, 3],
    },
    maxLatencyBudgetMs: {
        label: "Latency Budget",
        shortDescription: "Total runtime budget per turn.",
        longDescription: "Maximum wall-clock time the loop can spend on one turn before stopping with a budget exit.",
        presets: [
            10000,
            15000,
            20000,
            30000,
            45000,
            60000,
            90000,
            120000,
            180000,
            240000,
            300000,
        ],
    },
    interactiveCrawlLimit: {
        label: "Interactive Crawl Limit",
        shortDescription: "Live Discord history fetch ceiling.",
        longDescription: "Maximum number of live Discord messages a direct channel crawl may ingest when local history is not enough.",
        presets: [100, 250, 400, 600],
    },
    escalationFetchLimit: {
        label: "Escalation Fetch Limit",
        shortDescription: "Live refresh cap for scoped retrieval retries.",
        longDescription: "Maximum number of messages fetched during a scoped live refresh when retrieve_messages escalates beyond the cache.",
        presets: [50, 150, 250, 400],
    },
    approvalTimeoutMs: {
        label: "Approval Timeout",
        shortDescription: "How long to wait for admin approval on write/destructive actions.",
        longDescription: "Maximum time in milliseconds the runtime will wait for an admin to approve or deny a write or destructive tool call before auto-denying.",
        presets: [30000, 60000, 120000, 300000],
    },
};

function readModelProfiles(): ModelProfileConfig {
    try {
        const raw = fs.readFileSync(AppPaths.modelProfilesPath, "utf8");
        return JSON.parse(raw) as ModelProfileConfig;
    } catch {
        return {
            defaultProfile: "fast",
            profiles: {
                fast: {
                    chatModel: "google/gemini-3.1-flash-lite-preview",
                    analysisModel: "google/gemini-3.1-flash-lite-preview",
                    embeddingModel: "openai/text-embedding-3-small",
                    temperature: 0.5,
                    maxOutputTokens: 1200,
                    contextWindow: 1_000_000,
                },
                smarter: {
                    chatModel: "minimax/minimax-m2.7",
                    analysisModel: "minimax/minimax-m2.7",
                    embeddingModel: "openai/text-embedding-3-small",
                    temperature: 0.5,
                    maxOutputTokens: 1400,
                    contextWindow: 100_000,
                },
                alt: {
                    chatModel: "deepseek/deepseek-v3.2",
                    analysisModel: "deepseek/deepseek-v3.2",
                    embeddingModel: "openai/text-embedding-3-small",
                    temperature: 0.5,
                    maxOutputTokens: 1400,
                    contextWindow: 128_000,
                },
            },
        };
    }
}

function formatNumber(value: number): string {
    return value.toLocaleString("en-US");
}

function buildSettingsPanel(settings: BotSettings) {
    const profileConfig = readModelProfiles();
    const profiles = Object.keys(profileConfig.profiles);
    const selectedProfile = profileConfig.profiles[settings.modelProfile];
    const pruneThreshold = selectedProfile
        ? Math.floor(selectedProfile.contextWindow * CONTEXT_HEADROOM_RATIO)
        : null;

    const runtimeLines = (Object.keys(RUNTIME_SETTING_META) as RuntimeSettingKey[]).map((key) => {
        const meta = RUNTIME_SETTING_META[key];
        return `- **${meta.label}:** ${formatNumber(settings.runtime[key])} — ${meta.shortDescription}`;
    });

    const infoLines = [
        `**Profile:** ${settings.modelProfile}${selectedProfile ? ` · ${selectedProfile.chatModel}` : ""}`,
        selectedProfile ? `**Context:** ${formatNumber(selectedProfile.contextWindow)} tokens · **Max Output:** ${formatNumber(selectedProfile.maxOutputTokens)}` : null,
        `**Debug:** ${settings.debug ? "enabled" : "disabled"}`,
        `**Auto-Approve Writes (non-destructive):** ${settings.runtime.autoApproveWrites ? "enabled" : "disabled"}`,
        "",
        ...runtimeLines,
    ].filter((x): x is string => x !== null);

    if (pruneThreshold != null) {
        infoLines.push(`- **Prompt Prune Threshold:** ~${formatNumber(pruneThreshold)} tokens (80% of context)`);
    }

    const mainContainer = new ContainerBuilder()
        .setAccentColor(0x5865f2)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## ⚙️ Sophia Settings"),
            new TextDisplayBuilder().setContent(infoLines.join("\n"))
        );

    const profileSelect = new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
        new StringSelectMenuBuilder()
            .setCustomId("settings:profile")
            .setPlaceholder("Change model profile")
            .addOptions(
                profiles.map((p) => ({
                    label: p,
                    value: p,
                    default: p === settings.modelProfile,
                }))
            )
    );

    const runtimeSelect = new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
        new StringSelectMenuBuilder()
            .setCustomId("settings:runtime")
            .setPlaceholder("Change a runtime parameter")
            .addOptions(
                (Object.keys(RUNTIME_SETTING_META) as RuntimeSettingKey[]).map((key) => {
                    const meta = RUNTIME_SETTING_META[key];
                    return {
                        label: meta.label,
                        value: key,
                        description: `${meta.shortDescription} Current: ${formatNumber(settings.runtime[key])}`.slice(0, 100),
                    };
                })
            )
    );

    const buttonRow = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId("settings:auto-approve-writes:toggle")
            .setLabel(settings.runtime.autoApproveWrites ? "Disable Auto-Approve Writes" : "Enable Auto-Approve Writes")
            .setStyle(settings.runtime.autoApproveWrites ? ButtonStyle.Secondary : ButtonStyle.Primary),
        new ButtonBuilder()
            .setCustomId("settings:debug:toggle")
            .setLabel(settings.debug ? "Disable Debug" : "Enable Debug")
            .setStyle(settings.debug ? ButtonStyle.Secondary : ButtonStyle.Success),
        new ButtonBuilder()
            .setCustomId("settings:reset")
            .setLabel("Reset to Defaults")
            .setStyle(ButtonStyle.Danger)
    );

    return {
        components: [
            mainContainer,
            profileSelect,
            runtimeSelect,
            buttonRow,
        ],
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

function buildRuntimeValueSelect(key: RuntimeSettingKey, currentValue: number) {
    const meta = RUNTIME_SETTING_META[key];
    const presets = meta?.presets || [currentValue];
    const values = presets.includes(currentValue) ? presets : [...presets, currentValue].sort((a, b) => a - b);

    const container = new ContainerBuilder()
        .setAccentColor(0xfee75c)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`## Set ${meta.label}`),
            new TextDisplayBuilder().setContent(
                [`Current value: **${formatNumber(currentValue)}**`, meta.longDescription].join("\n")
            )
        );

    const select = new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
        new StringSelectMenuBuilder()
            .setCustomId(`settings:runtime:set:${key}`)
            .setPlaceholder("Pick a value")
            .addOptions(
                values.map((v) => ({
                    label: String(v),
                    value: String(v),
                    default: v === currentValue,
                }))
            )
    );

    return {
        components: [container, select],
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

export default {
    data: new SlashCommandBuilder()
        .setName("settings")
        .setDescription("View and change Sophia's runtime settings")
        .setContexts(0, 1, 2)
        .setIntegrationTypes(0)
        .setDMPermission(false),
    async execute(interaction: ChatInputCommandInteraction, _client: BotClient) {
        await SecurityService.initialize();

        if (!SecurityService.isAdmin(interaction.user.id)) {
            await interaction.reply({
                content: "❌ You don't have permission to use this command.",
                flags: MessageFlags.Ephemeral as const,
            });
            return;
        }

        const settings = SettingsService.load();
        await interaction.reply(buildSettingsPanel(settings));
    },
};

export function handleSettingsInteraction(customId: string): {
    type: "profile" | "runtime_pick" | "runtime_set" | "auto_approve_writes_toggle" | "debug_toggle" | "reset" | null;
    key?: RuntimeSettingKey;
} {
    if (customId === "settings:profile") return { type: "profile" };
    if (customId === "settings:runtime") return { type: "runtime_pick" };
    if (customId === "settings:auto-approve-writes:toggle") return { type: "auto_approve_writes_toggle" };
    if (customId === "settings:debug:toggle") return { type: "debug_toggle" };
    if (customId === "settings:reset") return { type: "reset" };
    if (customId.startsWith("settings:runtime:set:")) {
        return {
            type: "runtime_set",
            key: customId.slice("settings:runtime:set:".length) as RuntimeSettingKey,
        };
    }
    return { type: null };
}

export { buildSettingsPanel, buildRuntimeValueSelect };
