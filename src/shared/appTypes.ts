import type {
    AutocompleteInteraction,
    ButtonInteraction,
    ChatInputCommandInteraction,
    Client,
    Collection,
    ModalSubmitInteraction,
    StringSelectMenuInteraction,
} from "discord.js";
import type { DiscordToolName } from "@/shared/discordTools";

export type SupportedInteraction =
    | ChatInputCommandInteraction
    | AutocompleteInteraction
    | ButtonInteraction
    | StringSelectMenuInteraction
    | ModalSubmitInteraction;

export interface ModelProfile {
    chatModel: string;
    analysisModel: string;
    embeddingModel: string;
    temperature: number;
    maxOutputTokens: number;
    webEnabled?: boolean;
}

export interface ModelProfileConfig {
    defaultProfile: string;
    profiles: Record<string, ModelProfile>;
}

export interface AppConfig {
    discordToken: string;
    openRouterApiKey: string;
    openRouterBaseUrl: string;
    port: number;
    modelProfileName: string;
    modelProfile: ModelProfile;
}

export interface BotCommand {
    data: {
        name: string;
        toJSON(): unknown;
    };
    execute(interaction: ChatInputCommandInteraction, client: BotClient): Promise<void>;
    autocomplete?(
        interaction: AutocompleteInteraction,
        client: BotClient
    ): Promise<void>;
}

export type BotClient = Client & {
    commands: Collection<string, BotCommand>;
};

export interface RetrievedChunk {
    messageId: string;
    channelId: string;
    channelName: string;
    guildId: string | null;
    authorId: string;
    authorName: string;
    content: string;
    createdTimestamp: number;
    jumpLink: string;
    lexicalScore: number;
    semanticScore: number;
    recencyScore: number;
    totalScore: number;
}

export interface AnswerCitation {
    label: string;
    jumpLink: string;
}

export interface DiscordToolResult {
    tool: DiscordToolName | "finish";
    summary: string;
    data: unknown;
}

export interface SearchPlan {
    action: DiscordToolName | "finish";
    arguments: Record<string, string | number | undefined>;
    reason: string;
}

export interface RequestClassification {
    mode: "direct_answer" | "discord_grounded";
    reason: string;
}
