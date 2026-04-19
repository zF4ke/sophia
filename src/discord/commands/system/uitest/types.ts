import type { ChatInputCommandInteraction, SlashCommandSubcommandBuilder } from "discord.js";
import type { BotClient } from "@/shared/appTypes";

export type UiTestSubcommandModule = {
    name: string;
    description: string;
    register(builder: SlashCommandSubcommandBuilder): SlashCommandSubcommandBuilder;
    execute(interaction: ChatInputCommandInteraction, client: BotClient): Promise<void>;
};
