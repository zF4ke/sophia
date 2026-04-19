import {
    ChatInputCommandInteraction,
    SlashCommandBuilder,
} from "discord.js";
import type { BotClient } from "@/shared/appTypes";
import approvals from "@/discord/commands/system/uitest/approvals";
import settings from "@/discord/commands/system/uitest/settings";

const SUBCOMMANDS = [approvals, settings];
const uitestBuilder = new SlashCommandBuilder()
    .setName("uitest")
    .setDescription("[dev] Visual previews for Discord UI experiments")
    .setContexts(0, 1, 2)
    .setIntegrationTypes(0)
    .setDMPermission(false);

for (const subcommand of SUBCOMMANDS) {
    uitestBuilder.addSubcommand((sub) => subcommand.register(sub));
}

export = {
    data: uitestBuilder,
    async execute(interaction: ChatInputCommandInteraction, client: BotClient) {
        const subcommandName = interaction.options.getSubcommand();
        const subcommand = SUBCOMMANDS.find((candidate) => candidate.name === subcommandName);
        if (!subcommand) {
            throw new Error(`Unknown /uitest subcommand "${subcommandName}".`);
        }

        await subcommand.execute(interaction, client);
    },
};
