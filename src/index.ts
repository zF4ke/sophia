require("dotenv").config();

import { Client, GatewayIntentBits, Collection, Partials } from "discord.js";
import { loadEvents } from "./handlers/eventHandler";
import { loadCommands } from "./handlers/commandHandler";
import { getAppConfig } from "@/config/AppConfig";
import type { BotClient } from "@/types/app";

const { Guilds, GuildMembers, GuildMessages, MessageContent } =
    GatewayIntentBits;
const { User, Message, GuildMember, ThreadMember, Channel } = Partials;

const client = new Client({
    intents: [Guilds, GuildMembers, GuildMessages, MessageContent],
    partials: [User, Message, GuildMember, ThreadMember, Channel],
}) as BotClient;

client.commands = new Collection();

const config = getAppConfig();
loadEvents(client);
client.login(config.discordToken).then(async () => {
    await loadCommands(client);
});

//antiCrash
process.on("uncaughtException", function (error) {
    console.error(error.stack);
});

export default client;
export type { BotClient };
