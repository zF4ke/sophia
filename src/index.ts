require("dotenv").config();

import { Client, GatewayIntentBits, Collection, Partials } from "discord.js";
import { loadEvents } from "./handlers/eventHandler";
import { loadCommands } from "./handlers/commandHandler";
import { MessageService } from "./services/MessageService";

const { Guilds, GuildMembers, GuildMessages, MessageContent } =
    GatewayIntentBits;
const { User, Message, GuildMember, ThreadMember, Channel } = Partials;

type TDiscordClient = Client & {
    commands: Collection<string, any>;
};

const client = new Client({
    intents: [Guilds, GuildMembers, GuildMessages, MessageContent],
    partials: [User, Message, GuildMember, ThreadMember, Channel],
}) as TDiscordClient;

client.commands = new Collection();

client.login(process.env.CLIENT_TOKEN).then(() => {    
    loadEvents(client);
    loadCommands(client);
});

//antiCrash
process.on("uncaughtException", function (error) {
    console.error(error.stack);
});

export default client;

export type { TDiscordClient };