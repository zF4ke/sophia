import { Client, Collection, GatewayIntentBits, Partials } from "discord.js";
import type { BotClient } from "@/shared/appTypes";

export function createClient(): BotClient {
    const { Guilds, GuildMembers, GuildMessages, MessageContent } = GatewayIntentBits;
    const { User, Message, GuildMember, ThreadMember, Channel } = Partials;

    const client = new Client({
        intents: [Guilds, GuildMembers, GuildMessages, MessageContent],
        partials: [User, Message, GuildMember, ThreadMember, Channel],
    }) as BotClient;

    client.commands = new Collection();
    return client;
}
