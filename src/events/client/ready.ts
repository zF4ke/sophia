import { TDiscordClient } from "../..";

module.exports = {
    name: "ready",
    once: true,
    async execute(client: TDiscordClient) {
        if (!client.user) return;
    
        console.log(`${client.user.username} está online.`);
    }
}
