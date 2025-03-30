import { SecurityService } from "@/services/SecurityService";
import { TDiscordClient } from "../..";

module.exports = {
    name: "ready",
    once: true,
    async execute(client: TDiscordClient) {
        if (!client.user) return;
    
        console.log(`${client.user.username} está online.`);
        
                    
        // Initialize security service
        await SecurityService.initialize();
    }
}
