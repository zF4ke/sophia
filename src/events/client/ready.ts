import { SecurityService } from "@/services/SecurityService";
import { TDiscordClient } from "../..";
import app from "@/api/app";

module.exports = {
    name: "ready",
    once: true,
    async execute(client: TDiscordClient) {
        if (!client.user) return;
    
        console.log(`${client.user.username} está online.`);
        
        // Initialize security service
        await SecurityService.initialize();

        // Run API
        const port = process.env.PORT || 3002;

        app.listen(port, () => {
            console.log(`API rodando em http://localhost:${port}`); 
        });
    }
}
