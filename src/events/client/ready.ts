import { SecurityService } from "@/services/SecurityService";
import { getAppConfig } from "@/config/AppConfig";
import type { BotClient } from "@/types/app";
import app from "@/api/app";

export = {
    name: "ready",
    once: true,
    async execute(client: BotClient) {
        if (!client.user) return;
    
        console.log(`${client.user.username} está online.`);
        
        // Initialize security service
        await SecurityService.initialize();

        // Run API
        const { port } = getAppConfig();

        app.listen(port, () => {
            console.log(`API rodando em http://localhost:${port}`); 
        });
    }
};
