import { SecurityService } from "@/security/SecurityService";
import { getAppConfig } from "@/app/AppConfig";
import type { BotClient } from "@/shared/appTypes";
import app from "@/platform/http/app";

export = {
    name: "clientReady",
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
