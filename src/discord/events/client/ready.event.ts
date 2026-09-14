import { DiscordBackfillCrawler } from "@/discord/live/DiscordBackfillCrawler";
import { SecurityService } from "@/security/SecurityService";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { DreamingService } from "@/memory/DreamingService";
import { scheduleStore } from "@/runtime/scheduling/ScheduleStore";
import { Scheduler } from "@/runtime/scheduling/Scheduler";
import type { BotClient } from "@/shared/appTypes";

export = {
    name: "clientReady",
    once: true,
    async execute(client: BotClient) {
        if (!client.user) return;

        console.log(`${client.user.username} is online.`);
        await SecurityService.initialize();
        await taskStore.initialize();
        await scheduleStore.initialize();
        Scheduler.start(client);
        DreamingService.start(client);
        await DiscordBackfillCrawler.start(client);
    },
};
