import { taskStore } from "./TaskStore";
import { knowledgeStore } from "@/memory/KnowledgeStore";
import { SkillStore } from "@/memory/SkillStore";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

export async function forgetTask(taskId: string, actorId: string, channelId: string, guildId: string | null): Promise<void> {
    const requests = await taskStore.forget(taskId, actorId, channelId, guildId);
    // A durable deletion marker makes retries safe if later cleanup fails.
    await SkillStore.forgetTaskDrafts(taskId, actorId);
    await knowledgeStore.forgetTask(taskId, actorId);
    await DiscordMemoryService.forgetRuntimeRequests(requests);
}
