import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type { CapabilityContext } from "@/tools/types";
import { taskStore } from "./TaskStore";
import type { WorkingState } from "./TaskWorkspace";

export async function getWorkingState(context: Pick<CapabilityContext, "taskId" | "actorId" | "currentChannelId" | "guild">): Promise<WorkingState> {
    // Compatibility for direct tool callers. Every Discord runtime supplies a task.
    if (!context.taskId) return DiscordMemoryService;
    if (!context.actorId) throw new Error("Task owner is missing.");
    return taskStore.workspace(context.taskId, context.actorId, context.currentChannelId ?? "", context.guild?.id ?? null);
}
