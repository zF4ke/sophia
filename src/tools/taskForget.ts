import { z } from "zod";
import { forgetTask } from "@/runtime/tasks/TaskRetention";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";

export const taskForgetTool: ToolDefinition = {
    name: T.task_forget,
    catalog: { effect: "destructive", description: "Remove an inactive owned task's working data and pending learning.", evidenceRole: "discovery_only" },
    schema: { description: "Forget an exact inactive task owned by this requester in this location. Removes its notes, goals, evidence, files, usage, action records, runtime traces and pending learning drafts. Resolve uncertain actions first. Existing durable memories, approved skills, original Discord messages and operator backups have separate retention controls. The current active task cannot delete itself.", parameters: { type: "object", properties: { task_id: { type: "string" } }, required: ["task_id"] } },
    capability: { description: "Forget owned inactive work.", sideEffectLevel: "destructive", inputSchema: z.object({ task_id: z.string() }), outputSchema: z.any(), authRequirements: [], costClass: "cheap", latencyClass: "fast", preconditions: [], postconditions: [],
        async run(context, args) {
            if (!context.actorId) throw new Error("An authenticated owner is required.");
            const taskId = String(args.task_id);
            await forgetTask(taskId, context.actorId, context.currentChannelId ?? "", context.guild?.id ?? null);
            return { tool: T.task_forget, summary: `Forgot task ${taskId} and its working data.`, data: { taskId, forgotten: true } };
        } },
    strategy: { extractEvidence: () => [] }, display: { icon: "🗑️", labelPt: "Esquecer pedido" },
};
