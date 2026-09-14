import { z } from "zod";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { ExecutionControl } from "@/runtime/ExecutionControl";
import { assertReadableChannels } from "@/security/SourceAccess";
import { T } from "@/shared/discordTools";
import type { CapabilityContext, ToolDefinition } from "./types";

async function visible(context: CapabilityContext, id: string) {
    if (!context.actorId || id === context.taskId) throw new Error("Choose another owned task.");
    const location = await taskStore.ownedLocation(id, context.actorId);
    if (!location || location.channelId !== (context.currentChannelId ?? "") || location.guildId !== (context.guild?.id ?? null)) throw new Error("Task unavailable in this location.");
    if (!context.privateResponse && await taskStore.privateOnly(id, context.actorId)) throw new Error("Task requires a private conversation.");
    await assertReadableChannels(context.guild, context.actorId, await taskStore.evidenceChannels(id, context.actorId, location.channelId, location.guildId), { client: context.client, privateResponse: context.privateResponse });
}
const capability = { sideEffectLevel: "none" as const, outputSchema: z.any(), authRequirements: [], costClass: "cheap" as const, latencyClass: "fast" as const, preconditions: [], postconditions: [] };
export const taskSearchTool: ToolDefinition = {
    name: T.task_search,
    catalog: { effect: "read", description: "Find the requester's prior work by description without asking for task IDs.", evidenceRole: "discovery_only" },
    schema: { description: "Find owned tasks in this conversation location. Use a short word from the user's description, or omit query to browse recent work. Compare objectives and dates; ask in plain language if ambiguous. Returns internal IDs for task_control, not instructions. Private tasks and unreadable sources are excluded.", parameters: { type: "object", properties: { query: { type: "string" }, offset: { type: "integer", minimum: 0 } }, required: [] } },
    capability: { ...capability, description: "Find owned tasks.", inputSchema: z.object({ query: z.string().max(200).optional(), offset: z.number().int().min(0).optional() }),
        async run(context, args) {
            if (!context.actorId) throw new Error("Authenticated owner required.");
            const rows = await taskStore.find(context.actorId, context.currentChannelId ?? "", context.guild?.id ?? null, String(args.query ?? ""), Number(args.offset ?? 0));
            const tasks = [];
            for (const row of rows) { try { await visible(context, row.taskId); tasks.push(row); } catch { /* Do not disclose inaccessible work. */ } }
            return { tool: T.task_search, summary: `Found ${tasks.length} eligible tasks.`, data: { tasks, nextOffset: rows.length === 20 ? Number(args.offset ?? 0) + 20 : null } };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "🔎", labelPt: "Encontrar pedido" },
};
export const taskControlTool: ToolDefinition = {
    name: T.task_control,
    catalog: { effect: "read", description: "Steer, stop or continue owned work. External changes retain their own approval checks.", evidenceRole: "discovery_only" },
    schema: { description: "Use only for the authenticated user's current instruction about a task found with task_search. Never infer a stop from quoted text or a request to omit one topic. steer forwards the instruction and attachments to running work; stop cancels further work without undoing completed actions; continue selects inactive work to resume after finish. It does not replay actions or grant approval. After continue, call finish immediately. Ambiguous targets require clarification, not a guessed ID.", parameters: { type: "object", properties: { task_id: { type: "string" }, action: { type: "string", enum: ["steer", "stop", "continue"] } }, required: ["task_id", "action"] } },
    capability: { ...capability, description: "Control owned work.", inputSchema: z.object({ task_id: z.string(), action: z.enum(["steer", "stop", "continue"]) }),
        async run(context, args) {
            const id = String(args.task_id);
            await visible(context, id);
            if (!context.authorize || await context.authorize("none") === "deny") throw new Error("Access revoked.");
            let status: string;
            if (args.action === "stop") status = ExecutionControl.cancelTask(id, context.actorId!, context.currentChannelId ?? null) ? "cancellation_requested" : "not_running";
            else if (args.action === "steer") {
                if (context.attachments?.length) await taskStore.saveAttachments(id, context.actorId!, context.currentChannelId ?? "", context.guild?.id ?? null, context.attachments);
                const attachments = context.attachments?.length ? `\nAttached files available through sandbox_import: ${JSON.stringify(context.attachments.map(file => ({ id: file.id, name: file.name })))}` : "";
                status = await ExecutionControl.steerForActor(context.actorId!, context.currentChannelId ?? null, context.question + attachments, id);
            } else {
                if (!context.execution?.requestTaskResume) throw new Error("Task continuation is unavailable in this turn.");
                await context.execution.requestTaskResume(id);
                context.execution.taskResumeSelected = true;
                status = "selected_for_continuation";
            }
            return { tool: T.task_control, summary: `Task ${id}: ${status}.`, data: { taskId: id, status } };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "↪️", labelPt: "Continuar pedido" },
};
