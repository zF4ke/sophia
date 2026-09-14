import { z } from "zod";
import { scheduleStore, type ScheduleSpec } from "@/runtime/scheduling/ScheduleStore";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";
import { skillAudience } from "./skills";
const parameters = { type: "object", properties: {
    notification_policy: { type: "string", enum: ["always", "conditional"], description: "Use conditional for monitoring: keep quiet when the prompt's condition has no reportable change. Failures and required action remain reportable. Default always." },
    name: { type: "string" }, prompt: { type: "string", description: "The complete future task and what should be delivered here. This approval covers the scheduled result message, not arbitrary future mutations." }, cadence: { type: "string", enum: ["once", "interval", "daily", "weekly"] }, timezone: { type: "string", description: "IANA timezone, e.g. Europe/Lisbon." }, first_run_at: { type: "string", description: "ISO timestamp with UTC offset. Required for once/interval." }, interval_minutes: { type: "number" }, local_time: { type: "string", description: "HH:mm, required for daily/weekly." }, weekday: { type: "number", description: "1–7, Monday–Sunday, required for weekly." },
}, required: ["name", "prompt", "cadence", "timezone"] };
const inputSchema = z.object({ notification_policy: z.enum(["always", "conditional"]).optional(), name: z.string().min(2).max(100), prompt: z.string().min(5).max(20000), cadence: z.enum(["once", "interval", "daily", "weekly"]), timezone: z.string().min(1).max(100), first_run_at: z.string().optional(), interval_minutes: z.number().int().min(1).max(525600).optional(), local_time: z.string().optional(), weekday: z.number().int().min(1).max(7).optional() });
const common = { outputSchema: z.any(), authRequirements: [], costClass: "cheap" as const, latencyClass: "fast" as const, preconditions: [], postconditions: [] };
function spec(args: z.infer<typeof inputSchema>): ScheduleSpec { return { notificationPolicy: args.notification_policy, name: args.name, prompt: args.prompt, cadence: args.cadence, timezone: args.timezone, firstRunAt: args.first_run_at, intervalMinutes: args.interval_minutes, localTime: args.local_time, weekday: args.weekday }; }
export const scheduleCreateTool: ToolDefinition = {
    name: T.schedule_create, catalog: { effect: "write", description: "Schedule an explicitly requested future task and its result delivery here.", evidenceRole: "discovery_only" },
    schema: { description: "Create future work only when requested. It runs with this authenticated owner's current permissions and delivers its result in this channel. Daily/weekly schedules follow local wall time through DST. Missed recurring occurrences coalesce into one run. Interrupted or uncertain runs pause rather than replay.", parameters },
    capability: { ...common, description: "Schedule future work.", sideEffectLevel: "write", inputSchema, async run(context, args) {
        if (context.privateResponse && context.guild) throw new Error("An ephemeral request cannot silently create public scheduled delivery. Schedule from an enabled DM or an ordinary channel conversation.");
        const schedule = await scheduleStore.save(skillAudience(context), spec(inputSchema.parse(args)));
        return { tool: T.schedule_create, summary: `Scheduled ${schedule.spec.name}, ID ${schedule.id}, for ${new Date(schedule.nextAt).toISOString()} in <#${schedule.owner.channelId}>.`, data: { ...schedule, channelId: schedule.owner.channelId, channelMention: `<#${schedule.owner.channelId}>` } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "🗓️", labelPt: "Agendar pedido" },
};
export const scheduleUpdateTool: ToolDefinition = {
    name: T.schedule_update, catalog: { effect: "write", description: "Revise or reactivate an owned schedule using its current revision.", evidenceRole: "discovery_only" },
    schema: { description: "Replace an owned schedule's instructions and timing, using ID and current revision. Preserves owner and destination. A new future occurrence is required to reactivate paused work; this never replays an interrupted delivery.", parameters: { ...parameters, properties: { ...parameters.properties, id: { type: "string" }, revision: { type: "number" } }, required: [...parameters.required, "id", "revision"] } },
    capability: { ...common, description: "Revise schedule.", sideEffectLevel: "write", inputSchema: inputSchema.extend({ id: z.string(), revision: z.number().int().positive() }), async run(context, args) {
        if (context.privateResponse && context.guild) throw new Error("An ephemeral request cannot silently change public scheduled delivery. Revise it from the original ordinary channel conversation.");
        const schedule = await scheduleStore.save(skillAudience(context), spec(inputSchema.parse(args)), { id: String(args.id), revision: Number(args.revision) });
        return { tool: T.schedule_update, summary: `Updated schedule ${schedule.id}, revision ${schedule.revision}, next run ${new Date(schedule.nextAt).toISOString()}.`, data: { ...schedule, channelId: schedule.owner.channelId, channelMention: `<#${schedule.owner.channelId}>` } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "🗓️", labelPt: "Alterar agendamento" },
};
export const scheduleListTool: ToolDefinition = {
    name: T.schedule_list, catalog: { effect: "read", description: "List this owner's schedules and failures in the current location.", evidenceRole: "discovery_only" },
    schema: { description: "Inspect owned scheduled work, its status, revision and next due time. Paused work may need a new schedule time or task review.", parameters: { type: "object", properties: {}, required: [] } },
    capability: { ...common, description: "List schedules.", sideEffectLevel: "none", inputSchema: z.object({}), async run(context) { const schedules = await scheduleStore.list(skillAudience(context)); return { tool: T.schedule_list, summary: `${schedules.length} schedules in this location.`, data: { schedules } }; } }, strategy: { extractEvidence: () => [] }, display: { icon: "🗓️", labelPt: "Ver agendamentos" },
};
export const scheduleCancelTool: ToolDefinition = {
    name: T.schedule_cancel, catalog: { effect: "write", description: "Cancel an owned schedule and stop future runs.", evidenceRole: "discovery_only" },
    schema: { description: "Cancel a schedule by ID and current revision. An active run loses authority at its next checkpoint, and an unstarted delivery is suppressed. A send already in flight may still complete.", parameters: { type: "object", properties: { id: { type: "string" }, revision: { type: "number" } }, required: ["id", "revision"] } },
    capability: { ...common, description: "Cancel schedule.", sideEffectLevel: "write", inputSchema: z.object({ id: z.string(), revision: z.number().int().positive() }), async run(context, args) { await scheduleStore.cancel(skillAudience(context), String(args.id), Number(args.revision)); return { tool: T.schedule_cancel, summary: `Cancelled schedule ${args.id}.`, data: { id: args.id, cancelled: true } }; } }, strategy: { extractEvidence: () => [] }, display: { icon: "⏹️", labelPt: "Cancelar agendamento" },
};
