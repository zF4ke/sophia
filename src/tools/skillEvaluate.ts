import { z } from "zod";
import { SkillStore } from "@/memory/SkillStore";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { readableToolRecords } from "@/runtime/sourceEvidence";
import { readableSkills } from "./skills";
import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/runtime/PromptRegistry";
import { readModelProfiles } from "@/app/modelProfiles";
import { SettingsService } from "@/app/SettingsService";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";

const assessmentSchema = z.object({ passed: z.boolean(), findings: z.array(z.string().min(1).max(500)).min(1).max(10) });
export const skillEvaluateTool: ToolDefinition = {
    name: T.skill_evaluate, catalog: { effect: "read", description: "Evaluate a learned procedure against a completed owned task without executing its steps.", evidenceRole: "discovery_only" },
    schema: { description: "Review an owned skill revision using demonstrated successful calls from a completed task in this location. Defaults to the learned skill's originating task. Records a content-bound assessment; it neither promotes nor executes the procedure. Learned drafts need a passing assessment before skill_save can mark them ready. Changed content requires another evaluation.", parameters: { type: "object", properties: { id: { type: "string" }, revision: { type: "integer", minimum: 1 }, task_id: { type: "string" } }, required: ["id", "revision"] } },
    capability: { description: "Evaluate procedure evidence.", sideEffectLevel: "none", inputSchema: z.object({ id: z.string(), revision: z.number().int().positive(), task_id: z.string().optional() }), outputSchema: z.any(), authRequirements: [], costClass: "normal", latencyClass: "slow", preconditions: [], postconditions: [],
        async run(context, args) {
            const audience = { actorId: context.actorId ?? "", guildId: context.guild?.id ?? null, channelId: context.currentChannelId ?? null };
            const skill = await SkillStore.load(audience, String(args.id));
            if (!skill || skill.ownerId !== audience.actorId || skill.revision !== args.revision) throw new Error("Skill is unavailable, owned by someone else, or its revision changed.");
            if (!(await readableSkills(context, [skill])).length) throw new Error("The procedure's sources are unavailable to this audience.");
            const taskId = String(args.task_id ?? skill.provenance?.taskId ?? "");
            const completed = await taskStore.completedResult(taskId, audience.actorId, audience.channelId ?? "", audience.guildId);
            if (!completed) throw new Error("Evaluation requires a completed task owned by this requester in this location.");
            const records = await readableToolRecords(await taskStore.toolRuns(taskId, audience.actorId, audience.channelId ?? "", audience.guildId), context.guild, audience.actorId, { client: context.client, privateResponse: context.privateResponse, destinationChannelId: context.currentChannelId });
            if (records.some(record => record.sourceUnavailable)) throw new Error("A source of this task is no longer available. Use a completed task with current accessible evidence.");
            const successful = records.filter(record => !record.blocked && record.output && !record.output.errorMessage);
            const demonstrated = new Set(successful.map(record => record.tool as string));
            const missing = skill.capabilities.filter(name => !demonstrated.has(name));
            const profile = readModelProfiles().profiles[context.modelProfileName ?? SettingsService.load().modelProfile];
            if (!profile) throw new Error("Evaluation model profile is unavailable.");
            let assessment = { passed: false, findings: missing.map(name => `No accessible successful evidence for ${name}.`) };
            if (!missing.length) {
                const selected = successful.filter((record, index) => index < 20 || successful.findIndex(other => other.tool === record.tool) === index);
                assessment = assessmentSchema.parse(await ModelGateway.generateJson<unknown>([
                    { role: "system", content: PromptRegistry.load("memory/skill_evaluation") },
                    { role: "user", content: JSON.stringify({ skill, completedAnswer: completed.answer.slice(0, 8000), evidence: selected.map(record => ({ tool: record.tool, summary: record.summary.slice(0, 800) })) }) },
                ], { passed: false, findings: ["No valid model assessment returned."] }, { profile, maxOutputTokens: 1400, traceContext: { traceLabel: "skill-evaluation", questionPreview: skill.name } }));
            }
            context.execution?.checkpoint();
            if (context.authorize && await context.authorize("none") === "deny") throw new Error("Evaluation authority changed.");
            const data = await SkillStore.recordEvaluation(audience, skill, { taskId, model: missing.length ? "capability-check" : profile.chatModel, ...assessment });
            return { tool: T.skill_evaluate, summary: `Skill ${skill.id}, revision ${skill.revision}: ${assessment.passed ? "evaluation passed" : "revision required"}. Assessment ${data.id}.`, data };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "📋", labelPt: "Avaliar procedimento" },
};
