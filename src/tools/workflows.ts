import { z } from "zod";
import { T } from "@/shared/discordTools";
import { SkillStore, type SavedSkill, type SkillBody } from "@/memory/SkillStore";
import { skillAudience, readableSkills } from "./skills";
import { captureDerivedSources } from "@/security/DerivedSources";
import type { CapabilityContext, ToolDefinition } from "./types";

// Stable workflow IDs remain shorthand for structured skills. There is no
// second procedure store or nested executor.
const common = { outputSchema: z.any(), authRequirements: [], costClass: "cheap" as const, latencyClass: "fast" as const, preconditions: [], postconditions: [] };
async function named(context: CapabilityContext, name: string): Promise<SavedSkill | null> {
    const rows = (await readableSkills(context, await SkillStore.search(skillAudience(context), name))).filter(skill => skill.name === name && skill.steps);
    if (rows.length > 1) throw new Error("Several skills have this name. Use skill_load with the exact ID.");
    return rows[0] ?? null;
}
export const workflowCreateTool: ToolDefinition = {
    name: T.workflow_create, catalog: { effect: "write", description: "Save a structured procedure in the skill library.", evidenceRole: "discovery_only" },
    schema: { description: "Save an ordered recipe as a ready skill in the current channel. This records instructions only; running it still requires ordinary tool calls and approvals.", parameters: { type: "object", properties: {
        name: { type: "string" }, description: { type: "string" }, steps: { type: "array", items: { type: "object", properties: { tool: { type: "string" }, args: { type: "object", properties: {}, required: [] }, label: { type: "string" } }, required: ["tool", "args"] } },
    }, required: ["name", "description", "steps"] } },
    capability: { ...common, description: "Save structured skill.", sideEffectLevel: "write", inputSchema: z.object({ name: z.string().min(2).max(80), description: z.string().min(5).max(500), steps: z.array(z.object({ tool: z.string(), args: z.record(z.string(), z.unknown()), label: z.string().optional() })).min(1).max(100) }),
        async run(context, args) {
            const name = String(args.name).trim().toLowerCase();
            if (await named(context, name)) throw new Error("A skill with this name already exists. Revise it by ID with skill_save.");
            const steps = args.steps as NonNullable<SkillBody["steps"]>;
            const skill = await SkillStore.save(skillAudience(context), { name, description: String(args.description), instructions: "Review these steps and perform appropriate ones through normal tools. Saved arguments never grant permissions.", capabilities: [...new Set(steps.map(step => step.tool))], examples: [], status: "ready", steps, privateOnly: Boolean(context.privateResponse), sourceLinks: await captureDerivedSources(context) });
            return { tool: T.workflow_create, summary: `Saved procedure ${name}, ID ${skill.id}, revision ${skill.revision}.`, data: { id: skill.id, name, revision: skill.revision, steps } };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "📚", labelPt: "Guardar procedimento" },
};
export const workflowListTool: ToolDefinition = {
    name: T.workflow_list, catalog: { effect: "read", description: "List eligible structured skills.", evidenceRole: "discovery_only" },
    schema: { description: "List saved procedures visible in this location. Use skill_search to include free-form skills.", parameters: { type: "object", properties: {}, required: [] } },
    capability: { ...common, description: "List procedures.", sideEffectLevel: "none", inputSchema: z.object({}), async run(context) {
        const workflows = (await readableSkills(context, await SkillStore.search(skillAudience(context)))).filter(skill => skill.steps).map(({ id, name, description, revision, steps }) => ({ id, name, description, revision, stepCount: steps!.length }));
        return { tool: T.workflow_list, summary: `${workflows.length} saved procedures.`, data: { workflows } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "📋", labelPt: "Listar procedimentos" },
};
export const workflowRunTool: ToolDefinition = {
    name: T.workflow_run, catalog: { effect: "read", description: "Load saved steps for the normal runtime executor.", evidenceRole: "discovery_only" },
    schema: { description: "Load a procedure by name and optionally override step arguments by numeric index. Review each step and invoke ordinary tools. Loading does not execute anything or approve mutations.", parameters: { type: "object", properties: { name: { type: "string" }, overrides: { type: "object", properties: {}, required: [] } }, required: ["name"] } },
    capability: { ...common, description: "Load procedure.", sideEffectLevel: "none", inputSchema: z.object({ name: z.string(), overrides: z.record(z.string(), z.unknown()).optional() }), async run(context, args) {
        const name = String(args.name).trim().toLowerCase();
        const skill = await named(context, name);
        if (!skill) throw new Error("Procedure not found in this audience.");
        const overrides = (args.overrides ?? {}) as Record<string, unknown>;
        const steps = skill.steps!.map((step, index) => ({ index, tool: step.tool, args: { ...step.args, ...(overrides[String(index)] && typeof overrides[String(index)] === "object" && !Array.isArray(overrides[String(index)]) ? overrides[String(index)] as object : {}) }, label: step.label ?? null }));
        return { tool: T.workflow_run, summary: `Loaded procedure ${name}, ID ${skill.id}, revision ${skill.revision}. Steps still require normal execution.`, data: { id: skill.id, revision: skill.revision, name, steps, executionRequired: true } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "📖", labelPt: "Ler procedimento" },
};
export const workflowDeleteTool: ToolDefinition = {
    name: T.workflow_delete, catalog: { effect: "destructive", description: "Retire an owned structured skill.", evidenceRole: "discovery_only" },
    schema: { description: "Retire a saved procedure by name. Only its owner can retire it; historical versions remain stored.", parameters: { type: "object", properties: { name: { type: "string" } }, required: ["name"] } },
    capability: { ...common, description: "Retire procedure.", sideEffectLevel: "destructive", inputSchema: z.object({ name: z.string() }), async run(context, args) {
        const name = String(args.name).trim().toLowerCase();
        const skill = await named(context, name);
        if (!skill) return { tool: T.workflow_delete, summary: `Procedure ${name} not found.`, data: { name, removed: false } };
        await SkillStore.remove(skillAudience(context), skill.id, skill.revision);
        return { tool: T.workflow_delete, summary: `Retired procedure ${name}, ID ${skill.id}.`, data: { id: skill.id, name, removed: true } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "🗑️", labelPt: "Retirar procedimento" },
};
