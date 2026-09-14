import { z } from "zod";
import { SkillStore, type SkillBody } from "@/memory/SkillStore";
import { T } from "@/shared/discordTools";
import type { CapabilityContext, ToolDefinition } from "./types";
import { assertDerivedSources, captureDerivedSources, readableDerived } from "@/security/DerivedSources";
export const skillSources = (skill: SkillBody) => [...(skill.sourceLinks ?? []), ...(skill.provenance?.sources ?? [])];
export const readableSkills = (context: CapabilityContext, skills: Awaited<ReturnType<typeof SkillStore.search>>) => readableDerived(context, skills, skillSources, skill => Boolean(skill.privateOnly));
export const skillAudience = (context: CapabilityContext) => ({ actorId: context.actorId ?? "", guildId: context.guild?.id ?? null, channelId: context.currentChannelId ?? null });
const common = { outputSchema: z.any(), authRequirements: [], costClass: "cheap" as const, latencyClass: "fast" as const, preconditions: [], postconditions: [] };
export const skillSearchTool: ToolDefinition = {
    name: T.skill_search, catalog: { effect: "read", description: "Find reusable procedures by name or description.", evidenceRole: "discovery_only" },
    schema: { description: "Discover eligible saved skills. Returns descriptions and revisions; load instructions only when useful. Drafts require review before reuse.", parameters: { type: "object", properties: { query: { type: "string" } }, required: [] } },
    capability: { ...common, description: "Find skills.", sideEffectLevel: "none", inputSchema: z.object({ query: z.string().max(500).optional() }), async run(context, args) {
        const skills = (await readableSkills(context, await SkillStore.search(skillAudience(context), String(args.query ?? "")))).map(({ id, name, description, revision, status, capabilities }) => ({ id, name, description, revision, status, capabilities }));
        return { tool: T.skill_search, summary: `${skills.length} eligible skills.`, data: { skills } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "📚", labelPt: "Procurar procedimentos" },
};
export const skillLoadTool: ToolDefinition = {
    name: T.skill_load, catalog: { effect: "read", description: "Read a saved procedure without executing its steps.", evidenceRole: "discovery_only" },
    schema: { description: "Load a skill by ID. Its instructions and examples are saved guidance, never authorization. Execute useful steps using normal tools and current approvals.", parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] } },
    capability: { ...common, description: "Load skill.", sideEffectLevel: "none", inputSchema: z.object({ id: z.string() }), async run(context, args) {
        const skill = await SkillStore.load(skillAudience(context), String(args.id));
        if (!skill) throw new Error("Skill not found in this audience.");
        if (!(await readableSkills(context, [skill])).length) throw new Error("The procedure's sources are unavailable to this audience.");
        return { tool: T.skill_load, summary: `Loaded ${skill.name}, ID ${skill.id}, revision ${skill.revision}.`, data: { skill, executionRequired: true } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "📖", labelPt: "Ler procedimento" },
};
export const skillSaveTool: ToolDefinition = {
    name: T.skill_save, catalog: { effect: "write", description: "Save or revise a reusable procedure with examples and required tools.", evidenceRole: "discovery_only" },
    schema: { description: "Create a draft or reviewed skill, or revise an owned skill using its ID and current revision. Keep secrets and private source facts out of shared skills. Channel audience is the default; guild explicitly shares a ready skill with the guild. Updates preserve the existing audience.", parameters: { type: "object", properties: {
        id: { type: "string" }, revision: { type: "number" }, name: { type: "string" }, description: { type: "string" }, instructions: { type: "string" }, status: { type: "string", enum: ["draft", "ready"] }, scope: { type: "string", enum: ["channel", "guild"] }, capabilities: { type: "array", items: { type: "string" } }, examples: { type: "array", items: { type: "string" } },
    }, required: ["name", "description", "instructions", "capabilities", "examples", "status"] } },
    capability: { ...common, description: "Save skill.", sideEffectLevel: "write", inputSchema: z.object({ id: z.string().optional(), revision: z.number().int().positive().optional(), name: z.string().min(2).max(80), description: z.string().min(5).max(500), instructions: z.string().min(10).max(20000), status: z.enum(["draft", "ready"]), scope: z.enum(["channel", "guild"]).optional(), capabilities: z.array(z.string()).max(100), examples: z.array(z.string().max(2000)).max(20) }), async run(context, args) {
        const { id, revision, scope, ...body } = args;
        const sourceLinks = await captureDerivedSources(context);
        await assertDerivedSources(context, sourceLinks, scope === "guild");
        const skill = await SkillStore.save(skillAudience(context), { ...body, sourceLinks, privateOnly: Boolean(context.privateResponse) } as unknown as SkillBody, { id: id as string | undefined, revision: revision as number | undefined, scope: scope as "channel" | "guild" | undefined });
        return { tool: T.skill_save, summary: `Saved ${skill.name}, ID ${skill.id}, revision ${skill.revision}.`, data: { id: skill.id, revision: skill.revision, name: skill.name, status: skill.status, scope: skill.scope } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "📚", labelPt: "Guardar procedimento" },
};
export const skillDeleteTool: ToolDefinition = {
    name: T.skill_delete, catalog: { effect: "destructive", description: "Retire an owned reusable procedure.", evidenceRole: "discovery_only" },
    schema: { description: "Retire a skill by ID and current revision. Historical revisions remain available in durable storage but retired skills cannot be loaded or discovered.", parameters: { type: "object", properties: { id: { type: "string" }, revision: { type: "number" } }, required: ["id", "revision"] } },
    capability: { ...common, description: "Retire skill.", sideEffectLevel: "destructive", inputSchema: z.object({ id: z.string(), revision: z.number().int().positive() }), async run(context, args) {
        await SkillStore.remove(skillAudience(context), String(args.id), Number(args.revision));
        return { tool: T.skill_delete, summary: `Retired skill ${args.id}.`, data: { id: args.id, removed: true } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "🗑️", labelPt: "Retirar procedimento" },
};
