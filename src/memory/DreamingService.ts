import { PermissionFlagsBits, type Client } from "discord.js";
import { z } from "zod";
import { knowledgeStore, type DreamInput } from "./KnowledgeStore";
import { ActiveRequestTracker } from "@/app/ActiveRequestTracker";
import { BackgroundLoop } from "@/app/BackgroundLoop";
import { SettingsService } from "@/app/SettingsService";
import { AccessPolicy } from "@/security/AccessPolicy";
import { ModelGateway } from "@/ai/ModelGateway";
import { ModelUsage } from "@/ai/ModelUsage";
import { PromptRegistry } from "@/runtime/PromptRegistry";
import { SkillStore } from "./SkillStore";
import { assertDerivedSources } from "@/security/DerivedSources";

const resultSchema = z.object({ memories: z.array(z.object({ key: z.string().min(2).max(64), value: z.string().min(2).max(2000), scope: z.enum(["channel", "preference"]).optional() })).max(5),
    skill: z.object({ name: z.string().min(2).max(80), description: z.string().min(5).max(500), instructions: z.string().min(10).max(4000), capabilities: z.array(z.string()).max(20), examples: z.array(z.string().max(500)).max(3) }).optional() });

export class DreamingService {
    private static readonly loop = new BackgroundLoop();
    private static busy = false;
    static start(client: Client): void {
        this.loop.start(() => this.tick(input => this.authorized(client, input)), () => SettingsService.load().memory.dreamIntervalMs, error => console.error("[Dreaming]", error));
    }
    static stop() { this.loop.stop(); }
    private static async authorized(client: Client, input: DreamInput): Promise<boolean> {
        const a = input.audience;
        const guild = a.guildId ? await client.guilds.fetch(a.guildId).catch(() => null) : null;
        if (a.guildId && !guild) return false;
        if (await AccessPolicy.decide(a.actorId, guild, "none") === "deny") return false;
        try { await assertDerivedSources({ actorId: a.actorId, guild, currentChannelId: a.channelId, privateResponse: a.privateResponse, client, question: "Memory consolidation" }, input.sources); }
        catch { return false; }
        if (!guild) return true;
        const channel = a.channelId ? await guild.channels.fetch(a.channelId).catch(() => null) : null;
        const member = await guild.members.fetch({ user: a.actorId, force: true }).catch(() => null);
        return Boolean(member && channel?.permissionsFor(member)?.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.ReadMessageHistory]));
    }
    static async tick(authorize: (input: DreamInput) => Promise<boolean>): Promise<void> {
        if (this.busy || !ActiveRequestTracker.isIdle() || !SettingsService.load().memory.dreamingEnabled) return;
        this.busy = true;
        const release = ActiveRequestTracker.begin();
        try {
            const job = await knowledgeStore.nextDream();
            if (!job) return;
            try {
                if (!await authorize(job.input)) { await knowledgeStore.finishDream(job.id, []); return; }
                const result = resultSchema.parse(await ModelUsage.scope({ taskId: job.input.taskId, actorId: job.input.audience.actorId, job: job.id }, async () => ModelGateway.generateJson<unknown>([
                    { role: "system", content: PromptRegistry.load("memory/dreaming") },
                    { role: "user", content: JSON.stringify({ question: job.input.question, answer: job.input.answer, toolEvidence: job.input.toolEvidence ?? [], priorMemory: await knowledgeStore.dreamContext(job.input.audience) }) },
                ], { memories: [] }, { maxOutputTokens: 1600, traceContext: { traceLabel: "dreaming", questionPreview: "Idle memory consolidation" } })));
                if (!SettingsService.load().memory.dreamingEnabled || !await authorize(job.input)) { await knowledgeStore.finishDream(job.id, []); return; }
                if (result.skill && job.input.taskId && job.input.toolEvidence?.some(item => item.succeeded)) {
                    const demonstrated = new Set(job.input.toolEvidence.filter(item => item.succeeded).map(item => item.tool));
                    if (result.skill.capabilities.length && result.skill.capabilities.every(name => demonstrated.has(name))) {
                        await SkillStore.draftFromDream(job.input.audience, { ...result.skill, status: "draft" }, { dreamId: job.id, taskId: job.input.taskId, sources: job.input.sources });
                    }
                }
                await knowledgeStore.finishDream(job.id, result.memories);
            } catch (error) { await knowledgeStore.finishDream(job.id, [], error instanceof Error ? error.message : String(error)); }
        } finally { release(); this.busy = false; }
    }
}
