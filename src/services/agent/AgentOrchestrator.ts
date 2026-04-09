import type { Guild, User } from "discord.js";
import { ModelGateway } from "@/services/ai/ModelGateway";
import { PromptRegistry } from "@/services/prompt/PromptRegistry";
import { RequestClassifier } from "@/services/agent/RequestClassifier";
import { DiscordToolService } from "@/services/discord/DiscordToolService";
import { DiscordMemoryService } from "@/services/memory/DiscordMemoryService";
import type {
    AnswerCitation,
    DiscordToolResult,
    RequestClassification,
    SearchPlan,
} from "@/types/app";

const MAX_TOOL_STEPS = 4;

export class AgentOrchestrator {
    public static async answerQuestion(options: {
        question: string;
        user: User;
        guild: Guild | null;
        currentChannelId?: string | null;
    }): Promise<{
        answer: string;
        citations: AnswerCitation[];
        classification: RequestClassification;
        toolRuns: DiscordToolResult[];
    }> {
        const classification = await RequestClassifier.classify(options.question);

        if (classification.mode === "direct_answer") {
            const answer = await this.generateDirectAnswer(options.question);
            return {
                answer,
                citations: [],
                classification,
                toolRuns: [],
            };
        }

        const toolRuns = await this.runDiscordToolLoop(options);
        const searchResults = toolRuns
            .filter((result) => result.tool === "search_messages")
            .flatMap((result) => result.data as Array<any>);

        if (!searchResults.length || searchResults[0].totalScore < 0.15) {
            return {
                answer: PromptRegistry.load("guards/insufficient_evidence"),
                citations: [],
                classification,
                toolRuns,
            };
        }

        const evidence = this.formatEvidence(toolRuns);
        const answer = await ModelGateway.generateText([
            { role: "system", content: PromptRegistry.load("system/base") },
            { role: "system", content: PromptRegistry.load("system/grounded") },
            {
                role: "user",
                content: PromptRegistry.render("tasks/synthesize_answer", {
                    question: options.question,
                    evidence,
                }),
            },
        ]);

        const citations = searchResults.slice(0, 3).map((item: any, index: number) => ({
            label: `${item.channelName} · ${item.authorName}`,
            jumpLink: item.jumpLink,
        }));

        toolRuns.forEach((run) => {
            DiscordMemoryService.recordToolRun(
                options.guild?.id || null,
                options.currentChannelId || null,
                options.user.id,
                options.question,
                run.tool,
                run.summary
            );
        });

        return {
            answer,
            citations,
            classification,
            toolRuns,
        };
    }

    private static async generateDirectAnswer(question: string): Promise<string> {
        return ModelGateway.generateText([
            { role: "system", content: PromptRegistry.load("system/base") },
            { role: "user", content: question },
        ]);
    }

    private static async runDiscordToolLoop(options: {
        question: string;
        user: User;
        guild: Guild | null;
    }): Promise<DiscordToolResult[]> {
        const toolRuns: DiscordToolResult[] = [];

        for (let step = 0; step < MAX_TOOL_STEPS; step += 1) {
            const next = await this.planNextTool(options.question, toolRuns);
            if (next.action === "finish") {
                break;
            }

            const result = await this.executeTool(next, options.guild, options.question);
            toolRuns.push(result);

            if (result.tool === "search_messages") {
                const chunks = result.data as Array<any>;
                if (chunks.length && chunks[0].totalScore >= 0.45) {
                    break;
                }
            }
        }

        return toolRuns;
    }

    private static async planNextTool(
        question: string,
        toolRuns: DiscordToolResult[]
    ): Promise<SearchPlan> {
        const fallback: SearchPlan = {
            action: toolRuns.length ? "finish" : "search_messages",
            arguments: {
                query: question,
                scope: "guild",
                limit: 8,
            },
            reason: toolRuns.length
                ? "Stop after the initial evidence pass."
                : "Start with broad message search.",
        };

        const toolResults = toolRuns.length
            ? JSON.stringify(toolRuns, null, 2)
            : "No tool results yet.";

        return ModelGateway.generateJson<SearchPlan>(
            [
                { role: "system", content: "Return strict JSON only." },
                {
                    role: "user",
                    content: PromptRegistry.render("tasks/plan_discord_search", {
                        question,
                        tool_results: toolResults,
                    }),
                },
            ],
            fallback
        );
    }

    private static async executeTool(
        plan: SearchPlan,
        guild: Guild | null,
        question: string
    ): Promise<DiscordToolResult> {
        switch (plan.action) {
            case "search_messages":
                return DiscordToolService.searchMessages(
                    String(plan.arguments.query || question),
                    guild,
                    Number(plan.arguments.limit || 8)
                );
            case "read_message_thread":
                return DiscordToolService.readMessageThread(String(plan.arguments.messageId || ""));
            case "read_channel_summary":
                return DiscordToolService.readChannelSummary(String(plan.arguments.channelId || ""));
            case "list_relevant_channels":
                return DiscordToolService.listRelevantChannels(
                    String(plan.arguments.query || question),
                    guild
                );
            case "get_member_profile":
                return DiscordToolService.getMemberProfile(
                    guild,
                    String(plan.arguments.nameOrId || "")
                );
            case "list_members":
                return DiscordToolService.listMembers(
                    guild,
                    plan.arguments.filters ? String(plan.arguments.filters) : undefined
                );
            case "get_guild_context":
                return DiscordToolService.getGuildContext(guild);
            case "finish":
            default:
                return {
                    tool: "finish",
                    summary: "Stopped search loop.",
                    data: null,
                };
        }
    }

    private static formatEvidence(toolRuns: DiscordToolResult[]): string {
        const lines: string[] = [];

        for (const run of toolRuns) {
            lines.push(`Tool: ${run.tool}`);
            lines.push(`Summary: ${run.summary}`);

            if (run.tool === "search_messages") {
                const chunks = run.data as Array<any>;
                chunks.slice(0, 5).forEach((chunk, index) => {
                    lines.push(
                        `${index + 1}. [${chunk.channelName}] ${chunk.authorName}: ${chunk.content} (${chunk.jumpLink})`
                    );
                });
            } else {
                lines.push(JSON.stringify(run.data, null, 2));
            }

            lines.push("");
        }

        return lines.join("\n").trim();
    }
}
