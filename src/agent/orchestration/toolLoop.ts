import type { Guild, User } from "discord.js";
import type { DebugSessionReporter } from "@/discord/debug/types";
import { INTERACTIVE_CRAWL_LIMIT } from "@/discord/live/DiscordChannelCrawlService";
import type {
    ChannelCrawlResult,
    DiscordToolResult,
    RetrievalControllerDecision,
    SearchPlan,
} from "@/shared/appTypes";
import { assessGrounding, collectMemberEvidence } from "@/agent/orchestration/grounding";
import { getDebugItemCount } from "@/agent/orchestration/evidenceFormatting";
import { describePlannedTool } from "@/agent/orchestration/debugPlanningDetails";
import { executeTool } from "@/agent/orchestration/toolExecution";
import { decideRetrievalAction } from "@/agent/orchestration/retrievalController";
import type { SearchContext } from "@/agent/orchestration/types";
import type { ConversationResolutionContext } from "@/shared/appTypes";

const MAX_TOOL_STEPS = 7;
const SEARCH_MESSAGE_LIMIT = 30;

export async function runDiscordToolLoop(options: {
    question: string;
    user: User;
    guild: Guild | null;
    currentChannelId?: string | null;
    debugSession?: DebugSessionReporter | null;
    initialToolRuns?: DiscordToolResult[];
    seededFromContext?: boolean;
    priorContext?: ConversationResolutionContext | null;
}): Promise<{
    toolRuns: DiscordToolResult[];
    finalDecision: RetrievalControllerDecision;
}> {
    const toolRuns: DiscordToolResult[] = [...(options.initialToolRuns ?? [])];
    const context: SearchContext = {
        crawledChannelIds: new Set<string>(
            toolRuns
                .filter((result) => result.tool === "crawl_channel_messages")
                .map((result) => (result.data as ChannelCrawlResult).channelId)
        ),
        seededFromContext: options.seededFromContext ?? false,
        initialToolRuns: [...toolRuns],
        latestControllerDecision: null,
    };

    for (let step = 0; step < MAX_TOOL_STEPS; step += 1) {
        await options.debugSession?.setPlanning(step + 1);
        const shouldForceFinal =
            toolRuns.length > 0 &&
            assessGrounding(options.question, toolRuns).summary.sufficient;
        const decision = await decideRetrievalAction({
            question: options.question,
            guild: options.guild,
            currentChannelId: options.currentChannelId,
            toolRuns,
            priorContext: options.priorContext,
            context,
            forcedFinal: shouldForceFinal,
        });
        context.latestControllerDecision = decision;
        await options.debugSession?.setRouting?.(decision);

        if (decision.nextAction === "answer" || decision.nextAction === "best_effort_answer") {
            return {
                toolRuns,
                finalDecision: decision,
            };
        }

        const next = toSearchPlan(decision, options.question, toolRuns);
        await options.debugSession?.setToolRunning(
            next.action,
            describePlannedTool(next, options.guild, options.currentChannelId)
        );
        const result = await executeTool(
            next,
            options.guild,
            options.question,
            options.currentChannelId,
            (toolName, summary) => options.debugSession?.setToolProgress?.(toolName, summary)
        );
        const debugSummary = result.cacheStatus
            ? `cache ${result.cacheStatus} · ${result.summary}`
            : result.summary;
        await options.debugSession?.setToolResult(
            result.tool,
            debugSummary,
            getDebugItemCount(result)
        );
        toolRuns.push(result);
        updateSearchContext(context, result);
    }

    const finalDecision = await decideRetrievalAction({
        question: options.question,
        guild: options.guild,
        currentChannelId: options.currentChannelId,
        toolRuns,
        priorContext: options.priorContext,
        context,
        forcedFinal: true,
    });
    context.latestControllerDecision = finalDecision;
    await options.debugSession?.setRouting?.(finalDecision);

    return {
        toolRuns,
        finalDecision,
    };
}

function toSearchPlan(
    decision: RetrievalControllerDecision,
    question: string,
    toolRuns: DiscordToolResult[]
): SearchPlan {
    if (
        decision.nextAction === "crawl_channel_messages" &&
        !hasPriorScopedSearch(toolRuns, decision, question)
    ) {
        return {
            action: "search_messages",
            arguments: {
                query: decision.searchQuery || decision.topicText || question,
                limit: SEARCH_MESSAGE_LIMIT,
                channelIds: decision.channelIds?.length
                    ? decision.channelIds.join(",")
                    : undefined,
                authorId: decision.authorId,
            },
            reason: `${decision.reason} Search indexed memory before crawling the channel.`,
        };
    }

    switch (decision.nextAction) {
        case "search_messages":
            return {
                action: "search_messages",
                arguments: {
                    query: decision.searchQuery || decision.topicText || question,
                    limit: SEARCH_MESSAGE_LIMIT,
                    channelIds: decision.channelIds?.length
                        ? decision.channelIds.join(",")
                        : undefined,
                    authorId: decision.authorId,
                },
                reason: decision.reason,
            };
        case "get_member_profile":
            return {
                action: "get_member_profile",
                arguments: {
                    nameOrId:
                        decision.authorId ||
                        decision.authorQuery ||
                        decision.targetText ||
                        question,
                },
                reason: decision.reason,
            };
        case "list_members":
            const memberEvidence = collectMemberEvidence(toolRuns);
            return {
                action: "list_members",
                arguments: {
                    filters: decision.authorQuery || decision.targetText || undefined,
                    limit: 100,
                    offset: memberEvidence?.members.length || undefined,
                    sort: "joined_at",
                },
                reason: decision.reason,
            };
        case "get_guild_context":
            return {
                action: "get_guild_context",
                arguments: {},
                reason: decision.reason,
            };
        case "list_relevant_channels":
            return {
                action: "list_relevant_channels",
                arguments: {
                    query:
                        decision.channelHintText ||
                        decision.topicText ||
                        decision.searchQuery ||
                        question,
                },
                reason: decision.reason,
            };
        case "crawl_channel_messages":
            return {
                action: "crawl_channel_messages",
                arguments: {
                    channelId: decision.channelIds?.[0] || "",
                    limit: INTERACTIVE_CRAWL_LIMIT,
                    queryHint:
                        decision.channelHintText ||
                        decision.topicText ||
                        decision.searchQuery ||
                        question,
                },
                reason: decision.reason,
            };
        default:
            return {
                action: "search_messages",
                arguments: {
                    query: decision.searchQuery || decision.topicText || question,
                    limit: SEARCH_MESSAGE_LIMIT,
                },
                reason: decision.reason,
            };
    }
}

function updateSearchContext(context: SearchContext, result: DiscordToolResult): void {
    if (result.tool !== "crawl_channel_messages" || !result.data) {
        return;
    }

    const crawl = result.data as ChannelCrawlResult;
    context.crawledChannelIds.add(crawl.channelId);
}

function hasPriorScopedSearch(
    toolRuns: DiscordToolResult[],
    decision: RetrievalControllerDecision,
    question: string
): boolean {
    const expectedQuery = (decision.searchQuery || decision.topicText || question).trim();
    const expectedAuthorId = decision.authorId || null;
    const expectedChannelIds = [...(decision.channelIds || [])].sort();

    return toolRuns.some((run) => {
        if (run.tool !== "search_messages") {
            return false;
        }

        const data = Array.isArray(run.data) ? run.data : [];
        if (!data.length) {
            return true;
        }

        const channelsInRun = [...new Set(
            data
                .map((item: any) => String(item.channelId || ""))
                .filter(Boolean)
        )].sort();
        const authorsInRun = new Set(
            data
                .map((item: any) => String(item.authorId || ""))
                .filter(Boolean)
        );

        const channelScopeMatches =
            !expectedChannelIds.length ||
            (channelsInRun.length > 0 &&
                channelsInRun.every((channelId) => expectedChannelIds.includes(channelId)));
        const authorMatches = !expectedAuthorId || authorsInRun.has(expectedAuthorId);
        const summaryHintsWeakSearch =
            /no relevant|found \d+ relevant message chunks/i.test(run.summary);
        const queryMentioned =
            expectedQuery.length === 0 ||
            run.summary.toLowerCase().includes(expectedQuery.toLowerCase()) ||
            data.some((item: any) =>
                String(item.content || "").toLowerCase().includes(expectedQuery.toLowerCase())
            );

        return channelScopeMatches && authorMatches && (queryMentioned || summaryHintsWeakSearch);
    });
}
