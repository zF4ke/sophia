import type { Guild } from "discord.js";
import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import type {
    ChannelCandidate,
    DiscordToolResult,
    RetrievedChunk,
    RouteDecision,
    SearchPlan,
} from "@/shared/appTypes";
import {
    collectMemberEvidence,
    memberEvidenceNeedsMore,
} from "@/agent/orchestration/grounding";
import {
    extractRequestedOrdinal,
    isIdentityQuestion,
    isPersonMessageQuestion,
} from "@/agent/orchestration/questionAnalysis";
import type { SearchContext } from "@/agent/orchestration/types";

const MIN_SEARCH_SCORE = 0.15;

export async function planNextTool(
    question: string,
    toolRuns: DiscordToolResult[],
    guild: Guild | null,
    currentChannelId: string | null | undefined,
    context: SearchContext,
    routeDecision: RouteDecision
): Promise<SearchPlan> {
    const heuristic = planNextToolHeuristically(
        question,
        toolRuns,
        guild,
        currentChannelId,
        context,
        routeDecision
    );
    if (heuristic) {
        return heuristic;
    }

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
    const routeSummary = JSON.stringify(routeDecision, null, 2);

    return ModelGateway.generateJson<SearchPlan>(
        [
            { role: "system", content: "Return strict JSON only." },
            {
                role: "user",
                content: PromptRegistry.render("tasks/plan_discord_search", {
                    question,
                    tool_results: toolResults,
                    route_decision: routeSummary,
                }),
            },
        ],
        fallback,
        {
            traceContext: {
                traceLabel: "discord_tool_planning",
                questionPreview: question,
            },
        }
    );
}

function planNextToolHeuristically(
    question: string,
    toolRuns: DiscordToolResult[],
    guild: Guild | null,
    currentChannelId: string | null | undefined,
    context: SearchContext,
    routeDecision: RouteDecision
): SearchPlan | null {
    if (routeDecision.intent === "member_lookup") {
        return planMemberLookup(question, toolRuns);
    }

    if (routeDecision.intent === "server_context") {
        return planServerContext(toolRuns);
    }

    if (routeDecision.intent === "channel_target") {
        return planChannelTarget(question, toolRuns, context, routeDecision);
    }

    if (routeDecision.intent === "person_target") {
        return planPersonTarget(question, toolRuns, context, routeDecision);
    }

    return planBroadSearch(question, toolRuns, guild, currentChannelId, context);
}

function planMemberLookup(
    question: string,
    toolRuns: DiscordToolResult[]
): SearchPlan | null {
    const latestGuildContext = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "get_guild_context");
    const memberEvidence = collectMemberEvidence(toolRuns);
    const ordinal = extractRequestedOrdinal(question);

    if (!latestGuildContext) {
        return {
            action: "get_guild_context",
            arguments: {},
            reason: "Fetch server totals before listing members.",
        };
    }

    if (!memberEvidence) {
        return {
            action: "list_members",
            arguments: {
                limit: 100,
                offset: 0,
                sort: "joined_at",
            },
            reason: "Load the first page of members in join order.",
        };
    }

    if (memberEvidenceNeedsMore(question, memberEvidence, ordinal)) {
        return {
            action: "list_members",
            arguments: {
                filters: memberEvidence.filters || undefined,
                limit: memberEvidence.members.length < 100 ? 100 : memberEvidence.members.length,
                offset: memberEvidence.members.length,
                sort: memberEvidence.sort,
            },
            reason: "Fetch more members to cover the requested rank or full list.",
        };
    }

    return {
        action: "finish",
        arguments: {},
        reason: "Available member evidence is sufficient.",
    };
}

function planServerContext(toolRuns: DiscordToolResult[]): SearchPlan {
    const latestGuildContextRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "get_guild_context");
    if (!latestGuildContextRun) {
        return {
            action: "get_guild_context",
            arguments: {},
            reason: "Use live server metadata for current-server questions.",
        };
    }

    return {
        action: "finish",
        arguments: {},
        reason: "Live server metadata is already available.",
    };
}

function planChannelTarget(
    question: string,
    toolRuns: DiscordToolResult[],
    context: SearchContext,
    routeDecision: RouteDecision
): SearchPlan | null {
    const latestSearchRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "search_messages");
    const latestChannelListRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "list_relevant_channels");
    const latestCrawlRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "crawl_channel_messages");
    const targetChannelIds = routeDecision.channelIds ?? [];

    if (!latestSearchRun) {
        return {
            action: "search_messages",
            arguments: {
                query: question,
                limit: 8,
                channelIds: targetChannelIds.length ? targetChannelIds.join(",") : undefined,
            },
            reason: targetChannelIds.length
                ? "Try local memory search in the routed target channel first."
                : "Try local memory search for the routed channel target first.",
        };
    }

    const topChunk = (latestSearchRun.data as RetrievedChunk[])[0];
    if (topChunk && topChunk.totalScore >= MIN_SEARCH_SCORE) {
        return null;
    }

    const nextTargetChannelId = targetChannelIds.find(
        (channelId) => !context.crawledChannelIds.has(channelId)
    );
    if (nextTargetChannelId) {
        return {
            action: "crawl_channel_messages",
            arguments: {
                channelId: nextTargetChannelId,
                limit: 1000,
                queryHint: question,
            },
            reason: "Fetch and ingest messages from the routed target channel.",
        };
    }

    if (latestCrawlRun && targetChannelIds.length) {
        return {
            action: "search_messages",
            arguments: {
                query: question,
                limit: 8,
                channelIds: targetChannelIds.join(","),
            },
            reason: "Retry local search in the routed channel after crawling it.",
        };
    }

    if (!latestChannelListRun) {
        return {
            action: "list_relevant_channels",
            arguments: {
                query: question,
            },
            reason: routeDecision.targetText
                ? "Find likely channels matching the routed target."
                : "Find likely channels, including channels not yet indexed.",
        };
    }

    const nextChannel = pickChannelCandidate(
        latestChannelListRun.data as ChannelCandidate[],
        context,
        routeDecision.targetText
    );
    if (nextChannel) {
        return {
            action: "crawl_channel_messages",
            arguments: {
                channelId: nextChannel.channelId,
                limit: 1000,
                queryHint: question,
            },
            reason: "Fetch and ingest messages from an uncrawled candidate channel.",
        };
    }

    if (latestCrawlRun) {
        return {
            action: "search_messages",
            arguments: {
                query: question,
                limit: 8,
                channelIds: targetChannelIds.length ? targetChannelIds.join(",") : undefined,
            },
            reason: targetChannelIds.length
                ? "Retry local search in the routed channel after crawling it."
                : "Retry local search after crawling likely live channels.",
        };
    }

    return {
        action: "finish",
        arguments: {},
        reason: "No channel candidates remain to crawl for the routed target.",
    };
}

function planPersonTarget(
    question: string,
    toolRuns: DiscordToolResult[],
    context: SearchContext,
    routeDecision: RouteDecision
): SearchPlan | null {
    const latestSearchRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "search_messages");
    const latestChannelListRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "list_relevant_channels");
    const latestCrawlRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "crawl_channel_messages");
    const latestProfileRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "get_member_profile");
    const profile = latestProfileRun
        ? DiscordToolService.getMemberProfileResult(latestProfileRun)
        : null;
    const authorId = routeDecision.authorId ?? profile?.id;
    const authorTarget = routeDecision.authorQuery ?? routeDecision.targetText ?? undefined;
    const searchQuery = routeDecision.topicText ?? question;
    const channelHint = routeDecision.channelHintText ?? routeDecision.topicText ?? undefined;
    const targetChannelIds = routeDecision.channelIds ?? [];

    if (profile && isIdentityQuestion(question) && !isPersonMessageQuestion(question)) {
        return {
            action: "finish",
            arguments: {},
            reason: "Resolved member profile already answers this identity question.",
        };
    }

    if (!authorId && !latestProfileRun && authorTarget) {
        return {
            action: "get_member_profile",
            arguments: {
                nameOrId: authorTarget,
            },
            reason: "Resolve the routed person target before searching messages.",
        };
    }

    if (!latestSearchRun) {
        return {
            action: "search_messages",
            arguments: {
                query: searchQuery,
                limit: 8,
                authorId,
                channelIds: targetChannelIds.length
                    ? targetChannelIds.join(",")
                    : undefined,
            },
            reason: targetChannelIds.length
                ? "Try local memory search for the routed person target in the hinted channel first."
                : "Try local memory search for the routed person target first.",
        };
    }

    const topChunk = (latestSearchRun.data as RetrievedChunk[])[0];
    if (topChunk && topChunk.totalScore >= MIN_SEARCH_SCORE) {
        return null;
    }

    const nextTargetChannelId = targetChannelIds.find(
        (channelId) => !context.crawledChannelIds.has(channelId)
    );
    if (nextTargetChannelId) {
        return {
            action: "crawl_channel_messages",
            arguments: {
                channelId: nextTargetChannelId,
                limit: 1000,
                queryHint: searchQuery,
            },
            reason: "Fetch and ingest messages from the hinted channel for the routed person target.",
        };
    }

    if (latestCrawlRun && targetChannelIds.length) {
        return {
            action: "search_messages",
            arguments: {
                query: searchQuery,
                limit: 8,
                authorId,
                channelIds: targetChannelIds.join(","),
            },
            reason: "Retry scoped author search after crawling the hinted channel.",
        };
    }

    if (!latestChannelListRun) {
        return {
            action: "list_relevant_channels",
            arguments: {
                query: channelHint || searchQuery || authorTarget || question,
            },
            reason: "Find likely channels for the routed person target and topic.",
        };
    }

    const nextChannel = pickChannelCandidate(
        latestChannelListRun.data as ChannelCandidate[],
        context,
        channelHint
    );
    if (nextChannel) {
        return {
            action: "crawl_channel_messages",
            arguments: {
                channelId: nextChannel.channelId,
                limit: 1000,
                queryHint: searchQuery,
            },
            reason: "Fetch and ingest messages from a likely channel for the routed person target.",
        };
    }

    if (latestCrawlRun) {
        return {
            action: "search_messages",
            arguments: {
                query: searchQuery,
                limit: 8,
                authorId,
            },
            reason: "Retry author-scoped search after crawling likely channels.",
        };
    }

    return {
        action: "finish",
        arguments: {},
        reason: "No channels remain to search for the routed person target.",
    };
}

function planBroadSearch(
    question: string,
    toolRuns: DiscordToolResult[],
    guild: Guild | null,
    currentChannelId: string | null | undefined,
    context: SearchContext
): SearchPlan | null {
    const latestSearchRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "search_messages");
    const latestChannelListRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "list_relevant_channels");
    const latestCrawlRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "crawl_channel_messages");

    if (!latestSearchRun) {
        return {
            action: "search_messages",
            arguments: {
                query: question,
                limit: 8,
            },
            reason: "Try local memory search first.",
        };
    }

    const topChunk = (latestSearchRun.data as RetrievedChunk[])[0];
    if (topChunk && topChunk.totalScore >= MIN_SEARCH_SCORE) {
        return null;
    }

    if (!latestChannelListRun) {
        return {
            action: "list_relevant_channels",
            arguments: {
                query: question,
            },
            reason: "Find likely channels, including channels not yet indexed.",
        };
    }

    const nextChannel = pickChannelCandidate(
        latestChannelListRun.data as ChannelCandidate[],
        context
    );
    if (nextChannel) {
        return {
            action: "crawl_channel_messages",
            arguments: {
                channelId: nextChannel.channelId,
                limit: 1000,
                queryHint: question,
            },
            reason: "Fetch and ingest messages from an uncrawled candidate channel.",
        };
    }

    if (latestCrawlRun) {
        return {
            action: "search_messages",
            arguments: {
                query: question,
                limit: 8,
            },
            reason: "Retry local search after crawling live channels.",
        };
    }

    return {
        action: "finish",
        arguments: {},
        reason: "No candidate channels remain to crawl.",
    };
}

function pickChannelCandidate(
    candidates: ChannelCandidate[],
    context: SearchContext,
    targetText?: string | null
): ChannelCandidate | undefined {
    const uncrawled = candidates.filter(
        (candidate) => !context.crawledChannelIds.has(candidate.channelId)
    );

    if (!targetText) {
        return uncrawled[0];
    }

    const normalizedTarget = targetText.toLowerCase();
    const matched = uncrawled.find((candidate) =>
        candidate.channelName.toLowerCase().includes(normalizedTarget)
    );
    return matched ?? uncrawled[0];
}
