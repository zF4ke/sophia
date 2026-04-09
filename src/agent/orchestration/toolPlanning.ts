import type { Guild } from "discord.js";
import { ModelGateway } from "@/ai/ModelGateway";
import { PromptRegistry } from "@/agent/prompts/PromptRegistry";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import type {
    ChannelCandidate,
    DiscordToolResult,
    RetrievedChunk,
    SearchPlan,
} from "@/shared/appTypes";
import {
    collectMemberEvidence,
    memberEvidenceNeedsMore,
} from "@/agent/orchestration/grounding";
import {
    extractLikelyPersonName,
    extractMentionedChannelIds,
    extractRequestedOrdinal,
    isGuildContextQuestion,
    isMemberDiscoveryQuestion,
    isPersonCentricQuestion,
} from "@/agent/orchestration/questionAnalysis";
import type { SearchContext } from "@/agent/orchestration/types";

const MIN_SEARCH_SCORE = 0.15;

export async function planNextTool(
    question: string,
    toolRuns: DiscordToolResult[],
    guild: Guild | null,
    currentChannelId: string | null | undefined,
    context: SearchContext
): Promise<SearchPlan> {
    const heuristic = planNextToolHeuristically(
        question,
        toolRuns,
        guild,
        currentChannelId,
        context
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
    context: SearchContext
): SearchPlan | null {
    if (!isMemberDiscoveryQuestion(question)) {
        return planFallbackChannelCrawl(question, toolRuns, guild, currentChannelId, context);
    }

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

function planFallbackChannelCrawl(
    question: string,
    toolRuns: DiscordToolResult[],
    guild: Guild | null,
    currentChannelId: string | null | undefined,
    context: SearchContext
): SearchPlan | null {
    const latestGuildContextRun = [...toolRuns]
        .reverse()
        .find((run) => run.tool === "get_guild_context");
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
    const mentionedChannelIds = extractMentionedChannelIds(question);

    if (isGuildContextQuestion(question)) {
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

    if (!latestSearchRun && isPersonCentricQuestion(question) && !latestProfileRun) {
        const targetName = extractLikelyPersonName(question);
        if (targetName) {
            return {
                action: "get_member_profile",
                arguments: {
                    nameOrId: targetName,
                },
                reason: "Resolve the likely person before searching messages.",
            };
        }
    }

    if (!latestSearchRun) {
        return {
            action: "search_messages",
            arguments: {
                query: question,
                limit: 8,
                authorId: profile?.id,
                channelIds: mentionedChannelIds.length
                    ? mentionedChannelIds.join(",")
                    : undefined,
            },
            reason: mentionedChannelIds.length
                ? "Try local memory search in the mentioned channel first."
                : "Try local memory search first.",
        };
    }

    const topChunk = (latestSearchRun.data as RetrievedChunk[])[0];
    if (topChunk && topChunk.totalScore >= MIN_SEARCH_SCORE) {
        return null;
    }

    const nextMentionedChannelId = mentionedChannelIds.find(
        (channelId) => !context.crawledChannelIds.has(channelId)
    );
    if (nextMentionedChannelId) {
        return {
            action: "crawl_channel_messages",
            arguments: {
                channelId: nextMentionedChannelId,
                limit: 1000,
                queryHint: question,
            },
            reason: "Fetch and ingest messages from the mentioned channel.",
        };
    }

    if (latestCrawlRun && mentionedChannelIds.length) {
        return {
            action: "search_messages",
            arguments: {
                query: question,
                limit: 8,
                authorId: profile?.id,
                channelIds: mentionedChannelIds.join(","),
            },
            reason: "Retry local search in the mentioned channel after crawling it.",
        };
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

    const nextChannel = (latestChannelListRun.data as ChannelCandidate[]).find(
        (candidate) => !context.crawledChannelIds.has(candidate.channelId)
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
                authorId: profile?.id,
                channelIds: mentionedChannelIds.length
                    ? mentionedChannelIds.join(",")
                    : undefined,
            },
            reason: mentionedChannelIds.length
                ? "Retry local search in the mentioned channel after crawling it."
                : "Retry local search after crawling live channels.",
        };
    }

    return {
        action: "finish",
        arguments: {},
        reason: "No candidate channels remain to crawl.",
    };
}
