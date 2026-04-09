import type { Guild, User } from "discord.js";
import type { DebugSessionReporter } from "@/discord/debug/types";
import type {
    ChannelCrawlResult,
    DiscordToolResult,
    RetrievedChunk,
} from "@/shared/appTypes";
import { assessGrounding } from "@/agent/orchestration/grounding";
import {
    getDebugItemCount,
    getToolEvidenceCount,
} from "@/agent/orchestration/evidenceFormatting";
import { isMemberDiscoveryQuestion } from "@/agent/orchestration/questionAnalysis";
import { describePlannedTool } from "@/agent/orchestration/debugPlanningDetails";
import { planNextTool } from "@/agent/orchestration/toolPlanning";
import { executeTool } from "@/agent/orchestration/toolExecution";
import type { SearchContext } from "@/agent/orchestration/types";

const MAX_TOOL_STEPS = 4;
const STRONG_SEARCH_SCORE = 0.45;

export async function runDiscordToolLoop(options: {
    question: string;
    user: User;
    guild: Guild | null;
    currentChannelId?: string | null;
    debugSession?: DebugSessionReporter | null;
}): Promise<DiscordToolResult[]> {
    const toolRuns: DiscordToolResult[] = [];
    const context: SearchContext = {
        crawledChannelIds: new Set<string>(),
    };

    for (let step = 0; step < MAX_TOOL_STEPS; step += 1) {
        await options.debugSession?.setPlanning(step + 1);
        const next = await planNextTool(
            options.question,
            toolRuns,
            options.guild,
            options.currentChannelId,
            context
        );
        if (next.action === "finish") {
            break;
        }

        await options.debugSession?.setToolRunning(
            next.action,
            describePlannedTool(next, options.guild, options.currentChannelId)
        );
        const result = await executeTool(
            next,
            options.guild,
            options.question,
            options.currentChannelId
        );
        await options.debugSession?.setToolResult(
            result.tool,
            result.summary,
            getDebugItemCount(result)
        );
        toolRuns.push(result);
        updateSearchContext(context, result);

        if (shouldStopAfterTool(options.question, toolRuns, result)) {
            break;
        }
    }

    return toolRuns;
}

function shouldStopAfterTool(
    question: string,
    toolRuns: DiscordToolResult[],
    result: DiscordToolResult
): boolean {
    if (result.tool === "finish") {
        return true;
    }

    if (result.tool === "search_messages") {
        const chunks = result.data as RetrievedChunk[];
        return Boolean(chunks.length && chunks[0].totalScore >= STRONG_SEARCH_SCORE);
    }

    if (result.tool === "get_member_profile") {
        return getToolEvidenceCount(result) > 0;
    }

    if (result.tool === "get_guild_context") {
        return getToolEvidenceCount(result) > 0 && !isMemberDiscoveryQuestion(question);
    }

    if (result.tool === "list_members") {
        const grounding = assessGrounding(question, toolRuns);
        return grounding.summary.sufficient;
    }

    if (result.tool === "crawl_channel_messages") {
        return false;
    }

    return false;
}

function updateSearchContext(context: SearchContext, result: DiscordToolResult): void {
    if (result.tool !== "crawl_channel_messages") {
        return;
    }

    const crawl = result.data as ChannelCrawlResult;
    context.crawledChannelIds.add(crawl.channelId);
}
