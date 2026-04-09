import type { Guild, User } from "discord.js";
import type { DebugSessionReporter } from "@/discord/debug/types";
import type {
    ChannelCrawlResult,
    DiscordToolResult,
    RouteDecision,
} from "@/shared/appTypes";
import { assessGrounding } from "@/agent/orchestration/grounding";
import { decideGroundingFromAssessment } from "@/agent/orchestration/evidenceJudge";
import { getDebugItemCount } from "@/agent/orchestration/evidenceFormatting";
import { describePlannedTool } from "@/agent/orchestration/debugPlanningDetails";
import { planNextTool } from "@/agent/orchestration/toolPlanning";
import { executeTool } from "@/agent/orchestration/toolExecution";
import type { GroundingDecision, SearchContext } from "@/agent/orchestration/types";

const MAX_TOOL_STEPS = 4;

export async function runDiscordToolLoop(options: {
    question: string;
    user: User;
    guild: Guild | null;
    routeDecision: RouteDecision;
    currentChannelId?: string | null;
    debugSession?: DebugSessionReporter | null;
    initialToolRuns?: DiscordToolResult[];
    seededFromContext?: boolean;
}): Promise<{ toolRuns: DiscordToolResult[]; groundingDecision: GroundingDecision | null }> {
    const toolRuns: DiscordToolResult[] = [...(options.initialToolRuns ?? [])];
    const context: SearchContext = {
        crawledChannelIds: new Set<string>(
            toolRuns
                .filter((result) => result.tool === "crawl_channel_messages")
                .map((result) => (result.data as ChannelCrawlResult).channelId)
        ),
        seededFromContext: options.seededFromContext ?? false,
        initialToolRuns: [...toolRuns],
    };

    for (let step = 0; step < MAX_TOOL_STEPS; step += 1) {
        await options.debugSession?.setPlanning(step + 1);
        const next = await planNextTool(
            options.question,
            toolRuns,
            options.guild,
            options.currentChannelId,
            context,
            options.routeDecision
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

        const groundingDecision = await shouldStopAfterTool(
            options.question,
            toolRuns,
            result,
            options.routeDecision
        );
        if (groundingDecision) {
            return {
                toolRuns,
                groundingDecision,
            };
        }
    }

    return {
        toolRuns,
        groundingDecision: null,
    };
}

async function shouldStopAfterTool(
    question: string,
    toolRuns: DiscordToolResult[],
    result: DiscordToolResult,
    routeDecision: RouteDecision
): Promise<GroundingDecision | null> {
    if (result.tool === "finish") {
        return {
            sufficient: false,
            mode: "heuristic",
            reason: "Search loop stopped explicitly.",
            missingInformation: null,
        };
    }

    if (result.tool === "crawl_channel_messages" || result.tool === "list_relevant_channels") {
        return null;
    }

    const grounding = assessGrounding(question, toolRuns);
    if (
        grounding.summary.messageEvidenceCount <= 0 &&
        grounding.summary.liveEvidenceCount <= 0
    ) {
        return null;
    }

    const decision = await decideGroundingFromAssessment({
        question,
        assessment: grounding,
        toolRuns,
        routeDecision,
    });
    return decision.sufficient ? decision : null;
}

function updateSearchContext(context: SearchContext, result: DiscordToolResult): void {
    if (result.tool !== "crawl_channel_messages") {
        return;
    }

    const crawl = result.data as ChannelCrawlResult;
    context.crawledChannelIds.add(crawl.channelId);
}
