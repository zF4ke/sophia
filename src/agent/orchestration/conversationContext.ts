import { getRequestCacheContext } from "@/agent/orchestration/requestCacheContext";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type {
    ConversationResolutionContext,
    DiscordToolResult,
    RouteDecision,
} from "@/shared/appTypes";

const CONVERSATION_CONTEXT_TTL_MS = 20 * 60_000;
const MAX_CONTEXT_AGE_RESPONSES = 6;

export function findConversationResolutionContext(options: {
    guildId: string | null;
    currentChannelId: string | null;
}): ConversationResolutionContext | null {
    const requestContext = getRequestCacheContext();
    return DiscordMemoryService.getConversationResolutionContext({
        guildId: options.guildId,
        currentChannelId: options.currentChannelId,
        currentResponseOrdinal: requestContext.responseOrdinal,
        maxResponsesAgo: MAX_CONTEXT_AGE_RESPONSES,
    });
}

export function saveConversationResolutionContext(options: {
    guildId: string | null;
    currentChannelId: string | null;
    routeDecision: RouteDecision;
    toolRuns: DiscordToolResult[];
}): void {
    const requestContext = getRequestCacheContext();
    const profileRun = [...options.toolRuns]
        .reverse()
        .find((run) => run.tool === "get_member_profile");
    const profile = profileRun
        ? DiscordToolService.getMemberProfileResult(profileRun)
        : null;

    const shouldPersist =
        Boolean(profile) ||
        Boolean(options.routeDecision.channelIds?.length) ||
        Boolean(options.routeDecision.channelHintText) ||
        Boolean(options.routeDecision.topicText);
    if (!shouldPersist) {
        return;
    }

    const now = Date.now();
    DiscordMemoryService.saveConversationResolutionContext({
        guildId: options.guildId,
        channelId: options.currentChannelId,
        routeIntent: options.routeDecision.intent,
        targetText: options.routeDecision.targetText,
        authorId: options.routeDecision.authorId ?? profile?.id ?? null,
        authorQuery:
            options.routeDecision.authorQuery ??
            profile?.username ??
            options.routeDecision.targetText ??
            null,
        channelIds: options.routeDecision.channelIds ?? [],
        topicText: options.routeDecision.topicText ?? null,
        channelHintText: options.routeDecision.channelHintText ?? null,
        resolvedPerson: profile
            ? {
                  id: profile.id,
                  username: profile.username,
                  displayName: profile.displayName,
                  globalName: profile.globalName,
                  nickname: profile.nickname,
                  roles: profile.roles,
              }
            : options.routeDecision.resolvedPerson ?? null,
        createdTimestamp: now,
        expiryTimestamp: now + CONVERSATION_CONTEXT_TTL_MS,
        createdResponseOrdinal: requestContext.responseOrdinal,
    });
}
