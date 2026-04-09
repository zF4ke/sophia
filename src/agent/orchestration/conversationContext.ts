import { getRequestCacheContext } from "@/agent/orchestration/requestCacheContext";
import { DiscordToolService } from "@/discord/tools/DiscordToolService";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";
import type {
    ConversationResolutionContext,
    DiscordToolResult,
    RetrievalControllerDecision,
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
    controllerDecision: RetrievalControllerDecision;
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
        Boolean(options.controllerDecision.channelIds?.length) ||
        Boolean(options.controllerDecision.channelHintText) ||
        Boolean(options.controllerDecision.topicText);
    if (!shouldPersist) {
        return;
    }

    const now = Date.now();
    DiscordMemoryService.saveConversationResolutionContext({
        guildId: options.guildId,
        channelId: options.currentChannelId,
        routeIntent: options.controllerDecision.routeIntent,
        targetText: options.controllerDecision.targetText,
        authorId: options.controllerDecision.authorId ?? profile?.id ?? null,
        authorQuery:
            options.controllerDecision.authorQuery ??
            profile?.username ??
            options.controllerDecision.targetText ??
            null,
        channelIds: options.controllerDecision.channelIds ?? [],
        topicText: options.controllerDecision.topicText ?? null,
        channelHintText: options.controllerDecision.channelHintText ?? null,
        resolvedPerson: profile
            ? {
                  id: profile.id,
                  username: profile.username,
                  displayName: profile.displayName,
                  globalName: profile.globalName,
                  nickname: profile.nickname,
                  roles: profile.roles,
              }
            : options.controllerDecision.resolvedPerson ?? null,
        createdTimestamp: now,
        expiryTimestamp: now + CONVERSATION_CONTEXT_TTL_MS,
        createdResponseOrdinal: requestContext.responseOrdinal,
    });
}
