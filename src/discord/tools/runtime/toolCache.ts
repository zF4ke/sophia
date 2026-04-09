import { createHash } from "crypto";
import type { DiscordToolName } from "@/shared/discordTools";
import type { DiscordToolResult } from "@/shared/appTypes";
import { createQuestionFingerprint } from "@/agent/orchestration/questionFingerprint";
import { getRequestCacheContext } from "@/agent/orchestration/requestCacheContext";
import { DiscordMemoryService } from "@/memory/DiscordMemoryService";

const LIVE_TOOL_TTL_MS = 10 * 60_000;
const MEMORY_TOOL_TTL_MS = 20 * 60_000;
const MAX_CACHE_AGE_RESPONSES = 6;

type CacheableToolName = Exclude<DiscordToolName, "crawl_channel_messages">;

function normalizeCacheValue(value: unknown): unknown {
    if (typeof value === "string") {
        return createQuestionFingerprint(value);
    }

    if (Array.isArray(value)) {
        return value.map((item) => normalizeCacheValue(item));
    }

    if (value && typeof value === "object") {
        return Object.keys(value as Record<string, unknown>)
            .sort()
            .reduce<Record<string, unknown>>((acc, key) => {
                const normalized = normalizeCacheValue(
                    (value as Record<string, unknown>)[key]
                );
                if (
                    normalized !== undefined &&
                    normalized !== null &&
                    normalized !== ""
                ) {
                    acc[key] = normalized;
                }
                return acc;
            }, {});
    }

    return value ?? null;
}

function stableStringify(value: unknown): string {
    return JSON.stringify(normalizeCacheValue(value));
}

function getTtlMs(toolName: CacheableToolName): number {
    if (
        toolName === "get_guild_context" ||
        toolName === "get_member_profile" ||
        toolName === "list_members"
    ) {
        return LIVE_TOOL_TTL_MS;
    }

    return MEMORY_TOOL_TTL_MS;
}

export function buildToolCacheKey(
    toolName: CacheableToolName,
    guildId: string | null,
    args: Record<string, unknown>
): string {
    const digest = createHash("sha256")
        .update(
            stableStringify({
                toolName,
                guildId,
                args,
            })
        )
        .digest("hex");

    return `${toolName}:${digest}`;
}

export async function withCachedToolResult(
    toolName: CacheableToolName,
    guildId: string | null,
    args: Record<string, unknown>,
    executor: () => Promise<DiscordToolResult>
): Promise<DiscordToolResult> {
    const cacheKey = buildToolCacheKey(toolName, guildId, args);
    const requestContext = getRequestCacheContext();
    const cached = DiscordMemoryService.getCachedToolResult(
        cacheKey,
        requestContext.responseOrdinal,
        MAX_CACHE_AGE_RESPONSES
    );
    if (cached) {
        return {
            ...(cached.result as DiscordToolResult),
            cacheStatus: "hit",
        };
    }

    const result = await executor();
    const now = Date.now();
    DiscordMemoryService.saveCachedToolResult({
        cacheKey,
        toolName,
        guildId,
        argumentsJson: stableStringify(args),
        result: {
            ...result,
            cacheStatus: undefined,
        },
        createdTimestamp: now,
        expiryTimestamp: now + getTtlMs(toolName),
        createdResponseOrdinal: requestContext.responseOrdinal,
    });

    return {
        ...result,
        cacheStatus: "miss",
    };
}
