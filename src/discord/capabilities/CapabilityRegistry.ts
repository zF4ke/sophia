import { z } from "zod";
import type { Guild } from "discord.js";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import { DiscordLiveService } from "@/discord/live/DiscordLiveService";
import { UnifiedMessageRetrieval } from "@/discord/retrieval/UnifiedMessageRetrieval";
import { DISCORD_TOOL_EVIDENCE_ROLES, type DiscordToolName } from "@/shared/discordTools";
import type { DiscordToolResult, RetrievalMode } from "@/shared/appTypes";
import type { CapabilityManifest, ToolArguments } from "@/runtime/contracts";

type CapabilityContext = {
    guild: Guild | null;
    question: string;
    currentChannelId?: string | null;
    onProgress?: (toolName: string, summary: string) => Promise<void> | void;
};

type RuntimeCapability = CapabilityManifest & {
    run(
        context: CapabilityContext,
        args: ToolArguments
    ): Promise<DiscordToolResult>;
};

const chunkResultSchema = z.object({
    messageId: z.string(),
    channelId: z.string(),
    channelName: z.string(),
    guildId: z.string().nullable(),
    authorId: z.string(),
    authorName: z.string(),
    content: z.string(),
    createdTimestamp: z.number(),
    jumpLink: z.string(),
    lexicalScore: z.number(),
    semanticScore: z.number(),
    recencyScore: z.number(),
    totalScore: z.number(),
});

const retrievalModeSchema = z.enum(["history", "semantic", "mixed"]);
const semanticCursorSchema = z.object({
    lastScore: z.number(),
    lastCreatedTimestamp: z.number(),
    lastMessageId: z.string(),
});

const capabilities: RuntimeCapability[] = [
    {
        id: "retrieve_messages",
        kind: "tool",
        description:
            "Read scoped Discord channel history first, add semantic matches from the same scope, and continue with live history fetches when needed.",
        inputSchema: z.object({
            query: z.string(),
            limit: z.number().int().positive().optional(),
            channelIds: z.array(z.string()).optional(),
            authorId: z.string().optional(),
            beforeTimestamp: z.number().optional(),
            afterTimestamp: z.number().optional(),
            mode: retrievalModeSchema.optional(),
            cursor: z
                .object({
                    history: z.record(z.string(), z.string().nullable()).optional(),
                    semantic: semanticCursorSchema.nullable().optional(),
                })
                .optional(),
            excludedMessageIds: z.array(z.string()).optional(),
        }),
        outputSchema: z.object({
            query: z.string(),
            mode: retrievalModeSchema,
            cacheHit: z.boolean(),
            liveEscalated: z.boolean(),
            searchedChannelIds: z.array(z.string()),
            fetchedChannelIds: z.array(z.string()),
            cacheEnriched: z.boolean(),
            evidenceSufficient: z.boolean(),
            strongResultCount: z.number(),
            weakResultCount: z.number(),
            historyMessageCount: z.number(),
            semanticMatchCount: z.number(),
            sourceOrigin: z.enum(["none", "cache", "live_refresh", "cache_after_refresh"]),
            targetAuthorId: z.string().nullable(),
            targetChannelIds: z.array(z.string()),
            historyMessages: z.array(chunkResultSchema),
            semanticMatches: z.array(chunkResultSchema),
            combinedResults: z.array(chunkResultSchema),
            continuation: z.object({
                history: z.object({
                    perChannelOldestMessageId: z.record(z.string(), z.string().nullable()),
                    continuationAvailable: z.boolean(),
                }),
                semantic: z.object({
                    cursor: semanticCursorSchema.nullable(),
                    continuationAvailable: z.boolean(),
                }),
                perChannelOldestMessageId: z.record(z.string(), z.string().nullable()),
                continuationAvailable: z.boolean(),
            }),
            exhaustion: z.object({
                historyExhaustedChannelIds: z.array(z.string()),
                historyExhausted: z.boolean(),
                semanticExhausted: z.boolean(),
                exhaustedChannelIds: z.array(z.string()),
                exhausted: z.boolean(),
            }),
            accumulatedWindow: z.object({
                beforeTimestamp: z.number().nullable(),
                afterTimestamp: z.number().nullable(),
            }),
            accumulatedUniqueCount: z.number(),
            beforeTimestamp: z.number().nullable(),
            afterTimestamp: z.number().nullable(),
            excludedMessageIds: z.array(z.string()),
        }),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.retrieve_messages,
        preconditions: ["guild context should exist for live escalation"],
        postconditions: ["returns message evidence from the local cache after any needed live fetch"],
        async run(context, args) {
            const query = typeof args.query === "string" ? args.query : context.question;
            const channelIds = Array.isArray(args.channelIds)
                ? args.channelIds.filter((value): value is string => typeof value === "string" && value.trim().length > 0)
                : undefined;
            const authorId =
                typeof args.authorId === "string" && args.authorId.trim()
                    ? args.authorId.trim()
                    : undefined;
            const mode =
                typeof args.mode === "string" &&
                ["history", "semantic", "mixed"].includes(args.mode)
                    ? (args.mode as RetrievalMode)
                    : undefined;
            const cursor =
                args.cursor && typeof args.cursor === "object" && !Array.isArray(args.cursor)
                    ? {
                          history:
                              args.cursor.history &&
                              typeof args.cursor.history === "object" &&
                              !Array.isArray(args.cursor.history)
                                  ? Object.fromEntries(
                                        Object.entries(args.cursor.history)
                                            .filter(([, value]) => value == null || typeof value === "string")
                                            .map(([key, value]) => [key, value == null ? null : String(value)])
                                    )
                                  : undefined,
                          semantic:
                              args.cursor.semantic &&
                              typeof args.cursor.semantic === "object" &&
                              !Array.isArray(args.cursor.semantic) &&
                              typeof args.cursor.semantic.lastScore === "number" &&
                              typeof args.cursor.semantic.lastCreatedTimestamp === "number" &&
                              typeof args.cursor.semantic.lastMessageId === "string"
                                  ? {
                                        lastScore: args.cursor.semantic.lastScore,
                                        lastCreatedTimestamp: args.cursor.semantic.lastCreatedTimestamp,
                                        lastMessageId: args.cursor.semantic.lastMessageId,
                                    }
                                  : undefined,
                      }
                    : undefined;
            const excludedMessageIds = Array.isArray(args.excludedMessageIds)
                ? args.excludedMessageIds.filter(
                      (value): value is string => typeof value === "string" && value.trim().length > 0
                  )
                : undefined;

            const result = await UnifiedMessageRetrieval.retrieve({
                guild: context.guild,
                question: query,
                currentChannelId: context.currentChannelId,
                channelIds,
                authorId,
                mode,
                beforeTimestamp:
                    typeof args.beforeTimestamp === "number" ? args.beforeTimestamp : undefined,
                afterTimestamp:
                    typeof args.afterTimestamp === "number" ? args.afterTimestamp : undefined,
                cursor,
                excludedMessageIds,
                limit: Number(args.limit || 8),
                onProgress: context.onProgress,
            });

            const qualityLabel =
                result.historyMessageCount >= 2
                    ? "ordered history evidence"
                    : result.historyMessageCount >= 1
                      ? "partial history evidence"
                      : result.semanticMatchCount
                        ? "semantic evidence"
                        : "no message evidence";
                        const diagnosticsSuffix = result.retrievalDiagnostics?.scopedEmptyRetryAttempted
                                ? ` (scoped retry: ${result.retrievalDiagnostics.retryStrategy}, recovered=${
                                            result.retrievalDiagnostics.scopedEmptyRetryRecovered ? "yes" : "no"
                                    })`
                                : "";

            return {
                tool: "retrieve_messages",
                summary: result.combinedResults.length
                                        ? `${qualityLabel}; ${result.historyMessageCount} history and ${result.semanticMatchCount} semantic result(s) ${result.liveEscalated ? "after refreshing Discord history" : "from cached Discord history"}.${diagnosticsSuffix}`
                    : result.liveEscalated
                                            ? `No relevant messages found even after refreshing Discord history.${diagnosticsSuffix}`
                                            : `No relevant cached messages found yet.${diagnosticsSuffix}`,
                data: result,
            };
        },
    },
    {
        id: "resolve_member_identity",
        kind: "tool",
        description:
            "Resolve a member or bot in the current guild using exact ids, live guild fetches, and same-guild historical message authors.",
        inputSchema: z.object({
            query: z.string(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.resolve_member_identity,
        preconditions: ["guild context should exist"],
        postconditions: ["returns the best resolved member identity for the current guild"],
        async run(context, args) {
            const identity = await DiscordLiveService.resolveMemberIdentity(
                context.guild,
                String(args.query || context.question)
            );
            return {
                tool: "resolve_member_identity",
                summary: identity
                    ? identity.isCurrentGuildMember
                        ? `${identity.displayName} (@${identity.username}) resolved from the current guild.`
                        : `${identity.displayName} was resolved from remembered guild history, not current live membership.`
                    : "Member identity not resolved.",
                data: identity,
            };
        },
    },
    {
        id: "list_guild_structure",
        kind: "tool",
        description:
            "List readable live channels and categories in the current guild plus cached-only remembered entries.",
        inputSchema: z.object({
            targetText: z.string().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
        preconditions: ["guild context should exist"],
        postconditions: ["returns current-guild structure and cached-only remembered entries"],
        async run(context, args) {
            const entries = await DiscordGuildDiscoveryService.listGuildStructure(context.guild);
            const targetText =
                typeof args.targetText === "string" && args.targetText.trim()
                    ? args.targetText.trim()
                    : null;
            const focused = targetText
                ? await DiscordGuildDiscoveryService.resolveChannelTargets(
                      context.guild,
                      targetText,
                      context.currentChannelId
                  )
                : null;
            return {
                tool: "list_guild_structure",
                summary: entries.length
                    ? `Resolved ${entries.length} guild structure entries from live Discord and local memory.`
                    : "Guild structure unavailable.",
                data: {
                    query: targetText,
                    entries,
                    focusedEntries: focused?.entries || [],
                    focusedResolvedIds: focused?.resolvedIds || [],
                },
            };
        },
    },
    {
        id: "resolve_channel_targets",
        kind: "tool",
        description:
            "Resolve channel or category references in the current guild, including exact ids and category expansion.",
        inputSchema: z.object({
            targetText: z.string(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.resolve_channel_targets,
        preconditions: ["guild context should exist"],
        postconditions: ["returns resolved message-channel targets for the current guild"],
        async run(context, args) {
            const resolved = await DiscordGuildDiscoveryService.resolveChannelTargets(
                context.guild,
                String(args.targetText || context.question),
                context.currentChannelId
            );
            return {
                tool: "resolve_channel_targets",
                summary: resolved.entries.length
                    ? `Resolved ${resolved.entries.length} guild structure target(s) with ${resolved.resolvedIds.length} message channel id(s).`
                    : "No matching channel or category target was resolved.",
                data: resolved,
            };
        },
    },
    {
        id: "get_member_profile",
        kind: "tool",
        description: "Fetch a live guild member profile.",
        inputSchema: z.object({
            nameOrId: z.string(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "medium",
        evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.get_member_profile,
        preconditions: ["guild context should exist"],
        postconditions: ["returns live member identity/profile evidence"],
        async run(context, args) {
            const profile = await DiscordLiveService.getMemberProfile(
                context.guild,
                String(args.nameOrId || context.question)
            );
            return {
                tool: "get_member_profile",
                summary: profile
                    ? `${profile.displayName} (@${profile.username}) with ${profile.roles.length} visible roles.`
                    : "Member not found.",
                data: profile,
            };
        },
    },
    {
        id: "list_members",
        kind: "tool",
        description: "List live guild members with optional filtering.",
        inputSchema: z.object({
            filters: z.string().optional(),
            limit: z.number().int().positive().optional(),
            offset: z.number().int().nonnegative().optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_members,
        preconditions: ["guild context should exist"],
        postconditions: ["returns live member list data"],
        async run(context, args) {
            const members = await DiscordLiveService.listMembers(context.guild, {
                filters: typeof args.filters === "string" ? args.filters : undefined,
                limit: typeof args.limit === "number" ? args.limit : undefined,
                offset: typeof args.offset === "number" ? args.offset : undefined,
                sort: "joined_at",
            });
            return {
                tool: "list_members",
                summary: members.returnedCount
                    ? members.hasMore
                        ? `Showing ${members.returnedCount} of ${members.totalCount} members in join order.`
                        : `${members.returnedCount} members listed in join order.`
                    : "No matching members found.",
                data: members,
            };
        },
    },
    {
        id: "get_guild_context",
        kind: "tool",
        description: "Fetch live guild metadata.",
        inputSchema: z.object({}),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.get_guild_context,
        preconditions: ["guild context should exist"],
        postconditions: ["returns live guild context data"],
        async run(context) {
            const guildContext = await DiscordLiveService.getGuildContext(context.guild);
            return {
                tool: "get_guild_context",
                summary: guildContext
                    ? `${guildContext.name}: ${guildContext.memberCount} members and ${guildContext.channelCount} channels.`
                    : "Guild context unavailable.",
                data: guildContext,
            };
        },
    },
];

export class CapabilityRegistry {
    public static list(): CapabilityManifest[] {
        return capabilities.map(({ run, ...manifest }) => manifest);
    }

    public static get(id: DiscordToolName): RuntimeCapability {
        const capability = capabilities.find((item) => item.id === id);
        if (!capability) {
            throw new Error(`Unknown capability "${id}".`);
        }
        return capability;
    }

    public static describeForPrompt(): string {
        return this.list()
            .map((capability) => {
                const args = Object.entries((capability.inputSchema as z.ZodObject<any>).shape || {})
                    .map(([name]) => name)
                    .join(", ");
                return [
                    `${capability.id}: ${capability.description}`,
                    `cost=${capability.costClass}; latency=${capability.latencyClass}; evidence=${capability.evidenceRole}; args=${args || "none"}`,
                ].join("\n");
            })
            .join("\n\n");
    }
}
