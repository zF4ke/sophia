import { z } from "zod";
import { T } from "@/shared/discordTools";
import { DiscordGuildDiscoveryService } from "@/discord/live/DiscordGuildDiscoveryService";
import type { EvidenceItem } from "@/runtime/contracts";
import type {
    DiscordToolResult,
    GuildStructureEntry,
    ResolvedChannelTarget,
} from "@/shared/appTypes";
import type { ToolDefinition } from "./types";

// ── Payload type ────────────────────────────────────────────────────

export type GuildStructurePayload = {
    query?: string | null;
    entries?: Array<Record<string, unknown>>;
    focusedEntries?: Array<Record<string, unknown>>;
    focusedResolvedIds?: string[];
};

// ── Shared helpers (used by resolveChannelTargets too) ───────────────

export function normalizeLookupValue(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, " ")
        .trim();
}

export function isCategoryStructureEntry(
    entry: Pick<GuildStructureEntry, "type">,
): boolean {
    return (
        entry.type === "4" || entry.type.toLowerCase().includes("category")
    );
}

function formatTopic(topic: string | null | undefined): string {
    if (!topic) return "";
    const compact = topic.replace(/\s+/g, " ").trim();
    if (!compact) return "";
    const clipped = compact.length > 120 ? `${compact.slice(0, 117)}...` : compact;
    return ` Topic: ${clipped}.`;
}

export function asGuildStructureEntries(
    value: unknown,
): GuildStructureEntry[] {
    if (!Array.isArray(value)) return [];

    return value
        .filter(
            (item): item is Record<string, unknown> =>
                Boolean(item && typeof item === "object"),
        )
        .map((item) => ({
            id: String(item.id || ""),
            guildId: item.guildId == null ? null : String(item.guildId),
            name: String(item.name || ""),
            channelTopic:
                item.channelTopic == null
                    ? null
                    : String(item.channelTopic),
            type: String(item.type || "unknown"),
            parentCategoryId:
                item.parentCategoryId == null
                    ? null
                    : String(item.parentCategoryId),
            parentCategoryName:
                item.parentCategoryName == null
                    ? null
                    : String(item.parentCategoryName),
            isReadable: Boolean(item.isReadable),
            isViewable: Boolean(item.isViewable),
            isIndexed: Boolean(item.isIndexed),
            source: (
                item.source === "cached_only" ? "cached_only" : "live"
            ) as "live" | "cached_only",
            missingOrDeletedPossible: Boolean(
                item.missingOrDeletedPossible,
            ),
        }))
        .filter((entry) => entry.id && entry.name);
}

export function selectFocusedStructureEntries(
    entries: GuildStructureEntry[],
    query: string | null,
): GuildStructureEntry[] {
    if (!query) return [];

    const normalizedQuery = normalizeLookupValue(query);
    if (!normalizedQuery) return [];

    return entries
        .map((entry) => {
            const normalizedName = normalizeLookupValue(entry.name);
            const normalizedParent = normalizeLookupValue(
                entry.parentCategoryName || "",
            );
            let score = 0;

            if (normalizedName === normalizedQuery) score += 10;
            if (
                normalizedName &&
                normalizedQuery.includes(normalizedName)
            )
                score += 6;
            if (normalizedName.includes(normalizedQuery)) score += 5;
            if (
                normalizedParent &&
                normalizedQuery.includes(normalizedParent)
            )
                score += 3;
            if (normalizedParent && normalizedParent.includes(normalizedQuery))
                score += 2;

            return { entry, score };
        })
        .filter((item) => item.score > 0)
        .sort(
            (left, right) =>
                right.score - left.score ||
                left.entry.name.localeCompare(right.entry.name),
        )
        .slice(0, 4)
        .map((item) => item.entry);
}

export function buildStructureEvidenceItems(
    payload: GuildStructurePayload,
    summary: string,
): EvidenceItem[] {
    const allEntries = asGuildStructureEntries(payload.entries);
    const focusedEntries = asGuildStructureEntries(payload.focusedEntries);
    const query =
        typeof payload.query === "string" && payload.query.trim()
            ? payload.query.trim()
            : null;
    const targets = focusedEntries.length
        ? focusedEntries
        : selectFocusedStructureEntries(allEntries, query);

    if (!targets.length) {
        return allEntries.slice(0, 8).map((entry) => ({
            tool: T.list_guild_structure,
            summary,
            content: isCategoryStructureEntry(entry)
                ? `Category ${entry.name}. Viewable=${entry.isViewable ? "yes" : "no"}.`
                : `Channel #${entry.name}${entry.parentCategoryName ? ` in ${entry.parentCategoryName}` : ""}.${formatTopic(entry.channelTopic)} Readable=${entry.isReadable ? "yes" : "no"}. Indexed=${entry.isIndexed ? "yes" : "no"}.`,
            evidenceRole: "discovery_only" as const,
            strength: "metadata" as const,
            sourceOrigin: "none" as const,
            channelId: entry.id,
            channelName: entry.name,
        }));
    }

    const evidence: EvidenceItem[] = [];
    const seen = new Set<string>();

    for (const entry of targets) {
        if (seen.has(entry.id)) continue;
        seen.add(entry.id);

        if (isCategoryStructureEntry(entry)) {
            const children = allEntries.filter(
                (c) => c.parentCategoryId === entry.id,
            );
            const readableChildren = children.filter((c) => c.isReadable);
            const indexedChildren = readableChildren.filter(
                (c) => c.isIndexed,
            );
            const childLabels = readableChildren
                .slice(0, 6)
                .map((c) => `#${c.name}`);
            const childPhrase = childLabels.length
                ? childLabels.join(", ")
                : "none";

            evidence.push({
                tool: T.list_guild_structure,
                summary,
                content: `Category ${entry.name}. Visible channels under ${entry.name}: ${childPhrase}. Readable children: ${readableChildren.length}. Indexed children: ${indexedChildren.length}.`,
                evidenceRole: "discovery_only" as const,
                strength: "metadata",
                sourceOrigin: "none",
                channelId: entry.id,
                channelName: entry.name,
            });

            for (const child of readableChildren.slice(0, 4)) {
                if (seen.has(child.id)) continue;
                seen.add(child.id);
                evidence.push({
                    tool: T.list_guild_structure,
                    summary,
                    content: `Channel #${child.name} in category ${entry.name}.${formatTopic(child.channelTopic)} Readable=yes. Indexed=${child.isIndexed ? "yes" : "no"}.`,
                    evidenceRole: "discovery_only" as const,
                    strength: "metadata",
                    sourceOrigin: "none",
                    channelId: child.id,
                    channelName: child.name,
                });
            }
            continue;
        }

        evidence.push({
            tool: T.list_guild_structure,
            summary,
            content: `Channel #${entry.name}${entry.parentCategoryName ? ` in category ${entry.parentCategoryName}` : ""}.${formatTopic(entry.channelTopic)} Readable=${entry.isReadable ? "yes" : "no"}. Indexed=${entry.isIndexed ? "yes" : "no"}.`,
            evidenceRole: "discovery_only" as const,
            strength: "metadata",
            sourceOrigin: "none",
            channelId: entry.id,
            channelName: entry.name,
        });
    }

    return evidence;
}

// ── Schema parameters ───────────────────────────────────────────────

const parameters = {
    type: "object",
    properties: {
        targetText: {
            type: "string",
            description:
                "Optional text to focus on specific channels/categories.",
        },
    },
    required: [],
} as const;

// ── Tool definition ─────────────────────────────────────────────────

export const listGuildStructureTool: ToolDefinition = {
    name: T.list_guild_structure,

    catalog: {
        effect: "read",
        description:
            "List the current guild's readable channels and categories plus cached-only remembered entries.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "List channels and categories in the current guild with their IDs and types. Use this to discover which channels exist before searching them. Returns channel IDs you can pass to retrieve_messages.",
        parameters,
    },

    capability: {
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
        preconditions: ["guild context should exist"],
        postconditions: [
            "returns current-guild structure and cached-only remembered entries",
        ],
        async run(context, args) {
            const entries =
                await DiscordGuildDiscoveryService.listGuildStructure(
                    context.guild,
                );
            const targetText =
                typeof args.targetText === "string" &&
                args.targetText.trim()
                    ? args.targetText.trim()
                    : null;
            const focused = targetText
                ? await DiscordGuildDiscoveryService.resolveChannelTargets(
                      context.guild,
                      targetText,
                      context.currentChannelId,
                  )
                : null;
            return {
                tool: T.list_guild_structure,
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

    strategy: {
        extractEvidence(run: DiscordToolResult) {
            if (!run.data) return [];
            return buildStructureEvidenceItems(
                run.data as GuildStructurePayload,
                run.summary,
            );
        },

        extractResolvedChannel(run: DiscordToolResult): ResolvedChannelTarget | null {
            if (!run.data || typeof run.data !== "object") return null;

            const item = run.data as GuildStructurePayload &
                Record<string, unknown>;
            const entries = asGuildStructureEntries(item.focusedEntries);
            const resolvedIds = Array.isArray(item.focusedResolvedIds)
                ? (item.focusedResolvedIds as unknown[]).map(String)
                : [];
            const query =
                typeof item.query === "string" && item.query.trim()
                    ? item.query.trim()
                    : "";

            if (!query && !entries.length && !resolvedIds.length)
                return null;

            return {
                query,
                resolvedIds,
                entries,
                exactIdMatch: Boolean(item.exactIdMatch),
                confidence:
                    item.confidence === "exact" ||
                    item.confidence === "high" ||
                    item.confidence === "medium" ||
                    item.confidence === "low"
                        ? item.confidence
                        : resolvedIds.length
                          ? "high"
                          : "low",
            };
        },
    },

    display: { icon: "🗂️", labelPt: "Ler estrutura do servidor" },
};
