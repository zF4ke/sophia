import type { EvidenceItem } from "@/runtime/contracts";
import { DISCORD_TOOL_EVIDENCE_ROLES } from "@/shared/discordTools";
import type { GuildStructureEntry } from "@/shared/appTypes";
import type { GuildStructurePayload } from "./types";

export function normalizeLookupValue(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, " ")
        .trim();
}

export function isCategoryStructureEntry(entry: Pick<GuildStructureEntry, "type">): boolean {
    return entry.type === "4" || entry.type.toLowerCase().includes("category");
}

export function asGuildStructureEntries(value: unknown): GuildStructureEntry[] {
    if (!Array.isArray(value)) {
        return [];
    }

    return value
        .filter((item): item is Record<string, unknown> => Boolean(item && typeof item === "object"))
        .map((item) => ({
            id: String(item.id || ""),
            guildId: item.guildId == null ? null : String(item.guildId),
            name: String(item.name || ""),
            type: String(item.type || "unknown"),
            parentCategoryId: item.parentCategoryId == null ? null : String(item.parentCategoryId),
            parentCategoryName:
                item.parentCategoryName == null ? null : String(item.parentCategoryName),
            isReadable: Boolean(item.isReadable),
            isViewable: Boolean(item.isViewable),
            isIndexed: Boolean(item.isIndexed),
            source: (item.source === "cached_only" ? "cached_only" : "live") as
                | "live"
                | "cached_only",
            missingOrDeletedPossible: Boolean(item.missingOrDeletedPossible),
        }))
        .filter((entry) => entry.id && entry.name);
}

export function selectFocusedStructureEntries(
    entries: GuildStructureEntry[],
    query: string | null
): GuildStructureEntry[] {
    if (!query) {
        return [];
    }

    const normalizedQuery = normalizeLookupValue(query);
    if (!normalizedQuery) {
        return [];
    }

    return entries
        .map((entry) => {
            const normalizedName = normalizeLookupValue(entry.name);
            const normalizedParent = normalizeLookupValue(entry.parentCategoryName || "");
            let score = 0;

            if (normalizedName === normalizedQuery) {
                score += 10;
            }
            if (normalizedName && normalizedQuery.includes(normalizedName)) {
                score += 6;
            }
            if (normalizedName.includes(normalizedQuery)) {
                score += 5;
            }
            if (normalizedParent && normalizedQuery.includes(normalizedParent)) {
                score += 3;
            }
            if (normalizedParent && normalizedParent.includes(normalizedQuery)) {
                score += 2;
            }

            return { entry, score };
        })
        .filter((item) => item.score > 0)
        .sort((left, right) => right.score - left.score || left.entry.name.localeCompare(right.entry.name))
        .slice(0, 4)
        .map((item) => item.entry);
}

export function buildStructureEvidenceItems(
    payload: GuildStructurePayload,
    summary: string
): EvidenceItem[] {
    const allEntries = asGuildStructureEntries(payload.entries);
    const focusedEntries = asGuildStructureEntries(payload.focusedEntries);
    const query = typeof payload.query === "string" && payload.query.trim() ? payload.query.trim() : null;
    const targets = focusedEntries.length ? focusedEntries : selectFocusedStructureEntries(allEntries, query);

    if (!targets.length) {
        return allEntries.slice(0, 8).map((entry) => ({
            tool: "list_guild_structure",
            summary,
            content: isCategoryStructureEntry(entry)
                ? `Category ${entry.name}. Viewable=${entry.isViewable ? "yes" : "no"}.`
                : `Channel #${entry.name}${entry.parentCategoryName ? ` in ${entry.parentCategoryName}` : ""}. Readable=${entry.isReadable ? "yes" : "no"}. Indexed=${entry.isIndexed ? "yes" : "no"}.`,
            evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
            strength: "metadata" as const,
            sourceOrigin: "none" as const,
            channelId: entry.id,
            channelName: entry.name,
        }));
    }

    const evidence: EvidenceItem[] = [];
    const seen = new Set<string>();

    for (const entry of targets) {
        if (seen.has(entry.id)) {
            continue;
        }
        seen.add(entry.id);

        if (isCategoryStructureEntry(entry)) {
            const children = allEntries.filter((candidate) => candidate.parentCategoryId === entry.id);
            const readableChildren = children.filter((candidate) => candidate.isReadable);
            const indexedChildren = readableChildren.filter((candidate) => candidate.isIndexed);
            const childLabels = readableChildren.slice(0, 6).map((candidate) => `#${candidate.name}`);
            const childPhrase = childLabels.length ? childLabels.join(", ") : "none";

            evidence.push({
                tool: "list_guild_structure",
                summary,
                content: `Category ${entry.name}. Visible channels under ${entry.name}: ${childPhrase}. Readable children: ${readableChildren.length}. Indexed children: ${indexedChildren.length}.`,
                evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
                strength: "metadata",
                sourceOrigin: "none",
                channelId: entry.id,
                channelName: entry.name,
            });

            for (const child of readableChildren.slice(0, 4)) {
                if (seen.has(child.id)) {
                    continue;
                }
                seen.add(child.id);
                evidence.push({
                    tool: "list_guild_structure",
                    summary,
                    content: `Channel #${child.name} in category ${entry.name}. Readable=yes. Indexed=${child.isIndexed ? "yes" : "no"}.`,
                    evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
                    strength: "metadata",
                    sourceOrigin: "none",
                    channelId: child.id,
                    channelName: child.name,
                });
            }
            continue;
        }

        evidence.push({
            tool: "list_guild_structure",
            summary,
            content: `Channel #${entry.name}${entry.parentCategoryName ? ` in category ${entry.parentCategoryName}` : ""}. Readable=${entry.isReadable ? "yes" : "no"}. Indexed=${entry.isIndexed ? "yes" : "no"}.`,
            evidenceRole: DISCORD_TOOL_EVIDENCE_ROLES.list_guild_structure,
            strength: "metadata",
            sourceOrigin: "none",
            channelId: entry.id,
            channelName: entry.name,
        });
    }

    return evidence;
}
