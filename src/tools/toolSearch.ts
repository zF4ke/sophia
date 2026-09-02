import { z } from "zod";
import { T } from "@/shared/discordTools";
import { getToolExposure } from "./toolExposure";
import { ALL_TOOLS } from "./registry";
import type { ToolDefinition } from "./types";

function tokenize(q: string): string[] {
    return q
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .split(/\s+/)
        .filter((t) => t.length > 1);
}

function scoreTool(queryTokens: string[], tool: ToolDefinition): number {
    const hay = `${tool.name} ${tool.catalog.description} ${tool.schema.description}`.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
    let s = 0;
    for (const tok of queryTokens) {
        if (hay.includes(tok)) s += 1;
        if (tool.name.includes(tok)) s += 2; // name match is stronger
    }
    return s;
}

const params = {
    type: "object",
    properties: {
        query: {
            type: "string",
            description: "Keyword to find a tool (e.g. 'delete channel', 'send message', 'create role', 'fetch url', 'memory save').",
        },
        limit: {
            type: "number",
            description: "Max tools to return (1-8, default 5).",
        },
    },
    required: ["query"],
} as const;

export const toolSearchTool: ToolDefinition = {
    name: T.tool_search,

    catalog: {
        effect: "read",
        description: "Search for deferred tools by keyword and expose their schemas for the next turn.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Search the deferred tool catalog (write/destructive, workflows, fetch_url, etc.) by keyword. Use when you need a tool that isn't in the initial list — e.g. to delete, create, edit, or fetch a URL. Returns matching tool names, descriptions, and usage hints. After this, you can call the discovered tool in the next turn.",
        parameters: params,
    },

    capability: {
        description: "Search deferred tools.",
        inputSchema: z.object({
            query: z.string().min(2).max(200),
            limit: z.number().min(1).max(8).optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["returns matching deferred tools"],
        async run(_context, args) {
            const query = String(args.query || "").trim();
            const limit = Math.max(1, Math.min(8, Number(args.limit) || 5));
            const tokens = tokenize(query);
            if (!tokens.length) {
                return { tool: T.tool_search, summary: "Empty query.", data: null, errorMessage: "Provide a keyword like 'delete channel' or 'send message'." };
            }
            const deferred = ALL_TOOLS.filter((t) => getToolExposure(t.name as never) === "deferred");
            const scored = deferred
                .map((tool) => ({ tool, score: scoreTool(tokens, tool) }))
                .filter((x) => x.score > 0)
                .sort((a, b) => b.score - a.score)
                .slice(0, limit);

            // Fallback: if nothing scored, return all deferred names so model sees inventory
            const results = scored.length ? scored : deferred.slice(0, limit).map((tool) => ({ tool, score: 0 }));

            const data = results.map(({ tool, score }) => ({
                name: tool.name,
                description: tool.schema.description.slice(0, 300),
                effect: tool.catalog.effect,
                score,
                hint: `Call ${tool.name} next turn with its parameters.`,
            }));

            const summary = data.map((d) => `• ${d.name} (${d.effect}): ${d.description.slice(0, 120)}`).join("\n");
            return {
                tool: T.tool_search,
                summary: `Found ${data.length} tools for "${query}":\n${summary}`,
                data: { query, results: data, note: "Now call the tool you need — it's available next turn." },
            };
        },
    },

    strategy: { extractEvidence() { return []; } },
    display: { icon: "🔧", labelPt: "Procurar ferramenta" },
};
