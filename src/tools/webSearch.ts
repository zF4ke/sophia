import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";

// Brave Search if BRAVE_SEARCH_API_KEY is set, otherwise DuckDuckGo fallback.
async function braveSearch(query: string, count: number): Promise<{ title: string; url: string; snippet: string }[]> {
    const key = process.env.BRAVE_SEARCH_API_KEY;
    if (!key) return [];
    try {
        const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${count}`;
        const res = await fetch(url, {
            headers: { "X-Subscription-Token": key, Accept: "application/json" },
            signal: AbortSignal.timeout(8000),
        });
        if (!res.ok) return [];
        const json = (await res.json()) as { web?: { results?: { title: string; url: string; description: string }[] } };
        return (json.web?.results ?? []).map((r) => ({ title: r.title, url: r.url, snippet: r.description }));
    } catch {
        return [];
    }
}

async function duckDuckGoSearch(query: string, count: number): Promise<{ title: string; url: string; snippet: string }[]> {
    try {
        // Use DuckDuckGo html lite scraping fallback via text search endpoint
        const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
        const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
        if (!res.ok) return [];
        const json = (await res.json()) as {
            RelatedTopics?: { Text?: string; FirstURL?: string; Result?: string }[];
            AbstractText?: string;
            AbstractURL?: string;
        };
        const out: { title: string; url: string; snippet: string }[] = [];
        if (json.AbstractText && json.AbstractURL) {
            out.push({ title: "Abstract", url: json.AbstractURL, snippet: json.AbstractText });
        }
        for (const t of json.RelatedTopics ?? []) {
            if (t.FirstURL && t.Text) {
                out.push({ title: t.Text.slice(0, 80), url: t.FirstURL, snippet: t.Text });
                if (out.length >= count) break;
            }
        }
        return out.slice(0, count);
    } catch {
        return [];
    }
}

const webSearchParams = {
    type: "object",
    properties: {
        query: {
            type: "string",
            description: "Search query for the web (Portuguese or English). Keep it concise and keyword-focused.",
        },
        count: {
            type: "number",
            description: "Number of results to return (1-10, default 5).",
        },
    },
    required: ["query"],
} as const;

export const webSearchTool: ToolDefinition = {
    name: T.web_search,
    catalog: {
        effect: "read",
        description: "Search the web for current information (news, docs, facts) and return titles, URLs, and snippets.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Search the internet for current information. Use when the user asks about real-world facts, news, documentation, or anything beyond Discord history. Returns titles, URLs and snippets. Follow up with fetch_url to read a specific page.",
        parameters: webSearchParams,
    },
    capability: {
        description: "Search the web via Brave/DuckDuckGo fallback.",
        inputSchema: z.object({
            query: z.string().min(2).max(400),
            count: z.number().min(1).max(10).optional(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [],
        postconditions: ["returns web search results with title, url, snippet"],
        async run(_context, args) {
            const query = String(args.query || "").trim();
            if (!query) return { tool: T.web_search, summary: "Empty query.", data: null, errorMessage: "Query is required." };
            const count = Math.max(1, Math.min(10, Number(args.count) || 5));
            let results = await braveSearch(query, count);
            if (results.length === 0) results = await duckDuckGoSearch(query, count);
            // Final fallback: if both empty, tell model to try fetch with a known URL pattern or rephrase
            if (results.length === 0) {
                return {
                    tool: T.web_search,
                    summary: `No web results for "${query}" (try rephrasing or a more specific query).`,
                    data: { query, results: [], hint: "Try a different query or fetch a specific URL via fetch_url." },
                };
            }
            const summary = results.map((r) => `- ${r.title}: ${r.url}`).join("\n");
            return {
                tool: T.web_search,
                summary: `Found ${results.length} results for "${query}":\n${summary}`,
                data: { query, results },
            };
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "🌐", labelPt: "Pesquisa na web" },
};

const fetchUrlParams = {
    type: "object",
    properties: {
        url: { type: "string", description: "Full https URL to fetch and extract text from." },
        max_chars: { type: "number", description: "Max characters to return (default 8000, max 20000)." },
    },
    required: ["url"],
} as const;

function stripHtml(html: string): string {
    return html
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 20000);
}

export const fetchUrlTool: ToolDefinition = {
    name: T.fetch_url,
    catalog: { effect: "read", description: "Fetch a URL and extract its readable text content.", evidenceRole: "discovery_only" },
    schema: {
        description: "Fetch a web page by URL and return its text content (stripped HTML, truncated). Use after web_search to read a specific result.",
        parameters: fetchUrlParams,
    },
    capability: {
        description: "Fetch URL content.",
        inputSchema: z.object({ url: z.string().url(), max_chars: z.number().min(500).max(20000).optional() }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "normal",
        latencyClass: "medium",
        preconditions: [],
        postconditions: ["returns page text"],
        async run(_context, args) {
            let raw = String(args.url || "").trim();
            try {
                const u = new URL(raw);
                if (!["http:", "https:"].includes(u.protocol)) throw new Error("Only http/https allowed");
            } catch {
                return { tool: T.fetch_url, summary: "Invalid URL.", data: null, errorMessage: "URL must be a valid http/https URL." };
            }
            const maxChars = Math.max(500, Math.min(20000, Number(args.max_chars) || 8000));
            try {
                const res = await fetch(raw, {
                    headers: { "User-Agent": "Sophia/4.5 (+Discord bot; fetch_url)", Accept: "text/html, text/plain" },
                    signal: AbortSignal.timeout(10000),
                });
                if (!res.ok) return { tool: T.fetch_url, summary: `Fetch failed: ${res.status}`, data: null, errorMessage: `HTTP ${res.status}` };
                const text = await res.text();
                const contentType = res.headers.get("content-type") || "";
                const extracted = contentType.includes("html") ? stripHtml(text) : text.slice(0, maxChars);
                const sliced = extracted.slice(0, maxChars);
                return {
                    tool: T.fetch_url,
                    summary: `Fetched ${raw} (${sliced.length} chars)`,
                    data: { url: raw, content: sliced, truncated: extracted.length > maxChars },
                };
            } catch (e) {
                const msg = e instanceof Error ? e.message : String(e);
                return { tool: T.fetch_url, summary: `Fetch error: ${msg}`, data: null, errorMessage: msg };
            }
        },
    },
    strategy: { extractEvidence() { return []; } },
    display: { icon: "📄", labelPt: "Fetch URL" },
};
