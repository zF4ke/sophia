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

// Free, no API key needed — scrapes DuckDuckGo lite HTML.
async function duckDuckGoLiteSearch(query: string, count: number): Promise<{ title: string; url: string; snippet: string }[]> {
    try {
        const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
        const res = await fetch(url, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Sophia/4.5; +Discord)",
                Accept: "text/html",
            },
            signal: AbortSignal.timeout(8000),
        });
        if (!res.ok) return [];
        const html = await res.text();
        const out: { title: string; url: string; snippet: string }[] = [];
        // DDG lite: each result is <a class="result__a" href="...">title</a> + <a class="result__snippet">
        const titleRe = /<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
        const snippetRe = /<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/gi;
        const titles: { url: string; title: string }[] = [];
        let m: RegExpExecArray | null;
        while ((m = titleRe.exec(html)) !== null) {
            const rawUrl = m[1];
            // DDG wraps with /l/?uddg=...
            let decoded = rawUrl;
            try {
                if (rawUrl.includes("uddg=")) {
                    const u = new URL("https://duckduckgo.com" + rawUrl);
                    decoded = decodeURIComponent(u.searchParams.get("uddg") || rawUrl);
                } else if (rawUrl.startsWith("//")) {
                    decoded = "https:" + rawUrl;
                }
            } catch { /* keep raw */ }
            const title = m[2].replace(/<[^>]+>/g, "").trim().slice(0, 120);
            if (decoded.startsWith("http")) titles.push({ url: decoded, title });
            if (titles.length >= count) break;
        }
        const snippets: string[] = [];
        while ((m = snippetRe.exec(html)) !== null) {
            snippets.push(m[1].replace(/<[^>]+>/g, "").trim().slice(0, 300));
            if (snippets.length >= count) break;
        }
        for (let i = 0; i < titles.length; i++) {
            out.push({ title: titles[i].title || "Result", url: titles[i].url, snippet: snippets[i] || "" });
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
        description: "Search the web for free (no API key needed) — DuckDuckGo lite scrape with Brave optional upgrade. Returns titles, URLs, snippets.",
        evidenceRole: "discovery_only",
    },
    schema: {
        description:
            "Free web search (no API key needed). Searches DuckDuckGo lite HTML for free; uses Brave API only if BRAVE_SEARCH_API_KEY is set. Use when the user asks about real-world facts, news, docs, or anything beyond Discord. Returns titles, URLs and snippets. Follow with fetch_url to read a page. Always free.",
        parameters: webSearchParams,
    },
    capability: {
        description: "Free web search via DuckDuckGo lite scrape (Brave optional).",
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
            // Free-first: Brave is optional (needs key), lite scrape is always free.
            let results = await braveSearch(query, count);
            if (results.length === 0) results = await duckDuckGoLiteSearch(query, count);
            if (results.length === 0) results = await duckDuckGoSearch(query, count);
            if (results.length === 0) {
                return {
                    tool: T.web_search,
                    summary: `No web results for "${query}" (try rephrasing or a more specific query).`,
                    data: { query, results: [], hint: "Try a different query or fetch a specific URL via fetch_url.", free: true },
                };
            }
            const summary = results.map((r) => `- ${r.title}: ${r.url}`).join("\n");
            return {
                tool: T.web_search,
                summary: `Found ${results.length} results for "${query}":\n${summary}`,
                data: { query, results, free: true },
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
