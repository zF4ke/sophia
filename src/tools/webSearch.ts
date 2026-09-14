import { z } from "zod";
import { parseHTML } from "linkedom";
import { T } from "@/shared/discordTools";
import { SafeWebClient, publicWebUrl } from "@/runtime/web/SafeWebClient";
import { readablePage } from "@/runtime/web/ReadablePage";
import { taskStore } from "@/runtime/tasks/TaskStore";
import type { CapabilityContext, ToolDefinition } from "./types";
const common = { outputSchema: z.any(), authRequirements: [], costClass: "normal" as const, latencyClass: "medium" as const, preconditions: [], postconditions: [] };
type SearchResult = { title: string; url: string; snippet: string };
export function parseSearchResults(html: string, count: number): SearchResult[] {
    const { document } = parseHTML(html);
    const results: SearchResult[] = [];
    for (const block of document.querySelectorAll(".result")) {
        const anchor = block.querySelector("a.result__a");
        if (!anchor) continue;
        try {
            let url = new URL(anchor.getAttribute("href")!, "https://html.duckduckgo.com");
            if (url.hostname.endsWith("duckduckgo.com") && url.searchParams.has("uddg")) url = new URL(url.searchParams.get("uddg")!);
            const destination = publicWebUrl(url.toString()).toString();
            if (results.some(result => result.url === destination)) continue;
            results.push({ title: (anchor.textContent ?? destination).trim(), url: destination, snippet: (block.querySelector(".result__snippet")?.textContent ?? "").trim() });
            if (results.length >= count) break;
        } catch { /* A malformed result cannot become a source. */ }
    }
    if (!results.length && !document.querySelector(".no-results,.no-results__message")) throw new Error("Search provider returned an unrecognized or blocked page.");
    return results;
}
async function searchWeb(query: string, count: number, signal?: AbortSignal) {
    if (process.env.SERPER_API_KEY?.trim()) {
        const url = new URL("https://google.serper.dev/search");
        url.searchParams.set("q", query);
        url.searchParams.set("num", String(count));
        const response = await SafeWebClient.read(url.toString(), { signal, headers: { "X-API-KEY": process.env.SERPER_API_KEY.trim() } });
        const parsed = z.object({ organic: z.array(z.object({ title: z.string(), link: z.string(), snippet: z.string().optional() })) }).safeParse(JSON.parse(response.body));
        if (!parsed.success) throw new Error("Serper returned an invalid search response. Search did not complete.");
        const results: SearchResult[] = [];
        for (const result of parsed.data.organic) {
            try {
                const url = publicWebUrl(result.link).toString();
                if (!results.some(existing => existing.url === url)) results.push({ title: result.title, url, snippet: result.snippet ?? "" });
            } catch { /* Invalid destinations cannot become sources. */ }
            if (results.length >= count) break;
        }
        if (parsed.data.organic.length && !results.length) throw new Error("Serper returned results, but none had supported public URLs.");
        return { provider: "serper", results };
    }
    if (process.env.BRAVE_SEARCH_API_KEY) {
            const response = await SafeWebClient.read(`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${count}`, { signal, headers: { "X-Subscription-Token": process.env.BRAVE_SEARCH_API_KEY } });
            const data = z.object({ web: z.object({ results: z.array(z.object({ title: z.string(), url: z.string(), description: z.string().optional() })) }).optional() }).parse(JSON.parse(response.body));
            return { provider: "brave", results: (data.web?.results ?? []).slice(0, count).map(result => ({ title: result.title, url: publicWebUrl(result.url).toString(), snippet: result.description ?? "" })) };
    }
    const response = await SafeWebClient.read(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`, { signal });
    return { provider: "duckduckgo", results: parseSearchResults(response.body, count) };
}
export const webSearchTool: ToolDefinition = {
    name: T.web_search, catalog: { effect: "read", description: "Search public web pages and return candidate sources.", evidenceRole: "discovery_only" },
    schema: { description: "Search the web through configured Serper, Brave, or keyless DuckDuckGo. Results are snippets, not inspected page evidence. Open relevant pages with fetch_url before relying on their contents. Provider failures are distinct from an empty search.", parameters: { type: "object", properties: { query: { type: "string" }, count: { type: "number" } }, required: ["query"] } },
    capability: { ...common, description: "Search public web.", sideEffectLevel: "none", inputSchema: z.object({ query: z.string().min(2).max(400), count: z.number().int().min(1).max(10).optional() }), async run(context, args) {
        const query = String(args.query);
        const result = await searchWeb(query, Number(args.count ?? 5), context.execution?.signal);
        return { tool: T.web_search, summary: `${result.results.length} candidate web sources for ${query}.`, data: { query, ...result, evidenceType: "search_snippets", fetchedAt: new Date().toISOString() } };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "🌐", labelPt: "Pesquisar na web" },
};
interface SavedPage { title: string; content: string; links: Array<{ text: string; url: string }>; url: string; requestedUrl: string; fetchedAt: string; contentType: string }
function sliceSource(sourceId: string, page: SavedPage, offset: number, maxChars: number) {
    if (offset > page.content.length) throw new Error("Offset is past the end of this source.");
    const content = page.content.slice(offset, offset + maxChars);
    return { sourceId, title: page.title, url: page.url, requestedUrl: page.requestedUrl, fetchedAt: page.fetchedAt, contentType: page.contentType,
        content, totalCharacters: page.content.length, offset, nextOffset: offset + content.length < page.content.length ? offset + content.length : null,
        truncated: offset + content.length < page.content.length, links: page.links.slice(0, 50), evidenceType: "opened_page" };
}
function owner(context: CapabilityContext) {
    if (!context.taskId || !context.actorId || !context.currentChannelId) throw new Error("An owned task is required to preserve web evidence.");
    return [context.taskId, context.actorId, context.currentChannelId, context.guild?.id ?? null] as const;
}
export const fetchUrlTool: ToolDefinition = {
    name: T.fetch_url, catalog: { effect: "read", description: "Read a public page and preserve a task-owned source.", evidenceRole: "discovery_only" },
    schema: { description: "Read public HTTP(S) text/HTML/JSON. Preserves the full extracted source in this task and returns its first section, final URL, title and timestamp. Continue with source_read using sourceId and nextOffset. Does not execute page scripts or access private networks. Transport limit is 2 MiB.", parameters: { type: "object", properties: { url: { type: "string" }, max_chars: { type: "number" } }, required: ["url"] } },
    capability: { ...common, description: "Read a public page.", sideEffectLevel: "none", inputSchema: z.object({ url: z.string().url(), max_chars: z.number().int().min(500).max(20000).optional() }), async run(context, args) {
        const task = owner(context);
        const response = await SafeWebClient.read(String(args.url), { signal: context.execution?.signal });
        const page: SavedPage = { ...readablePage(response), url: response.url, requestedUrl: String(args.url), fetchedAt: new Date().toISOString() };
        const id = await taskStore.saveSource(...task, page);
        return { tool: T.fetch_url, summary: `Read ${page.title} at ${page.url}. Source ${id}, ${page.content.length} characters.`, data: sliceSource(id, page, 0, Number(args.max_chars ?? 8000)) };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "📄", labelPt: "Ler página" },
};
export const sourceReadTool: ToolDefinition = {
    name: T.source_read, catalog: { effect: "read", description: "Read another section of a preserved task source.", evidenceRole: "discovery_only" },
    schema: { description: "Read a preserved web source by sourceId and character offset. This reads the captured page without refetching or changing its timestamp. Sources belong to the current actor, task and location.", parameters: { type: "object", properties: { source_id: { type: "string" }, offset: { type: "number" }, max_chars: { type: "number" } }, required: ["source_id"] } },
    capability: { ...common, description: "Read preserved source.", sideEffectLevel: "none", inputSchema: z.object({ source_id: z.string(), offset: z.number().int().nonnegative().optional(), max_chars: z.number().int().min(500).max(20000).optional() }), async run(context, args) {
        const id = String(args.source_id);
        const page = await taskStore.source(id, ...owner(context)) as SavedPage | null;
        if (!page) throw new Error("Source does not belong to this task and location.");
        return { tool: T.source_read, summary: `Read source ${id} at character ${args.offset ?? 0}.`, data: sliceSource(id, page, Number(args.offset ?? 0), Number(args.max_chars ?? 8000)) };
    } }, strategy: { extractEvidence: () => [] }, display: { icon: "📄", labelPt: "Continuar a ler fonte" },
};
