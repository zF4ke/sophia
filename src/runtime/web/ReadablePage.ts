import { parseHTML } from "linkedom";
import type { WebResponse } from "./SafeWebClient";
export function readablePage(response: WebResponse) {
    const contentType = String(response.headers["content-type"] ?? "text/plain").toLowerCase();
    if (!contentType.includes("html") && !contentType.startsWith("text/") && !contentType.includes("json") && !contentType.includes("xml")) throw new Error(`Unsupported page type ${contentType}. Import files through the workspace instead.`);
    if (!contentType.includes("html")) return { title: response.url, content: response.body, links: [] as Array<{ text: string; url: string }>, contentType };
    const { document } = parseHTML(response.body);
    const title = document.querySelector("title")?.textContent?.trim() || response.url;
    for (const node of document.querySelectorAll("script,style,noscript,iframe,svg,nav,footer,form,[hidden],[aria-hidden='true']")) node.remove();
    const root = document.querySelector("article,main,[role='main']") ?? document.body ?? document.documentElement;
    const links = [...root.querySelectorAll("a[href]")].flatMap(link => {
        try { const url = new URL(link.getAttribute("href")!, response.url); return ["http:", "https:"].includes(url.protocol) ? [{ text: (link.textContent ?? "").trim(), url: url.toString() }] : []; } catch { return []; }
    });
    const content = ((root as unknown as { innerText?: string }).innerText ?? root.textContent ?? "").replace(/\r/g, "").replace(/[\t ]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
    return { title, content, links, contentType };
}
