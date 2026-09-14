import http from "node:http";
import https from "node:https";
import { lookup } from "node:dns/promises";
import { isIP } from "node:net";

export function isPublicAddress(address: string): boolean {
    if (isIP(address) === 4) {
        const octets = address.split(".").map(Number);
        const number = octets.reduce((n, octet) => ((n << 8) | octet) >>> 0, 0);
        const denied: Array<[number, number]> = [[0x00000000, 8], [0x0a000000, 8], [0x64400000, 10], [0x7f000000, 8], [0xa9fe0000, 16],
            [0xac100000, 12], [0xc0000000, 24], [0xc0000200, 24], [0xc0586300, 24], [0xc0a80000, 16], [0xc6120000, 15], [0xc6336400, 24], [0xcb007100, 24], [0xe0000000, 4], [0xf0000000, 4]];
        return address !== "168.63.129.16" && !denied.some(([network, bits]) => (number >>> (32 - bits)) === (network >>> (32 - bits)));
    }
    if (isIP(address) !== 6 || address.includes("%")) return false;
    const [first, second] = address.toLowerCase().split(":").map(part => parseInt(part || "0", 16));
    return first >= 0x2000 && first <= 0x3ffe && first !== 0x2002 && !(first === 0x2001 && (second < 0x200 || second === 0xdb8));
}
export function publicWebUrl(raw: string): URL {
    const url = new URL(raw);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || (url.port && url.port !== "80" && url.port !== "443")) throw new Error("Only public HTTP(S) pages on standard ports are supported.");
    const host = url.hostname.replace(/^\[|\]$/g, "");
    if (host.toLowerCase() === "localhost" || host.endsWith(".localhost") || (isIP(host) && !isPublicAddress(host))) throw new Error("Private or reserved network destinations are unavailable.");
    url.hash = "";
    return url;
}
export interface WebResponse { url: string; status: number; headers: http.IncomingHttpHeaders; body: string }
export class SafeWebClient {
    static readonly maxBytes = 2 * 1024 * 1024;
    static async resolveHost(host: string) { return lookup(host, { all: true, verbatim: true }); }
    static async requestOnce(url: URL, address: { address: string; family: number }, signal: AbortSignal, headers: Record<string, string>): Promise<WebResponse> {
        return new Promise((resolve, reject) => {
            const request = (url.protocol === "https:" ? https : http).request(url, {
                agent: false, signal, headers: { "User-Agent": "Sophia/5", Accept: "text/html, text/plain, application/json", "Accept-Encoding": "identity", ...headers },
                // Resolve exactly once and pin the socket to that validated address.
                lookup: ((_host: string, options: { all?: boolean }, callback: (...args: unknown[]) => void) => options.all ? callback(null, [address]) : callback(null, address.address, address.family)) as never,
            }, response => {
                const chunks: Buffer[] = [];
                let bytes = 0;
                response.on("data", (chunk: Buffer) => {
                    bytes += chunk.length;
                    if (bytes > this.maxBytes) { response.destroy(new Error("Page exceeds the 2 MiB read limit.")); return; }
                    chunks.push(chunk);
                });
                response.on("error", reject);
                response.on("end", () => {
                    if (response.headers["content-encoding"] && response.headers["content-encoding"] !== "identity") { reject(new Error("Server returned unsupported compressed content.")); return; }
                    resolve({ url: url.toString(), status: response.statusCode ?? 0, headers: response.headers, body: Buffer.concat(chunks).toString("utf8") });
                });
            });
            request.on("error", reject);
            request.end();
        });
    }
    static async read(raw: string, options: { signal?: AbortSignal; headers?: Record<string, string> } = {}): Promise<WebResponse> {
        const deadline = AbortSignal.timeout(20_000);
        const signal = options.signal ? AbortSignal.any([deadline, options.signal]) : deadline;
        let url = publicWebUrl(raw);
        for (let hop = 0; hop <= 5; hop++) {
            signal.throwIfAborted();
            const hostname = url.hostname.replace(/^\[|\]$/g, "");
            let onAbort: (() => void) | undefined;
            const addresses = await Promise.race([
                this.resolveHost(hostname),
                new Promise<never>((_, reject) => { onAbort = () => reject(new Error("Web read cancelled or timed out.")); signal.addEventListener("abort", onAbort, { once: true }); }),
            ]).finally(() => { if (onAbort) signal.removeEventListener("abort", onAbort); });
            if (!addresses.length || addresses.some(address => !isPublicAddress(address.address))) throw new Error("Host resolves to a private or reserved network address.");
            signal.throwIfAborted();
            const response = await this.requestOnce(url, addresses[0], signal, hop === 0 ? options.headers ?? {} : {});
            if ([301, 302, 303, 307, 308].includes(response.status) && response.headers.location) {
                url = publicWebUrl(new URL(response.headers.location, url).toString());
                continue;
            }
            if (response.status < 200 || response.status >= 300) throw new Error(`HTTP ${response.status} from ${url.hostname}.`);
            return response;
        }
        throw new Error("Too many page redirects.");
    }
}
