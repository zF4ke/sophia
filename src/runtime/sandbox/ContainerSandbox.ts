import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import { SettingsService } from "@/app/SettingsService";

export const sandboxFileSchema = z.object({ path: z.string(), data: z.string() });
export const sandboxResultSchema = z.object({ exitCode: z.number().int(), stdout: z.string().max(200000), stderr: z.string().max(200000), files: z.array(sandboxFileSchema).max(100) });
export type SandboxFile = z.infer<typeof sandboxFileSchema> & { sourceMessageIds?: string[]; sourceChannelIds?: string[] };
export type SandboxResult = z.infer<typeof sandboxResultSchema>;
export const MAX_WORKSPACE_BYTES = 20 * 1024 * 1024;

export function safeWorkspacePath(value: string): boolean {
    return value.length > 0 && value.length < 240 && value.split("/").every(part => part !== "" && part !== "." && part !== ".." &&
        !/[\\<>:"|?*\x00-\x1f]/.test(part) && !/[. ]$/.test(part) && !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part));
}
export function validateFiles(files: SandboxFile[]): void {
    let total = 0;
    const names = new Set<string>();
    for (const file of files) {
        if (!safeWorkspacePath(file.path) || names.has(file.path.toLowerCase()) || file.data.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(file.data)) throw new Error("Invalid workspace file.");
        names.add(file.path.toLowerCase());
        total += Buffer.byteLength(file.data, "base64");
    }
    if (files.length > 100 || total > MAX_WORKSPACE_BYTES) throw new Error("Workspace transfer exceeds 100 files or 20 MiB. Process a smaller selection.");
}
export function sandboxDockerArgs(name: string, image: string): string[] {
    return ["run", "--rm", "--pull=never", "--name", name, "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--pids-limit=64", "--memory=512m", "--memory-swap=512m", "--cpus=1", "--user=65534:65534", "--log-driver=none",
        "--tmpfs=/workspace:rw,nosuid,nodev,noexec,size=128m,mode=1777", "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=128m,mode=1777",
        ...["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"].flatMap(key => ["--env", `${key}=`]),
        "--env", "HOME=/tmp", "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_NUM_THREADS=1", "-i", "--", image];
}
export class ContainerSandbox {
    static async execute(input: { language: "python" | "javascript" | "shell"; code: string; files?: SandboxFile[]; timeoutMs?: number }, signal?: AbortSignal): Promise<SandboxResult> {
        validateFiles(input.files ?? []);
        if (!SettingsService.load().sandbox.enabled) throw new Error("The isolated workspace is disabled in settings.");
        const timeoutMs = input.timeoutMs ?? 60_000;
        if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 100 || timeoutMs > 300_000 || input.code.length > 100_000) throw new Error("Invalid sandbox operation limits.");
        if (signal?.aborted) throw new Error("Sandbox operation cancelled.");
        const name = `sophia-${randomUUID()}`;
        const result = await new Promise<string>((resolve, reject) => {
            const child = spawn("docker", sandboxDockerArgs(name, SettingsService.load().sandbox.image), { windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
            const chunks: Buffer[] = [];
            let bytes = 0;
            let errors = "";
            let settled = false;
            const finish = (error?: Error, value?: string) => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                signal?.removeEventListener("abort", stop);
                if (error) reject(error); else resolve(value!);
            };
            const stop = () => {
                if (settled) return;
                const cleanup = spawn("docker", ["rm", "-f", name], { windowsHide: true, stdio: "ignore" });
                cleanup.on("error", () => {});
                const cleanupTimer = setTimeout(() => cleanup.kill(), 10_000);
                cleanupTimer.unref();
                cleanup.once("close", () => clearTimeout(cleanupTimer));
                child.kill();
                finish(new Error("Sandbox operation cancelled, timed out, or exceeded output limits."));
            };
            const timer = setTimeout(stop, timeoutMs + 10_000);
            signal?.addEventListener("abort", stop, { once: true });
            child.stdout.on("data", (chunk: Buffer) => { bytes += chunk.length; if (bytes > MAX_WORKSPACE_BYTES * 1.5) stop(); else chunks.push(chunk); });
            child.stderr.on("data", (chunk: Buffer) => { errors = (errors + chunk.toString()).slice(-4000); });
            child.on("error", error => finish(error));
            child.stdin.on("error", () => {});
            child.on("close", code => { code === 0 ? finish(undefined, Buffer.concat(chunks).toString("utf8")) : finish(new Error(`Isolated workspace unavailable: ${errors || `Docker exited ${code}`}`)); });
            child.stdin.end(JSON.stringify({ ...input, timeoutMs }));
        });
        const output = sandboxResultSchema.parse(JSON.parse(result));
        validateFiles(output.files);
        return output;
    }
}
