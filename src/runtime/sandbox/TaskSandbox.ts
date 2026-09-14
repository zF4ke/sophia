import { ContainerSandbox, MAX_WORKSPACE_BYTES, safeWorkspacePath, validateFiles, type SandboxFile } from "./ContainerSandbox";
import { taskStore } from "../tasks/TaskStore";
import type { CapabilityContext } from "@/tools/types";
import { assertReadableChannels } from "@/security/SourceAccess";
import { ExecutionStopped } from "../ExecutionControl";

export class TaskSandbox {
    private static readonly locks = new Map<string, Promise<unknown>>();
    static async owned(context: CapabilityContext): Promise<string> {
        if (!context.taskId || !context.actorId || !await taskStore.ownsActive(context.taskId, context.actorId, context.currentChannelId ?? "", context.guild?.id ?? null)) throw new Error("An active owned task is required for the workspace.");
        return context.taskId;
    }
    static async files(context: CapabilityContext) {
        const id = await this.owned(context);
        const files = await taskStore.files(id, context.actorId!, context.currentChannelId ?? "", context.guild?.id ?? null);
        const sources = files.flatMap(file => file.sourceMessageIds ?? []);
        context.execution?.watchSources(sources);
        if (await taskStore.hasDeletedSources(sources)) throw new ExecutionStopped("source_changed");
        await assertReadableChannels(context.guild, context.actorId, files.flatMap(file => file.sourceChannelIds ?? []), { client: context.client, privateResponse: context.privateResponse, destinationChannelId: context.currentChannelId });
        return files;
    }
    static async change<T>(context: CapabilityContext, operation: (files: SandboxFile[]) => Promise<{ files: SandboxFile[]; result: T }>): Promise<T> {
        const id = await this.owned(context);
        const previous = this.locks.get(id) ?? Promise.resolve();
        const work = previous.catch(() => {}).then(async () => {
            const current = await this.files(context);
            const output = await operation(current);
            if (context.execution?.signal.aborted) throw new ExecutionStopped(context.execution.sourceInvalidated ? "source_changed" : "cancelled");
            const sourceMessageIds = [...new Set(current.flatMap(file => file.sourceMessageIds ?? []))];
            const sourceChannelIds = [...new Set(current.flatMap(file => file.sourceChannelIds ?? []))];
            // Generated outputs may depend on any input; preserve that audience
            // even when a program renames, summarizes or replaces the inputs.
            output.files = output.files.map(file => {
                const original = current.find(input => input.path === file.path && input.data === file.data);
                return original ?? { ...file, sourceMessageIds: [...new Set([...sourceMessageIds, ...(file.sourceMessageIds ?? [])])], sourceChannelIds: [...new Set([...sourceChannelIds, ...(file.sourceChannelIds ?? [])])] };
            });
            await assertReadableChannels(context.guild, context.actorId, output.files.flatMap(file => file.sourceChannelIds ?? []), { client: context.client, privateResponse: context.privateResponse, destinationChannelId: context.currentChannelId });
            validateFiles(output.files);
            if (context.execution?.signal.aborted) throw new ExecutionStopped(context.execution.sourceInvalidated ? "source_changed" : "cancelled");
            await taskStore.replaceFiles(id, context.actorId!, context.currentChannelId ?? "", context.guild?.id ?? null, output.files, context.requestId ?? undefined);
            return output.result;
        });
        this.locks.set(id, work);
        try { return await work; } finally { if (this.locks.get(id) === work) this.locks.delete(id); }
    }
    static async run(context: CapabilityContext, input: { language: "python" | "javascript" | "shell"; code: string; timeoutMs?: number }) {
        return this.change(context, async files => {
            const result = await ContainerSandbox.execute({ ...input, files }, context.execution?.signal);
            return { files: result.files, result: { exitCode: result.exitCode, stdout: result.stdout, stderr: result.stderr,
                files: result.files.map(file => ({ path: file.path, bytes: Buffer.byteLength(file.data, "base64") })) } };
        });
    }
    static async importAttachment(context: CapabilityContext, attachmentId: string, destination: string) {
        if (!safeWorkspacePath(destination)) throw new Error("Invalid workspace path.");
        const attachment = context.attachments?.find(file => file.id === attachmentId);
        if (!attachment) throw new Error("Attachment is not available in this turn.");
        const url = new URL(attachment.url);
        if (url.protocol !== "https:" || !["cdn.discordapp.com", "media.discordapp.net"].includes(url.hostname) || attachment.size > MAX_WORKSPACE_BYTES) throw new Error("Unsupported attachment URL or size.");
        return this.change(context, async files => {
            const deadline = AbortSignal.timeout(30_000);
            const response = await fetch(url, { redirect: "error", signal: context.execution?.signal ? AbortSignal.any([deadline, context.execution.signal]) : deadline });
            if (!response.ok || !response.body) throw new Error(`Attachment download failed (${response.status}). The link may have expired.`);
            const chunks: Uint8Array[] = [];
            let size = 0;
            for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
                size += chunk.length;
                if (size > MAX_WORKSPACE_BYTES) throw new Error("Attachment exceeds the workspace transfer limit.");
                chunks.push(chunk);
            }
            const next = [...files.filter(file => file.path !== destination), { path: destination, data: Buffer.concat(chunks).toString("base64") }];
            return { files: next, result: { path: destination, bytes: size } };
        });
    }
}
