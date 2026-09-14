import { z } from "zod";
import { AttachmentBuilder } from "discord.js";
import { TaskSandbox } from "@/runtime/sandbox/TaskSandbox";
import { inspectTaskMedia } from "@/runtime/sandbox/MediaInspection";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";
const common = { outputSchema: z.any(), authRequirements: [], costClass: "normal" as const, latencyClass: "slow" as const, preconditions: [], postconditions: [] };
export const sandboxInspectTool: ToolDefinition = {
    name: T.sandbox_inspect, catalog: { effect: "read", description: "View workspace images or timestamped video frames.", evidenceRole: "discovery_only" },
    schema: { description: "Inspect 1–8 workspace images, or one video at explicit timestamps in seconds. Returns actual image previews to your next model turn. Video inspection samples frames only; it does not transcribe audio or inspect intervening frames. Use ffprobe in sandbox_run to find duration first.", parameters: { type: "object", properties: { paths: { type: "array", items: { type: "string" } }, timestamps_seconds: { type: "array", items: { type: "number" } } }, required: ["paths"] } },
    capability: { ...common, description: "Inspect task media.", sideEffectLevel: "none", inputSchema: z.object({ paths: z.array(z.string()).min(1).max(8), timestamps_seconds: z.array(z.number().finite().nonnegative()).min(1).max(8).optional() }),
        async run(context, args) {
            if (!context.inputModalities?.includes("image")) {
                throw new Error("The current model does not accept images. Select an image-capable profile before visual inspection.");
            }
            const result = await inspectTaskMedia(context, args.paths as string[], args.timestamps_seconds as number[] | undefined);
            return { tool: T.sandbox_inspect, summary: `Inspected ${result.previews.length} previews from ${result.sources.join(", ")}.`, data: { sources: result.sources, timestampsSeconds: result.timestamps, descriptions: result.previews.map(image => image.label) }, images: result.previews };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "🔎", labelPt: "Inspecionar ficheiros" },
};
export const sandboxRunTool: ToolDefinition = {
    name: T.sandbox_run, catalog: { effect: "read", description: "Run isolated Python, JavaScript or shell code on task files.", evidenceRole: "discovery_only" },
    schema: { description: "Execute code in an isolated container with no network or host access. Task files live in /workspace; Python includes pandas, matplotlib, Pillow, pypdf, openpyxl. Node and ffmpeg are available. Files persist between calls. Operations can run up to 300 seconds; this is a process resource limit, not a task budget. Export large work in parts. Use sandbox_import for attached files and sandbox_publish to deliver an output.", parameters: {
        type: "object", properties: { language: { type: "string", enum: ["python", "javascript", "shell"] }, code: { type: "string" }, timeout_ms: { type: "number", description: "100–300000 milliseconds; default 60000." } }, required: ["language", "code"] } },
    capability: { ...common, description: "Run isolated code.", sideEffectLevel: "none", inputSchema: z.object({ language: z.enum(["python", "javascript", "shell"]), code: z.string().max(100000), timeout_ms: z.number().int().min(100).max(300000).optional() }),
        async run(context, args) {
            const output = await TaskSandbox.run(context, { language: args.language as "python" | "javascript" | "shell", code: String(args.code), timeoutMs: args.timeout_ms as number | undefined });
            return { tool: T.sandbox_run, summary: `Sandbox exited ${output.exitCode}; ${output.files.length} files available.`, data: output };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "🔧", labelPt: "Executar código" },
};
export const sandboxImportTool: ToolDefinition = {
    name: T.sandbox_import, catalog: { effect: "read", description: "Import a supplied attachment into the isolated task workspace.", evidenceRole: "discovery_only" },
    schema: { description: "Download an attachment supplied in this turn, by its authenticated attachment ID, to a relative workspace path. Other URLs and host files are unavailable.", parameters: { type: "object", properties: { attachment_id: { type: "string" }, path: { type: "string" } }, required: ["attachment_id", "path"] } },
    capability: { ...common, description: "Import attachment.", sideEffectLevel: "none", inputSchema: z.object({ attachment_id: z.string(), path: z.string().max(240) }),
        async run(context, args) { const output = await TaskSandbox.importAttachment(context, String(args.attachment_id), String(args.path)); return { tool: T.sandbox_import, summary: `Imported ${output.path} (${output.bytes} bytes).`, data: output }; } },
    strategy: { extractEvidence: () => [] }, display: { icon: "📎", labelPt: "Importar ficheiro" },
};
export const sandboxPublishTool: ToolDefinition = {
    publicationTarget: async (context) => {
        await TaskSandbox.files({ ...context, privateResponse: !context.guild });
        return context.currentChannelId ?? "";
    },
    name: T.sandbox_publish, catalog: { effect: "write", description: "Send a generated workspace file to the current channel.", evidenceRole: "discovery_only" },
    schema: { description: "Publish an existing task file in this channel, with the usual approval policy. Maximum attachment size 8 MiB. Returns message ID and link.", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path"] } },
    capability: { ...common, description: "Publish workspace file.", sideEffectLevel: "write", inputSchema: z.object({ path: z.string(), content: z.string().max(1500).optional() }),
        async run(context, args) {
            const file = (await TaskSandbox.files(context)).find(file => file.path === args.path);
            if (!file) throw new Error("Workspace file not found.");
            const data = Buffer.from(file.data, "base64");
            if (data.length > 8 * 1024 * 1024) throw new Error("File exceeds 8 MiB. Split or compress it first.");
            const channel = context.guild
                ? await context.guild.channels.fetch(context.currentChannelId ?? "")
                : await context.client?.channels.fetch(context.currentChannelId ?? "");
            if (!channel?.isSendable()) throw new Error("The destination channel is unavailable.");
            if (!context.guild && (!channel.isDMBased() || !("recipientId" in channel) || channel.recipientId !== context.actorId)) {
                throw new Error("The DM destination does not belong to the requester.");
            }
            const message = await channel.send({ content: String(args.content ?? "") || undefined, files: [new AttachmentBuilder(data, { name: file.path.split("/").at(-1)! })], allowedMentions: { parse: [] } });
            return { tool: T.sandbox_publish, summary: `Sent file in message ${message.id} to <#${channel.id}>.`, data: { messageId: message.id, channelId: channel.id, channelMention: `<#${channel.id}>`, jumpLink: message.url } };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "📎", labelPt: "Enviar ficheiro" },
};
