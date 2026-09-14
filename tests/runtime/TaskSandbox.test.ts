import { afterEach, expect, it, vi } from "vitest";
import { taskStore } from "@/runtime/tasks/TaskStore";
import { TaskSandbox } from "@/runtime/sandbox/TaskSandbox";
import { ContainerSandbox } from "@/runtime/sandbox/ContainerSandbox";
import { inspectTaskMedia } from "@/runtime/sandbox/MediaInspection";
import type { CapabilityContext } from "@/tools/types";
import { sandboxInspectTool, sandboxPublishTool } from "@/tools/sandbox";

afterEach(() => vi.restoreAllMocks());
it("inherits source provenance through generated files and rechecks revoked channel access", async () => {
    const taskId = await taskStore.create({ actorId: "owner", guildId: "guild", channelId: "channel", conversationId: "channel", objective: "Analyse source" });
    const has = vi.fn().mockReturnValue(true);
    const guild = { id: "guild", members: { fetch: vi.fn().mockResolvedValue({}) }, channels: { fetch: vi.fn().mockResolvedValue({ isTextBased: () => true, permissionsFor: () => ({ has }) }) } };
    const c = { taskId, actorId: "owner", guild: guild as never, currentChannelId: "channel", question: "Analyse" };
    await taskStore.replaceFiles(taskId, "owner", "channel", "guild", [{ path: "source.json", data: "YQ==", sourceMessageIds: ["source-message"], sourceChannelIds: ["private-channel"] }]);
    vi.spyOn(ContainerSandbox, "execute").mockResolvedValue({ exitCode: 0, stdout: "", stderr: "", files: [{ path: "summary.txt", data: "Yg==" }] });
    await TaskSandbox.run(c, { language: "python", code: "pass" });
    expect(await TaskSandbox.files(c)).toEqual([{ path: "summary.txt", data: "Yg==", sourceMessageIds: ["source-message"], sourceChannelIds: ["private-channel"] }]);
    has.mockReturnValue(false);
    await expect(TaskSandbox.files(c)).rejects.toThrow("not available");
    await taskStore.invalidateCorpusMessage("source-message");
    expect(await taskStore.files(taskId, "owner", "channel", "guild")).toEqual([]);
});
it("rejects visual inspection before execution for a text-only profile", async () => {
    const execute = vi.spyOn(ContainerSandbox, "execute");
    await expect(sandboxInspectTool.capability.run({ ...await context(), inputModalities: ["text"] }, { paths: ["photo.png"] })).rejects.toThrow("does not accept images");
    expect(execute).not.toHaveBeenCalled();
});
it("publishes a task file only to its authenticated owner's DM", async () => {
    const c = await context();
    await TaskSandbox.change(c, async () => ({ files: [{ path: "result.txt", data: "YQ==" }], result: null }));
    const send = vi.fn().mockResolvedValue({ id: "message", url: "https://discord.com/channels/@me/dm/message" });
    const destination = { id: "dm", recipientId: "other", isSendable: () => true, isDMBased: () => true, send };
    c.client = { channels: { fetch: vi.fn().mockResolvedValue(destination) } } as any;
    await expect(sandboxPublishTool.capability.run(c, { path: "result.txt" })).rejects.toThrow("does not belong");
    expect(send).not.toHaveBeenCalled();
    destination.recipientId = "owner";
    const result = await sandboxPublishTool.capability.run(c, { path: "result.txt" });
    expect(result.data).toMatchObject({ messageId: "message", channelId: "dm" });
    expect(send).toHaveBeenCalledOnce();
});
async function context(): Promise<CapabilityContext> {
    const taskId = await taskStore.create({ actorId: "owner", guildId: null, channelId: "dm", conversationId: "dm", objective: "Analyse file" });
    return { taskId, actorId: "owner", guild: null, currentChannelId: "dm", question: "Analyse file" };
}
it("serializes simultaneous file changes without losing either result", async () => {
    const c = await context();
    const edit = (path: string) => TaskSandbox.change(c, async files => ({ files: [...files, { path, data: "YQ==" }], result: path }));
    await Promise.all([edit("first.txt"), edit("second.txt")]);
    expect((await TaskSandbox.files(c)).map(file => file.path)).toEqual(["first.txt", "second.txt"]);
    await expect(TaskSandbox.files({ ...c, actorId: "other" })).rejects.toThrow("owned task");
    await expect(TaskSandbox.files({ ...c, currentChannelId: "another" })).rejects.toThrow("owned task");
});
it("preserves existing files when a container returns an invalid export", async () => {
    const c = await context();
    await TaskSandbox.change(c, async () => ({ files: [{ path: "keep.txt", data: "YQ==" }], result: null }));
    vi.spyOn(ContainerSandbox, "execute").mockResolvedValue({ exitCode: 0, stdout: "", stderr: "", files: [{ path: "../escape", data: "YQ==" }] });
    await expect(TaskSandbox.run(c, { language: "python", code: "pass" })).rejects.toThrow("Invalid workspace file");
    expect(await TaskSandbox.files(c)).toEqual([{ path: "keep.txt", data: "YQ==" }]);
});
it("rejects foreign attachment IDs and paths before any download", async () => {
    const c = await context();
    const download = vi.spyOn(globalThis, "fetch");
    await expect(TaskSandbox.importAttachment(c, "not-supplied", "file.png")).rejects.toThrow("not available");
    await expect(TaskSandbox.importAttachment(c, "any", "../file.png")).rejects.toThrow("Invalid workspace path");
    expect(download).not.toHaveBeenCalled();
});
it("returns source-labelled visual bytes while preserving other task files", async () => {
    const c = await context();
    await TaskSandbox.change(c, async () => ({ files: [{ path: "clip.mp4", data: "YQ==" }], result: null }));
    vi.spyOn(ContainerSandbox, "execute").mockImplementation(async input => {
        const prefix = input.code.match(/\.preview-[a-f0-9-]+/)![0];
        return { exitCode: 0, stdout: "untrusted decoder output", stderr: "", files: [...input.files!, { path: `${prefix}-0.jpg`, data: Buffer.from([0xff, 0xd8, 0xff, 0x00]).toString("base64") }] };
    });
    const inspected = await inspectTaskMedia(c, ["clip.mp4"], [12.5]);
    expect(inspected.previews[0].label).toContain("12.5 seconds");
    expect(inspected.previews[0].label).toContain("audio not inspected");
    expect(inspected.previews[0].url).toMatch(/^data:image\/jpeg;base64,/);
    expect((await TaskSandbox.files(c)).some(file => file.path === "clip.mp4")).toBe(true);
    await expect(inspectTaskMedia(c, ["clip.mp4"], [-1])).rejects.toThrow("nonnegative");
});
