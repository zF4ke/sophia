import { describe, expect, it } from "vitest";
import { ContainerSandbox } from "@/runtime/sandbox/ContainerSandbox";
import { inspectTaskMedia } from "@/runtime/sandbox/MediaInspection";
import { taskStore } from "@/runtime/tasks/TaskStore";

describe.skipIf(process.env.LIVE_SANDBOX !== "1")("real container isolation", () => {
    it("runs all languages and transfers only workspace files", async () => {
        const py = await ContainerSandbox.execute({ language: "python", code: "import pathlib, pandas, matplotlib, pypdf, openpyxl\np=pathlib.Path('input.txt')\npathlib.Path('output.txt').write_text(p.read_text().upper())\nprint('PYTHON_OK')", files: [{ path: "input.txt", data: Buffer.from("hello").toString("base64") }] });
        expect(py.exitCode).toBe(0);
        expect(py.stdout).toContain("PYTHON_OK");
        expect(Buffer.from(py.files.find(f => f.path === "output.txt")!.data, "base64").toString()).toBe("HELLO");
        expect((await ContainerSandbox.execute({ language: "javascript", code: "console.log(6 * 7)" })).stdout.trim()).toBe("42");
        expect((await ContainerSandbox.execute({ language: "shell", code: "printf SHELL_OK" })).stdout).toBe("SHELL_OK");
    });
    it("blocks network, host secrets and root writes, and omits symbolic links", async () => {
        const result = await ContainerSandbox.execute({ language: "python", code: "import os, socket, pathlib\nassert os.getuid() == 65534\nassert not os.environ.get('DISCORD_TOKEN')\nassert not os.environ.get('OPENCODE_API_KEY')\ntry:\n pathlib.Path('/root-write').write_text('bad')\n raise AssertionError('root writable')\nexcept OSError as e:\n assert e.errno in (13, 30)\ns=socket.socket(); s.settimeout(2)\ntry:\n s.connect(('1.1.1.1',443))\n raise AssertionError('network connected')\nexcept OSError: pass\npathlib.Path('link').symlink_to('/etc/passwd')\nprint('ISOLATED')" });
        expect(result.exitCode, result.stderr).toBe(0);
        expect(result.stdout.trim()).toBe("ISOLATED");
        expect(result.files).toEqual([]);
    });
    it("bounds an operation that does not finish", async () => {
        const result = await ContainerSandbox.execute({ language: "python", code: "import time\ntime.sleep(20)", timeoutMs: 200 });
        expect(result.exitCode).toBe(124);
    });
    it("decodes a timestamped video through the owned task workspace", async () => {
        const video = await ContainerSandbox.execute({ language: "shell", code: "ffmpeg -v error -f lavfi -i color=c=blue:s=64x64:d=2 -c:v libx264 clip.mp4" });
        expect(video.exitCode, video.stderr).toBe(0);
        const taskId = await taskStore.create({ actorId: "media-owner", guildId: null, channelId: "media-dm", conversationId: "media-dm", objective: "Inspect generated video" });
        await taskStore.replaceFiles(taskId, "media-owner", "media-dm", null, video.files);
        const context = { taskId, actorId: "media-owner", guild: null, currentChannelId: "media-dm", question: "Inspect" };
        const output = await inspectTaskMedia(context, ["clip.mp4"], [0, 1]);
        expect(output.previews).toHaveLength(2);
        expect(output.timestamps).toEqual([0, 1]);
        expect(output.previews.every(p => p.url.startsWith("data:image/jpeg;base64,"))).toBe(true);
        await expect(inspectTaskMedia(context, ["clip.mp4"], [10])).rejects.toThrow("Timestamp outside video duration");
    });
});
