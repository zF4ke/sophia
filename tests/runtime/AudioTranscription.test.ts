import { afterEach, expect, it, vi } from "vitest";
import { transcribeTaskAudio } from "@/runtime/sandbox/AudioTranscription";
import { ContainerSandbox } from "@/runtime/sandbox/ContainerSandbox";
import { TaskSandbox } from "@/runtime/sandbox/TaskSandbox";
import { ModelGateway, providerMessages } from "@/ai/ModelGateway";
import * as profiles from "@/app/modelProfiles";
import { taskStore } from "@/runtime/tasks/TaskStore";
afterEach(() => vi.restoreAllMocks());

it("sends actual audio bytes, preserves source boundaries and saves an owned transcript", async () => {
    vi.spyOn(profiles, "readModelProfiles").mockReturnValue({ defaultProfile: "audio", profiles: { audio: { chatModel: "configured/audio", embeddingModel: "configured/embedding", inputModalities: ["text", "audio"], temperature: 0, maxOutputTokens: 4000, contextWindow: 32000 } } });
    const taskId = await taskStore.create({ actorId: "owner", guildId: null, channelId: "dm", conversationId: "dm", objective: "Transcribe attachment" });
    const context = { taskId, actorId: "owner", guild: null, currentChannelId: "dm", question: "Transcribe" };
    await TaskSandbox.change(context, async () => ({ files: [{ path: "recording.mp4", data: "YQ==" }], result: null }));
    const wav = Buffer.alloc(44); wav.write("RIFF", 0); wav.write("WAVE", 8);
    vi.spyOn(ContainerSandbox, "execute").mockImplementation(async input => {
        const name = input.code.match(/\.audio-[a-f0-9-]+\.wav/)![0];
        return { exitCode: 0, stdout: "", stderr: "", files: [...input.files!, { path: name, data: wav.toString("base64") }] };
    });
    const generate = vi.spyOn(ModelGateway, "generateText").mockResolvedValue("Speaker 1: Hello. [inaudible]");
    const output = await transcribeTaskAudio(context, { path: "recording.mp4", start: 60, duration: 30, profile: "audio" });
    expect(output).toMatchObject({ source: "recording.mp4", startSeconds: 60, requestedEndSeconds: 90 });
    expect(providerMessages(generate.mock.calls[0][0])).toMatchObject([{}, { content: [{ type: "text" }, { type: "input_audio", input_audio: { format: "wav", data: wav.toString("base64") } }] }]);
    const saved = (await TaskSandbox.files(context)).find(file => file.path === output.transcriptPath)!;
    expect(Buffer.from(saved.data, "base64").toString()).toContain("[inaudible]");
    expect((await TaskSandbox.files(context)).some(file => file.path.startsWith(".audio-"))).toBe(false);
    await expect(transcribeTaskAudio(context, { path: "recording.mp4", start: 0, duration: 10, profile: "missing" })).rejects.toThrow("audio input");
});
