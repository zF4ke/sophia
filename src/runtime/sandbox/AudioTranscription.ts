import { randomUUID } from "node:crypto";
import { TaskSandbox } from "./TaskSandbox";
import { ContainerSandbox } from "./ContainerSandbox";
import { ModelGateway } from "@/ai/ModelGateway";
import { readModelProfiles } from "@/app/modelProfiles";
import { SettingsService } from "@/app/SettingsService";
import { profileUsesResponsesApi } from "@/ai/ResponsesAdapter";
import { PromptRegistry } from "../PromptRegistry";
import type { CapabilityContext } from "@/tools/types";

export async function transcribeTaskAudio(context: CapabilityContext, input: { path: string; start: number; duration: number; profile?: string; language?: string }) {
    const config = readModelProfiles();
    const selected = input.profile ?? SettingsService.load().modelProfile;
    const profile = config.profiles[selected];
    if (!profile?.inputModalities?.includes("audio") || profileUsesResponsesApi(profile)) {
        const available = Object.entries(config.profiles).filter(([, item]) => item.inputModalities?.includes("audio") && !profileUsesResponsesApi(item)).map(([name]) => name);
        throw new Error(`Choose a configured chat-completions model profile declaring audio input support. Available profiles: ${available.join(", ") || "none; configure one in model-profiles.json"}.`);
    }
    if (!Number.isFinite(input.start) || input.start < 0 || !Number.isFinite(input.duration) || input.duration <= 0 || input.duration > 180) throw new Error("Select a nonnegative start and 1–180 seconds of audio.");
    const clip = await TaskSandbox.change(context, async files => {
        if (!files.some(file => file.path === input.path)) throw new Error("Audio source is not in this task workspace.");
        const name = `.audio-${randomUUID()}.wav`;
        const request = JSON.stringify({ source: `/workspace/${input.path}`, name, start: input.start, duration: input.duration });
        const code = `import json,subprocess,wave\nr=json.loads(${JSON.stringify(request)})\nsubprocess.run(['ffmpeg','-v','error','-ss',str(r['start']),'-i',r['source'],'-t',str(r['duration']),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',r['name']],check=True)\nwith wave.open(r['name'],'rb') as w:\n    if w.getnframes()==0: raise ValueError('No audio in requested range')\n`;
        const output = await ContainerSandbox.execute({ language: "python", code, files, timeoutMs: 120000 }, context.execution?.signal);
        if (output.exitCode !== 0) throw new Error(`Audio decoding failed: ${output.stderr}`);
        const audio = output.files.find(file => file.path === name);
        const bytes = audio ? Buffer.from(audio.data, "base64") : Buffer.alloc(0);
        if (bytes.length < 44 || bytes.length > 8 * 1024 * 1024 || bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WAVE") throw new Error("Decoder returned no valid WAV clip.");
        return { files, result: audio!.data };
    });
    context.execution?.checkpoint();
    const transcript = await ModelGateway.generateText([
        { role: "system", content: PromptRegistry.load("media/transcription") },
        { role: "user", content: JSON.stringify({ source: input.path, startSeconds: input.start, requestedDurationSeconds: input.duration, languageHint: input.language ?? null }), audio: [{ data: clip, format: "wav" }] },
    ], { profile, temperature: 0, maxOutputTokens: Math.min(8192, profile.maxOutputTokens), traceContext: { traceLabel: "audio_transcription" } });
    context.execution?.checkpoint();
    if (!transcript.trim()) throw new Error("The transcription model returned no text.");
    const path = `transcript-${randomUUID()}.txt`;
    const text = `Source: ${input.path}\nRequested audio window: ${input.start}–${input.start + input.duration} seconds (may end earlier at EOF).\nModel: ${profile.chatModel}\n\n${transcript}`;
    await TaskSandbox.change(context, async files => ({ files: [...files, { path, data: Buffer.from(text).toString("base64") }], result: null }));
    return { source: input.path, startSeconds: input.start, requestedEndSeconds: input.start + input.duration, transcript, transcriptPath: path, model: profile.chatModel };
}
