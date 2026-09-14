import { z } from "zod";
import { T } from "@/shared/discordTools";
import { transcribeTaskAudio } from "@/runtime/sandbox/AudioTranscription";
import type { ToolDefinition } from "./types";

export const transcribeAudioTool: ToolDefinition = {
    name: T.sandbox_transcribe, catalog: { effect: "read", description: "Transcribe a source-labelled audio or video clip from the task workspace.", evidenceRole: "discovery_only" },
    schema: { description: "Extract and transcribe up to 180 seconds from an owned audio/video file. Decode inside the sandbox, send the WAV clip to a configured audio-capable model and save a transcript file. For longer sources, process consecutive windows. Uses the selected model unless profile is supplied. Profile must declare audio input and support chat completions.", parameters: { type: "object", properties: {
        path: { type: "string" }, start_seconds: { type: "number", description: "Nonnegative start; default 0." }, duration_seconds: { type: "number", description: "1–180 seconds; default 120." }, profile: { type: "string", description: "Existing audio-capable model profile name." }, language: { type: "string", description: "Optional language hint, not a translation request." },
    }, required: ["path"] } },
    capability: { description: "Transcribe task audio.", sideEffectLevel: "none", inputSchema: z.object({ path: z.string().min(1).max(240), start_seconds: z.number().finite().nonnegative().optional(), duration_seconds: z.number().min(1).max(180).optional(), profile: z.string().max(100).optional(), language: z.string().max(80).optional() }), outputSchema: z.any(), authRequirements: [], costClass: "normal", latencyClass: "slow", preconditions: [], postconditions: [],
        async run(context, args) {
            const result = await transcribeTaskAudio(context, { path: String(args.path), start: Number(args.start_seconds ?? 0), duration: Number(args.duration_seconds ?? 120), profile: args.profile as string | undefined, language: args.language as string | undefined });
            return { tool: T.sandbox_transcribe, summary: `Transcribed ${result.source} from ${result.startSeconds} to at most ${result.requestedEndSeconds} seconds. Saved ${result.transcriptPath}.`, data: result };
        } }, strategy: { extractEvidence: () => [] }, display: { icon: "🎧", labelPt: "Transcrever áudio" },
};
