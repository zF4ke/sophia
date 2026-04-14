import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { FileSystemService } from "@/shared/storage/FileSystemService";

type ChatMessage = {
    role: "system" | "user" | "assistant";
    content: string;
};

type ToolTraceMessage =
    | ChatMessage
    | {
          role: "assistant";
          content: string | null;
          tool_calls: unknown;
      }
    | {
          role: "tool";
          tool_call_id: string;
          content: string;
      };

type ModelTraceEntry = {
    timestamp: string;
    callKind: "text" | "json" | "tool_chat";
    model: string;
    traceLabel: string;
    questionPreview: string | null;
    durationMs: number;
    webMode?: string;
    webContext?: string;
    webStatus?: string;
    webSearchRequests?: number;
    messages: ToolTraceMessage[];
    rawOutput: string;
    normalizedOutput?: string;
    blankOutput?: boolean;
    toolCalls?: unknown;
    finishReason?: string;
    parsedJson?: unknown;
    parseError?: string;
    traceEvents?: Array<{ label: string; detail: string; timestamp: number }>;
};

const LOGS_DIR = path.join(AppPaths.storageRoot, "logs");

function getLogFilePath(): string {
    const date = new Date().toISOString().slice(0, 10);
    return path.join(LOGS_DIR, `model-output-${date}.jsonl`);
}

export class ModelTraceLogger {
    public static log(entry: ModelTraceEntry): void {
        try {
            FileSystemService.ensureDirectoryExists(LOGS_DIR);
            fs.appendFileSync(getLogFilePath(), `${JSON.stringify(entry)}\n`, "utf8");
        } catch (error) {
            console.error("Error writing model trace log:", error);
        }
    }
}

