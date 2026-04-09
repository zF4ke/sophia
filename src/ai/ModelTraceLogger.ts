import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { FileSystemService } from "@/platform/storage/FileSystemService";

type ChatMessage = {
    role: "system" | "user" | "assistant";
    content: string;
};

type ModelTraceEntry = {
    timestamp: string;
    callKind: "text" | "json";
    model: string;
    traceLabel: string;
    questionPreview: string | null;
    durationMs: number;
    webMode?: string;
    webContext?: string;
    webStatus?: string;
    webSearchRequests?: number;
    messages: ChatMessage[];
    rawOutput: string;
    normalizedOutput?: string;
    blankOutput?: boolean;
    parsedJson?: unknown;
    parseError?: string;
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
