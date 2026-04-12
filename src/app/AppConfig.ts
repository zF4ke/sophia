import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";
import { FileSystemService } from "@/shared/storage/FileSystemService";
import type { AppConfig, ModelProfileConfig } from "@/shared/appTypes";

function readModelProfiles(): ModelProfileConfig {
    const raw = fs.readFileSync(AppPaths.modelProfilesPath, "utf8");
    const parsed = JSON.parse(raw) as ModelProfileConfig;

    if (!parsed.defaultProfile || !parsed.profiles?.[parsed.defaultProfile]) {
        throw new Error("Invalid model profile configuration.");
    }

    return parsed;
}

export function getAppConfig(): AppConfig {
    const profiles = readModelProfiles();
    const modelProfileName = process.env.MODEL_PROFILE || profiles.defaultProfile;
    const modelProfile = profiles.profiles[modelProfileName];

    if (!modelProfile) {
        throw new Error(`Unknown model profile "${modelProfileName}".`);
    }

    const discordToken = process.env.DISCORD_TOKEN || process.env.CLIENT_TOKEN;
    const openRouterApiKey = process.env.OPENROUTER_API_KEY || process.env.OPENAI_API_KEY;

    if (!discordToken) {
        throw new Error("Missing DISCORD_TOKEN.");
    }

    if (!openRouterApiKey) {
        throw new Error("Missing OPENROUTER_API_KEY.");
    }

    FileSystemService.ensureDirectoryExists(FileSystemService.getBaseStorageDir());
    const runtimeDir = path.join(AppPaths.storageRoot, "runtime");
    FileSystemService.ensureDirectoryExists(runtimeDir);

    return {
        discordToken,
        openRouterApiKey,
        openRouterBaseUrl:
            process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1",
        modelProfileName,
        modelProfile,
        runtime: {
            operationalDbPath:
                process.env.RUNTIME_OPERATIONAL_DB_PATH ||
                path.join(runtimeDir, "operational.sqlite"),
            checkpointDbPath:
                process.env.RUNTIME_CHECKPOINT_DB_PATH ||
                path.join(runtimeDir, "checkpoints.sqlite"),
            maxToolCalls: Number(process.env.RUNTIME_LOOP_MAX_TOOL_CALLS || 6),
            maxResearchPasses: Number(process.env.RUNTIME_LOOP_MAX_RESEARCH_PASSES || 4),
            maxRepeatedCallSignature: Number(
                process.env.RUNTIME_LOOP_MAX_REPEATED_CALL_SIGNATURE || 1
            ),
            maxLatencyBudgetMs: Number(process.env.RUNTIME_LOOP_MAX_LATENCY_BUDGET_MS || 15000),
            maxPriorTurns: Number(process.env.RUNTIME_CONTEXT_MAX_PRIOR_TURNS || 5),
            maxChannelMessages: Number(process.env.RUNTIME_CONTEXT_MAX_CHANNEL_MESSAGES || 15),
            maxToolRunsContext: Number(process.env.RUNTIME_CONTEXT_MAX_TOOL_RUNS || 12),
            maxEvidenceSlice: Number(process.env.RUNTIME_CONTEXT_MAX_CARRIED_EVIDENCE_ITEMS || 32),
            maxContextPreviewEvidenceItems: Number(
                process.env.DEBUG_CONTEXT_PREVIEW_MAX_EVIDENCE_ITEMS || 6
            ),
            maxResolveChannelTargetEvidenceItems: Number(
                process.env.TOOL_RESOLVE_CHANNEL_TARGETS_MAX_EVIDENCE_ITEMS || 6
            ),
            maxRetrieveHistoryEvidenceItems: Number(
                process.env.TOOL_RETRIEVE_MESSAGES_MAX_HISTORY_EVIDENCE_ITEMS || 30
            ),
            maxRetrieveSemanticEvidenceItems: Number(
                process.env.TOOL_RETRIEVE_MESSAGES_MAX_SEMANTIC_EVIDENCE_ITEMS || 30
            ),
            maxRetrieveEvidenceContentChars: Number(
                process.env.TOOL_RETRIEVE_MESSAGES_MAX_EVIDENCE_CONTENT_CHARS || 260
            ),
            interactiveCrawlLimit: Number(process.env.TOOL_RETRIEVE_MESSAGES_MAX_INTERACTIVE_CRAWL_MESSAGES || 250),
            escalationFetchLimit: Number(process.env.TOOL_RETRIEVE_MESSAGES_MAX_ESCALATION_FETCH_MESSAGES || 150),
            retrievalHistoryLimit: Number(process.env.TOOL_RETRIEVE_MESSAGES_DEFAULT_LIMIT || 50),
            retrievalContextWindow: Number(process.env.TOOL_RETRIEVE_MESSAGES_AROUND_CONTEXT_WINDOW || 15),
        },
    };
}
