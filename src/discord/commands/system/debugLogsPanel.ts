import fs from "fs";
import path from "path";
import {
    ActionRowBuilder,
    ButtonBuilder,
    ButtonStyle,
    ContainerBuilder,
    MessageFlags,
    StringSelectMenuBuilder,
    TextDisplayBuilder,
} from "discord.js";
import { AppPaths } from "@/app/AppPaths";

export interface DebugLogTraceEntry {
    timestamp: string;
    callKind: string;
    model: string;
    traceLabel: string;
    questionPreview: string | null;
    durationMs: number;
    rawOutput: string;
    normalizedOutput?: string;
    messages?: unknown[];
    toolCalls?: unknown;
    finishReason?: string;
    parsedJson?: unknown;
    parseError?: string;
    webMode?: string;
    webContext?: string;
    webStatus?: string;
    webSearchRequests?: number;
    blankOutput?: boolean;
    traceEvents?: Array<{ label: string; detail: string; timestamp: number }>;
}

export type DebugLogsView = "overview" | "prompt" | "messages" | "tools" | "trace" | "notes" | "json";

export interface DebugLogsPanelState {
    ownerUserId: string;
    selectedFile: string;
    selectedEntryIndex: number;
    entryListPage: number;
    activeView: DebugLogsView;
    contentPage: number;
    updatedAt: number;
}

export interface DebugToolCallRecord {
    id: string;
    name: string;
    argumentsText: string;
    outputText: string;
    assistantText: string;
}

const LOGS_DIR = path.join(AppPaths.storageRoot, "logs");
const FILE_OPTIONS_LIMIT = 25;
const ENTRY_OPTIONS_PAGE_SIZE = 25;
const CONTENT_PAGE_CHARS = 3400;
const PANEL_TTL_MS = 6 * 60 * 60 * 1000;

const DEBUG_LOGS_FILE_SELECT_ID = "debug:logs:file";
const DEBUG_LOGS_ENTRY_SELECT_ID = "debug:logs:entry";
const DEBUG_LOGS_VIEW_SELECT_ID = "debug:logs:view";
const DEBUG_LOGS_ENTRY_PAGE_PREV_ID = "debug:logs:entry-page:prev";
const DEBUG_LOGS_ENTRY_PAGE_NEXT_ID = "debug:logs:entry-page:next";
const DEBUG_LOGS_CONTENT_PAGE_PREV_ID = "debug:logs:content-page:prev";
const DEBUG_LOGS_CONTENT_PAGE_NEXT_ID = "debug:logs:content-page:next";

const VIEW_LABELS: Record<DebugLogsView, string> = {
    overview: "Overview",
    prompt: "System Prompt",
    messages: "Messages",
    tools: "Tool Calls",
    trace: "Runtime Trace",
    notes: "Notes",
    json: "Raw JSON",
};

const panelStateByMessageId = new Map<string, DebugLogsPanelState>();

function trimText(value: string | null | undefined, maxLength = 140): string {
    const compact = String(value || "").replace(/\s+/g, " ").trim();
    if (!compact) {
        return "—";
    }
    return compact.length > maxLength ? `${compact.slice(0, maxLength - 3)}...` : compact;
}

function formatDuration(durationMs: number | null | undefined): string {
    if (!durationMs) {
        return "?";
    }
    return `${(durationMs / 1000).toFixed(2)}s`;
}

function formatTimestamp(timestamp: string | null | undefined): string {
    if (!timestamp) {
        return "unknown time";
    }

    try {
        return new Date(timestamp).toLocaleString("en-GB");
    } catch {
        return timestamp;
    }
}

function formatMaybeJson(value: unknown): string {
    if (typeof value === "string") {
        const trimmed = value.trim();
        if (!trimmed) {
            return "";
        }

        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            try {
                return JSON.stringify(JSON.parse(trimmed), null, 2);
            } catch {
                return value;
            }
        }

        return value;
    }

    if (value == null) {
        return "";
    }

    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

function splitIntoPages(content: string, maxChars = CONTENT_PAGE_CHARS): string[] {
    if (!content.trim()) {
        return ["No content."];
    }

    const normalized = content.replace(/\r\n/g, "\n");
    if (normalized.length <= maxChars) {
        return [normalized];
    }

    const lines = normalized.split("\n");
    const pages: string[] = [];
    let current = "";

    const pushCurrent = () => {
        if (current) {
            pages.push(current);
            current = "";
        }
    };

    for (const line of lines) {
        if (line.length > maxChars) {
            pushCurrent();
            for (let index = 0; index < line.length; index += maxChars) {
                pages.push(line.slice(index, index + maxChars));
            }
            continue;
        }

        const next = current ? `${current}\n${line}` : line;
        if (next.length > maxChars) {
            pushCurrent();
            current = line;
            continue;
        }

        current = next;
    }

    pushCurrent();
    return pages.length ? pages : ["No content."];
}

function asRecord(value: unknown): Record<string, unknown> | null {
    return value && typeof value === "object" ? (value as Record<string, unknown>) : null;
}

function asArray(value: unknown): unknown[] {
    return Array.isArray(value) ? value : [];
}

function getMessageRole(message: unknown): string {
    const record = asRecord(message);
    return typeof record?.role === "string" ? record.role : "unknown";
}

function getMessageContent(message: unknown): string {
    const record = asRecord(message);
    if (!record || !("content" in record)) {
        return "";
    }
    return typeof record.content === "string" ? record.content : formatMaybeJson(record.content);
}

function getToolCallId(message: unknown): string {
    const record = asRecord(message);
    return typeof record?.tool_call_id === "string" ? record.tool_call_id : "";
}

function getToolCalls(message: unknown): Array<Record<string, unknown>> {
    const record = asRecord(message);
    return asArray(record?.tool_calls).flatMap((item) => {
        const parsed = asRecord(item);
        return parsed ? [parsed] : [];
    });
}

function normalizeToolCall(call: Record<string, unknown>): { id: string; name: string; argumentsText: string } {
    const fn = asRecord(call.function);
    return {
        id: typeof call.id === "string" ? call.id : "unknown-call-id",
        name: typeof fn?.name === "string" ? fn.name : "unknown_tool",
        argumentsText:
            typeof fn?.arguments === "string"
                ? formatMaybeJson(fn.arguments)
                : formatMaybeJson(fn?.arguments),
    };
}

function extractToolOutputs(entry: DebugLogTraceEntry): Map<string, string> {
    const outputsById = new Map<string, string>();

    for (const message of asArray(entry.messages)) {
        if (getMessageRole(message) === "tool") {
            outputsById.set(getToolCallId(message), formatMaybeJson(getMessageContent(message)));
        }
    }

    return outputsById;
}

function extractTraceToolResultSummary(entry: DebugLogTraceEntry, toolName: string): string | null {
    for (const event of asArray(entry.traceEvents)) {
        const record = asRecord(event);
        if (record?.label !== "tool_result") {
            continue;
        }

        const detail = typeof record.detail === "string" ? record.detail : "";
        if (detail.startsWith(`${toolName}: `)) {
            return detail;
        }
    }

    return null;
}

function findToolOutputInLaterEntries(
    entries: DebugLogTraceEntry[],
    currentEntryIndex: number,
    toolCallId: string,
    toolName: string
): string | null {
    for (let index = currentEntryIndex - 1; index >= 0; index -= 1) {
        const output = extractToolOutputs(entries[index]).get(toolCallId);
        if (output) {
            return output;
        }

        const traceSummary = extractTraceToolResultSummary(entries[index], toolName);
        if (traceSummary) {
            return traceSummary;
        }
    }

    return null;
}

function extractFinalAnswer(entry: DebugLogTraceEntry): string {
    const records = extractToolCallRecords(entry);
    const finishCall = [...records].reverse().find((record) => record.name === "finish");
    if (finishCall?.argumentsText) {
        try {
            const parsed = JSON.parse(finishCall.argumentsText) as { answer?: string };
            if (typeof parsed.answer === "string" && parsed.answer.trim()) {
                return parsed.answer.trim();
            }
        } catch {
            // Ignore malformed finish args and fall back below.
        }
    }

    return entry.normalizedOutput?.trim() || entry.rawOutput?.trim() || "";
}

function buildOverviewText(
    entry: DebugLogTraceEntry,
    entryIndex: number,
    totalEntries: number,
    entries: DebugLogTraceEntry[] = []
): string {
    const finalAnswer = extractFinalAnswer(entry);
    const messageCount = Array.isArray(entry.messages) ? entry.messages.length : 0;
    const toolCount = extractToolCallRecords(entry, entries, entryIndex).length;

    return [
        `Entry: ${entryIndex + 1} of ${totalEntries}`,
        `Timestamp: ${formatTimestamp(entry.timestamp)}`,
        `Model: ${entry.model || "unknown"}`,
        `Trace Label: ${entry.traceLabel || "unlabeled"}`,
        `Call Kind: ${entry.callKind || "unknown"}`,
        `Duration: ${formatDuration(entry.durationMs)}`,
        `Finish Reason: ${entry.finishReason || "n/a"}`,
        `Web: mode=${entry.webMode || "off"}; status=${entry.webStatus || "off"}; requests=${entry.webSearchRequests || 0}`,
        `Messages: ${messageCount}`,
        `Tool Calls: ${toolCount}`,
        `Blank Output: ${entry.blankOutput ? "yes" : "no"}`,
        "",
        `Question Preview:\n${entry.questionPreview || "—"}`,
        "",
        `Final Answer Preview:\n${finalAnswer || "—"}`,
    ].join("\n");
}

function buildPromptText(entry: DebugLogTraceEntry): string {
    const systemPrompt = asArray(entry.messages).find((message) => getMessageRole(message) === "system");
    return getMessageContent(systemPrompt) || "No system prompt recorded.";
}

function buildMessagesText(entry: DebugLogTraceEntry): string {
    const messages = asArray(entry.messages).filter((m) => getMessageRole(m) !== "system");
    if (!messages.length) {
        return "No messages recorded.";
    }

    return messages
        .map((message, index) => {
            const role = getMessageRole(message).toUpperCase();
            const toolCalls = getToolCalls(message).map(normalizeToolCall);
            const content = getMessageContent(message);
            const toolCallId = getToolCallId(message);

            const lines = [`[${index + 1}] ${role}`];
            if (toolCalls.length) {
                lines.push(
                    "Tool Calls:",
                    ...toolCalls.map((call, toolIndex) =>
                        `  ${toolIndex + 1}. ${call.name} (${call.id})\n${call.argumentsText || "{}"}`
                    )
                );
            }
            if (toolCallId) {
                lines.push(`Tool Call Id: ${toolCallId}`);
            }
            if (content) {
                lines.push("Content:", content);
            }
            return lines.join("\n");
        })
        .join("\n\n");
}

export function extractToolCallRecords(
    entry: DebugLogTraceEntry,
    entries: DebugLogTraceEntry[] = [],
    entryIndex = -1
): DebugToolCallRecord[] {
    const messages = asArray(entry.messages);
    const outputsById = extractToolOutputs(entry);

    const records: DebugToolCallRecord[] = [];
    const recordedIds = new Set<string>();

    for (const message of messages) {
        if (getMessageRole(message) !== "assistant") {
            continue;
        }

        const assistantText = getMessageContent(message);
        for (const toolCall of getToolCalls(message).map(normalizeToolCall)) {
            records.push({
                id: toolCall.id,
                name: toolCall.name,
                argumentsText: toolCall.argumentsText || "{}",
                outputText: outputsById.get(toolCall.id) || "No tool output recorded.",
                assistantText,
            });
            recordedIds.add(toolCall.id);
        }
    }

    // Include outgoing calls from this iteration (not yet in messages; output is in the next entry)
    for (const item of asArray(entry.toolCalls)) {
        const parsed = asRecord(item);
        if (!parsed) continue;
        const call = normalizeToolCall(parsed);
        if (recordedIds.has(call.id)) continue;
        const laterOutput =
            entryIndex >= 0 && entries.length
                ? findToolOutputInLaterEntries(entries, entryIndex, call.id, call.name)
                : null;
        records.push({
            id: call.id,
            name: call.name,
            argumentsText: call.argumentsText || "{}",
            outputText: laterOutput || "(output logged in a later entry)",
            assistantText: "",
        });
    }

    return records;
}

function buildToolsText(entry: DebugLogTraceEntry, entries: DebugLogTraceEntry[] = [], entryIndex = -1): string {
    const records = extractToolCallRecords(entry, entries, entryIndex);
    if (!records.length) {
        return "No tool calls recorded for this entry.";
    }

    return records
        .map((record, index) => {
            const lines = [
                `Tool ${index + 1}: ${record.name}`,
                `Call Id: ${record.id}`,
            ];

            if (record.assistantText) {
                lines.push("Assistant Content:", record.assistantText);
            }

            lines.push("Arguments:", record.argumentsText || "{}", "Output:", record.outputText || "No output.");
            return lines.join("\n");
        })
        .join("\n\n");
}

function buildNotesText(entry: DebugLogTraceEntry, entries: DebugLogTraceEntry[] = [], entryIndex = -1): string {
    const records = extractToolCallRecords(entry, entries, entryIndex);
    const noteRecords = records.filter((r) => r.name === "note_add" || r.name === "note_list" || r.name === "note_clear");

    const blocks: string[] = [];

    if (noteRecords.length) {
        blocks.push(
            "## Note tool calls",
            ...noteRecords.map((record, index) => {
                const lines = [`#${index + 1} ${record.name}`];
                lines.push("Arguments:", record.argumentsText || "{}");
                lines.push("Output:", record.outputText || "No output.");
                return lines.join("\n");
            })
        );
    }

    const partialIndexBlock = buildPartialIndexBlock(records);
    if (partialIndexBlock) {
        blocks.push("", "## Partial index (retrieve_messages)", partialIndexBlock);
    }

    if (!blocks.length) {
        return "No notes or partial-index signals recorded for this entry.";
    }

    return blocks.join("\n\n");
}

function buildPartialIndexBlock(records: DebugToolCallRecord[]): string | null {
    const sections: string[] = [];
    for (const record of records) {
        if (record.name !== "retrieve_messages") continue;
        const output = record.outputText || "";
        if (!output.includes("partialIndex")) continue;
        try {
            // Try extract JSON block
            const firstBrace = output.indexOf("{");
            if (firstBrace < 0) continue;
            const parsed = JSON.parse(output.slice(firstBrace));
            const partial = (parsed as Record<string, unknown>)?.partialIndex;
            const hints = (parsed as Record<string, unknown>)?.partialIndexHints;
            if (!partial && !hints) continue;
            const lines = [`Call ${record.id} (${record.name}):`];
            if (partial && typeof partial === "object") {
                for (const [channelId, info] of Object.entries(partial as Record<string, unknown>)) {
                    lines.push(`• #${channelId}: ${JSON.stringify(info)}`);
                }
            }
            if (Array.isArray(hints)) {
                lines.push("Hints:", ...hints.map((h) => `- ${String(h)}`));
            }
            sections.push(lines.join("\n"));
        } catch {
            // skip
        }
    }
    return sections.length ? sections.join("\n\n") : null;
}

function buildTraceText(entry: DebugLogTraceEntry): string {
    const events = entry.traceEvents;
    if (!events || !events.length) {
        return "No runtime trace events recorded for this entry.\n\nTrace events are only attached to agent_loop_iter_N log entries.";
    }

    const firstTs = events[0]?.timestamp ?? 0;
    return events
        .map((e) => {
            const relMs = Math.max(0, e.timestamp - firstTs);
            const relS = (relMs / 1000).toFixed(2);
            return `+${relS}s [${e.label}] ${e.detail}`;
        })
        .join("\n");
}

export function buildLogEntryViewText(
    entry: DebugLogTraceEntry,
    view: DebugLogsView,
    entryIndex = 0,
    totalEntries = 1,
    entries: DebugLogTraceEntry[] = []
): string {
    if (view === "prompt") {
        return buildPromptText(entry);
    }
    if (view === "messages") {
        return buildMessagesText(entry);
    }
    if (view === "tools") {
        return buildToolsText(entry, entries, entryIndex);
    }
    if (view === "trace") {
        return buildTraceText(entry);
    }
    if (view === "notes") {
        return buildNotesText(entry, entries, entryIndex);
    }
    if (view === "json") {
        return JSON.stringify(entry, null, 2);
    }
    return buildOverviewText(entry, entryIndex, totalEntries, entries);
}

function pruneExpiredStates(): void {
    const now = Date.now();
    for (const [messageId, state] of panelStateByMessageId.entries()) {
        if (now - state.updatedAt > PANEL_TTL_MS) {
            panelStateByMessageId.delete(messageId);
        }
    }
}

export function getLogFiles(): string[] {
    if (!fs.existsSync(LOGS_DIR)) {
        return [];
    }

    return fs.readdirSync(LOGS_DIR)
        .filter((file) => file.startsWith("model-output-") && file.endsWith(".jsonl"))
        .sort()
        .reverse();
}

export function readLogEntries(filename: string): DebugLogTraceEntry[] {
    const filePath = path.join(LOGS_DIR, filename);
    if (!fs.existsSync(filePath)) {
        return [];
    }

    const lines = fs.readFileSync(filePath, "utf8")
        .split(/\r?\n/)
        .filter(Boolean)
        .reverse();
    const entries: DebugLogTraceEntry[] = [];

    for (const line of lines) {
        try {
            entries.push(JSON.parse(line) as DebugLogTraceEntry);
        } catch {
            // Skip malformed JSONL rows.
        }
    }

    return entries;
}

function clampState(state: DebugLogsPanelState): DebugLogsPanelState {
    const files = getLogFiles().slice(0, FILE_OPTIONS_LIMIT);
    const selectedFile = files.includes(state.selectedFile) ? state.selectedFile : files[0] || "";
    const entries = selectedFile ? readLogEntries(selectedFile) : [];
    const maxEntryIndex = Math.max(0, entries.length - 1);
    const selectedEntryIndex = Math.min(Math.max(state.selectedEntryIndex, 0), maxEntryIndex);
    const maxEntryListPage = Math.max(0, Math.ceil(entries.length / ENTRY_OPTIONS_PAGE_SIZE) - 1);
    const derivedEntryListPage = Math.floor(selectedEntryIndex / ENTRY_OPTIONS_PAGE_SIZE);
    const fixedEntryListPage = entries.length ? derivedEntryListPage : Math.min(Math.max(state.entryListPage, 0), maxEntryListPage);
    const entry = entries[selectedEntryIndex];
    const contentPages = entry
        ? splitIntoPages(buildLogEntryViewText(entry, state.activeView, selectedEntryIndex, entries.length, entries))
        : ["No log entries found."];
    const maxContentPage = Math.max(0, contentPages.length - 1);

    return {
        ...state,
        selectedFile,
        selectedEntryIndex,
        entryListPage: fixedEntryListPage,
        contentPage: Math.min(Math.max(state.contentPage, 0), maxContentPage),
    };
}

export function createInitialDebugLogsPanelState(ownerUserId: string, preferredFile?: string | null): DebugLogsPanelState {
    const files = getLogFiles().slice(0, FILE_OPTIONS_LIMIT);
    return clampState({
        ownerUserId,
        selectedFile: preferredFile && files.includes(preferredFile) ? preferredFile : files[0] || "",
        selectedEntryIndex: 0,
        entryListPage: 0,
        activeView: "overview",
        contentPage: 0,
        updatedAt: Date.now(),
    });
}

export function rememberDebugLogsPanelState(messageId: string, state: DebugLogsPanelState): void {
    pruneExpiredStates();
    panelStateByMessageId.set(messageId, clampState({ ...state, updatedAt: Date.now() }));
}

export function getDebugLogsPanelState(messageId: string): DebugLogsPanelState | null {
    pruneExpiredStates();
    const state = panelStateByMessageId.get(messageId);
    return state ? clampState(state) : null;
}

export function buildDebugLogsPanel(state: DebugLogsPanelState) {
    const normalizedState = clampState(state);
    const files = getLogFiles().slice(0, FILE_OPTIONS_LIMIT);

    if (!files.length) {
        return {
            components: [
                new ContainerBuilder()
                    .setAccentColor(0x99aab5)
                    .addTextDisplayComponents(
                        new TextDisplayBuilder().setContent("## Debug Logs"),
                        new TextDisplayBuilder().setContent("No log files found in storage/logs.")
                    ),
            ],
            flags: MessageFlags.IsComponentsV2 as const,
        };
    }

    const entries = readLogEntries(normalizedState.selectedFile);
    const selectedEntry = entries[normalizedState.selectedEntryIndex];
    const entryStart = normalizedState.entryListPage * ENTRY_OPTIONS_PAGE_SIZE;
    const entryOptions = entries.slice(entryStart, entryStart + ENTRY_OPTIONS_PAGE_SIZE);
    const entryPageCount = Math.max(1, Math.ceil(entries.length / ENTRY_OPTIONS_PAGE_SIZE));

    const viewText = selectedEntry
        ? buildLogEntryViewText(
              selectedEntry,
              normalizedState.activeView,
              normalizedState.selectedEntryIndex,
                            entries.length,
                            entries
          )
        : "No log entries found in this file.";
    const contentPages = splitIntoPages(viewText);
    const currentContent = contentPages[normalizedState.contentPage] || contentPages[0] || "No content.";
    const finalAnswer = selectedEntry ? extractFinalAnswer(selectedEntry) : "";

    const header = new ContainerBuilder()
        .setAccentColor(0x5865f2)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent("## Debug Logs Panel"),
            new TextDisplayBuilder().setContent(
                [
                    `File: ${normalizedState.selectedFile}`,
                    `Entry: ${entries.length ? normalizedState.selectedEntryIndex + 1 : 0} of ${entries.length}`,
                    `Entry Page: ${normalizedState.entryListPage + 1} of ${entryPageCount}`,
                    `View: ${VIEW_LABELS[normalizedState.activeView]}`,
                    `Content Page: ${normalizedState.contentPage + 1} of ${contentPages.length}`,
                    selectedEntry ? `Model: ${selectedEntry.model || "unknown"} · ${formatDuration(selectedEntry.durationMs)} · ${selectedEntry.traceLabel || "unlabeled"}` : null,
                    selectedEntry ? `Question: ${trimText(selectedEntry.questionPreview, 180)}` : null,
                    finalAnswer ? `Final Answer: ${trimText(finalAnswer, 180)}` : null,
                ].filter(Boolean).join("\n")
            )
        );

    const content = new ContainerBuilder()
        .setAccentColor(normalizedState.activeView === "json" ? 0xfee75c : 0x57f287)
        .addTextDisplayComponents(
            new TextDisplayBuilder().setContent(`### ${VIEW_LABELS[normalizedState.activeView]}`),
            new TextDisplayBuilder().setContent(currentContent)
        );

    const fileRow = new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
        new StringSelectMenuBuilder()
            .setCustomId(DEBUG_LOGS_FILE_SELECT_ID)
            .setPlaceholder("Choose a log file")
            .addOptions(
                files.map((file) => ({
                    label: file.replace("model-output-", "").replace(".jsonl", ""),
                    value: file,
                    default: file === normalizedState.selectedFile,
                }))
            )
    );

    const entryRow = new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
        new StringSelectMenuBuilder()
            .setCustomId(DEBUG_LOGS_ENTRY_SELECT_ID)
            .setPlaceholder(entries.length ? "Choose a log entry" : "No entries in this file")
            .setDisabled(!entryOptions.length)
            .addOptions(
                entryOptions.length
                    ? entryOptions.map((entry, offset) => {
                          const absoluteIndex = entryStart + offset;
                          return {
                              label: `#${absoluteIndex + 1} ${new Date(entry.timestamp).toLocaleTimeString("en-GB")} · ${trimText(entry.traceLabel || entry.callKind, 45)}`.slice(0, 100),
                              value: String(absoluteIndex),
                              description: `${entry.callKind || "unknown"} · ${formatDuration(entry.durationMs)} · ${trimText(entry.questionPreview, 60)}`.slice(0, 100),
                              default: absoluteIndex === normalizedState.selectedEntryIndex,
                          };
                      })
                    : [
                          {
                              label: "No entries",
                              value: "0",
                              description: "This file has no valid JSONL entries.",
                              default: true,
                          },
                      ]
            )
    );

    const viewRow = new ActionRowBuilder<StringSelectMenuBuilder>().addComponents(
        new StringSelectMenuBuilder()
            .setCustomId(DEBUG_LOGS_VIEW_SELECT_ID)
            .setPlaceholder("Choose what to inspect")
            .addOptions(
                (Object.keys(VIEW_LABELS) as DebugLogsView[]).map((view) => ({
                    label: VIEW_LABELS[view],
                    value: view,
                    default: view === normalizedState.activeView,
                }))
            )
    );

    const navRow = new ActionRowBuilder<ButtonBuilder>().addComponents(
        new ButtonBuilder()
            .setCustomId(DEBUG_LOGS_ENTRY_PAGE_PREV_ID)
            .setLabel("Newer Logs")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(normalizedState.entryListPage === 0),
        new ButtonBuilder()
            .setCustomId(DEBUG_LOGS_ENTRY_PAGE_NEXT_ID)
            .setLabel("Older Logs")
            .setStyle(ButtonStyle.Secondary)
            .setDisabled(normalizedState.entryListPage >= entryPageCount - 1),
        new ButtonBuilder()
            .setCustomId(DEBUG_LOGS_CONTENT_PAGE_PREV_ID)
            .setLabel("Prev Page")
            .setStyle(ButtonStyle.Primary)
            .setDisabled(normalizedState.contentPage === 0),
        new ButtonBuilder()
            .setCustomId(DEBUG_LOGS_CONTENT_PAGE_NEXT_ID)
            .setLabel("Next Page")
            .setStyle(ButtonStyle.Primary)
            .setDisabled(normalizedState.contentPage >= contentPages.length - 1)
    );

    return {
        components: [header, content, fileRow, entryRow, viewRow, navRow],
        flags: MessageFlags.IsComponentsV2 as const,
    };
}

export {
    DEBUG_LOGS_CONTENT_PAGE_NEXT_ID,
    DEBUG_LOGS_CONTENT_PAGE_PREV_ID,
    DEBUG_LOGS_ENTRY_PAGE_NEXT_ID,
    DEBUG_LOGS_ENTRY_PAGE_PREV_ID,
    DEBUG_LOGS_ENTRY_SELECT_ID,
    DEBUG_LOGS_FILE_SELECT_ID,
    DEBUG_LOGS_VIEW_SELECT_ID,
};