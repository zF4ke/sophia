import { describe, expect, it } from "vitest";
import {
    buildLogEntryViewText,
    extractToolCallRecords,
    type DebugLogTraceEntry,
} from "@/discord/commands/system/debugLogsPanel";

const sampleEntry: DebugLogTraceEntry = {
    timestamp: "2026-04-14T02:50:16.507Z",
    callKind: "tool_chat",
    model: "google/gemini-3.1-flash-lite-preview",
    traceLabel: "agent_loop_iter_3",
    questionPreview: "who posted the youtube link?",
    durationMs: 1421,
    rawOutput: "",
    normalizedOutput: "",
    finishReason: "tool_calls",
    messages: [
        { role: "system", content: "You are Sophia." },
        { role: "user", content: "who posted the youtube link?" },
        {
            role: "assistant",
            content: null,
            tool_calls: [
                {
                    id: "tool_1",
                    type: "function",
                    function: {
                        name: "resolve_channel_targets",
                        arguments: "{\"targetText\":\"comandos\"}",
                    },
                },
            ],
        },
        {
            role: "tool",
            tool_call_id: "tool_1",
            content: "{\"summary\":\"Resolved channel\",\"data\":{\"resolvedIds\":[\"123\"]}}",
        },
        {
            role: "assistant",
            content: null,
            tool_calls: [
                {
                    id: "tool_2",
                    type: "function",
                    function: {
                        name: "finish",
                        arguments: "{\"answer\":\"Nao encontrei referencia ao autor da musica.\"}",
                    },
                },
            ],
        },
        {
            role: "tool",
            tool_call_id: "tool_2",
            content: "{\"ok\":true}",
        },
    ],
};

const pendingToolEntry: DebugLogTraceEntry = {
    timestamp: "2026-04-14T03:30:56.912Z",
    callKind: "tool_chat",
    model: "google/gemini-3.1-flash-lite-preview",
    traceLabel: "agent_loop_iter_1",
    questionPreview: "who posted the youtube link?",
    durationMs: 1185,
    rawOutput: "",
    normalizedOutput: "",
    finishReason: "tool_calls",
    messages: [
        { role: "system", content: "You are Sophia." },
        { role: "user", content: "who posted the youtube link?" },
    ],
    toolCalls: [
        {
            id: "tool_retrieve_messages_1",
            type: "function",
            function: {
                name: "retrieve_messages",
                arguments: "{\"query\":\"youtube\"}",
            },
        },
    ],
};

const laterToolOutputEntry: DebugLogTraceEntry = {
    timestamp: "2026-04-14T03:31:12.511Z",
    callKind: "text",
    model: "google/gemini-3.1-flash-lite-preview",
    traceLabel: "synthesis_fallback",
    questionPreview: "who posted the youtube link?",
    durationMs: 1676,
    rawOutput: "",
    normalizedOutput: "",
    messages: [
        { role: "system", content: "You are Sophia." },
        { role: "user", content: "who posted the youtube link?" },
        {
            role: "tool",
            tool_call_id: "tool_retrieve_messages_1",
            content: "{\"summary\":\"No relevant messages found even after refreshing Discord history.\"}",
        },
    ],
};

const laterTraceOnlyEntry: DebugLogTraceEntry = {
    timestamp: "2026-04-14T03:31:12.511Z",
    callKind: "text",
    model: "google/gemini-3.1-flash-lite-preview",
    traceLabel: "synthesis_fallback",
    questionPreview: "who posted the youtube link?",
    durationMs: 1676,
    rawOutput: "",
    normalizedOutput: "",
    messages: [
        { role: "system", content: "You are Sophia." },
        { role: "user", content: "who posted the youtube link?" },
    ],
    traceEvents: [
        {
            label: "tool_result",
            detail: "retrieve_messages: No relevant messages found even after refreshing Discord history.",
            timestamp: 1713065472511,
        },
    ],
};

describe("debug logs panel helpers", () => {
    it("extracts tool calls with parsed arguments and outputs", () => {
        const records = extractToolCallRecords(sampleEntry);

        expect(records).toHaveLength(2);
        expect(records[0].name).toBe("resolve_channel_targets");
        expect(records[0].argumentsText).toContain('"targetText": "comandos"');
        expect(records[0].outputText).toContain('"summary": "Resolved channel"');
        expect(records[1].name).toBe("finish");
        expect(records[1].argumentsText).toContain("Nao encontrei referencia ao autor da musica.");
    });

    it("resolves pending tool outputs from later entries in the same log", () => {
        const entries = [laterToolOutputEntry, pendingToolEntry];
        const records = extractToolCallRecords(pendingToolEntry, entries, 1);
        const toolsText = buildLogEntryViewText(pendingToolEntry, "tools", 1, entries.length, entries);

        expect(records).toHaveLength(1);
        expect(records[0].name).toBe("retrieve_messages");
        expect(records[0].outputText).toContain("No relevant messages found even after refreshing Discord history.");
        expect(toolsText).toContain("No relevant messages found even after refreshing Discord history.");
    });

    it("falls back to later trace summaries when no tool message was logged", () => {
        const entries = [laterTraceOnlyEntry, pendingToolEntry];
        const records = extractToolCallRecords(pendingToolEntry, entries, 1);
        const toolsText = buildLogEntryViewText(pendingToolEntry, "tools", 1, entries.length, entries);

        expect(records).toHaveLength(1);
        expect(records[0].outputText).toContain(
            "retrieve_messages: No relevant messages found even after refreshing Discord history."
        );
        expect(toolsText).toContain(
            "retrieve_messages: No relevant messages found even after refreshing Discord history."
        );
    });

    it("renders dedicated tool and prompt views with full recorded content", () => {
        const promptText = buildLogEntryViewText(sampleEntry, "prompt");
        const toolsText = buildLogEntryViewText(sampleEntry, "tools");
        const jsonText = buildLogEntryViewText(sampleEntry, "json");

        expect(promptText).toContain("You are Sophia.");
        expect(toolsText).toContain("Tool 1: resolve_channel_targets");
        expect(toolsText).toContain("Output:");
        expect(toolsText).toContain('"resolvedIds": [');
        expect(jsonText).toContain("agent_loop_iter_3");
    });
});