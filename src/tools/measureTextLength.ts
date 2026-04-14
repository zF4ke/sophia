import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";

const parameters = {
    type: "object",
    properties: {
        text: {
            type: "string",
            description: "The text to measure.",
        },
    },
    required: ["text"],
} as const;

export const measureTextLengthTool: ToolDefinition = {
    name: T.measure_text_length,

    catalog: {
        effect: "read",
        description: "Measure the length of a text string in characters, words, and lines.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Measure the length of a text string. Returns character count, word count, and line count. Useful when the user asks about text size or length.",
        parameters,
    },

    capability: {
        description: "Measure the length of a text string in characters, words, and lines.",
        inputSchema: z.object({
            text: z.string().describe("The text to measure."),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: ["returns text length metrics"],
        async run(_context, args) {
            const text = String(args.text || "");
            const characters = text.length;
            const words = text.trim() ? text.trim().split(/\s+/).length : 0;
            const lines = text.split(/\r?\n/).length;
            return {
                tool: T.measure_text_length,
                summary: `${characters} characters, ${words} words, ${lines} lines.`,
                data: { characters, words, lines },
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "📏", labelPt: "Medir texto" },
};
