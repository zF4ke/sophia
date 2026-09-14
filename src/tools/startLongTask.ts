import { z } from "zod";
import { T } from "@/shared/discordTools";
import type { ToolDefinition } from "./types";

/**
 * Control tool: `start_long_task`.
 *
 * The Runtime intercepts this tool *by name* before dispatch (similar to
 * how `finish` is intercepted), so the capability's `run()` never actually
 * executes. It exists here purely so:
 *   - the tool schema ships to every model provider,
 *   - the capability registry treats it as a known tool, and
 *   - the tool catalog consistency tests stay green.
 *
 * The runtime prepares a larger evidence context. It does not change execution limits.
 */

const parameters = {
    type: "object",
    properties: {
        reason: {
            type: "string",
            description:
                "Short human-readable justification. Shown to the user in the progress status line and written to runtime trace. Example: 'paginar 2000 mensagens do @one_person em 10 canais'.",
        },
    },
    required: ["reason"],
} as const;

export const startLongTaskTool: ToolDefinition = {
    name: T.start_long_task,

    catalog: {
        // Stays "read" so it never enters the write/destructive tiers
        // or triggers the approval gate.
        effect: "read",
        description:
            "Prepare a larger evidence context for sustained research. No budget increase is required to continue working.",
        evidenceRole: "discovery_only",
    },

    schema: {
        description:
            "Optionally declare sustained research to retain a larger evidence slice. Ordinary tasks already have no default tool-call cap. This does not change permissions or an explicit operator limit. Only one declaration per turn is needed.",
        parameters,
    },

    capability: {
        description:
            "Inert runtime-control capability. Runtime intercepts this tool name before dispatch to prepare evidence context; this run() is never actually invoked.",
        inputSchema: z.object({
            reason: z.string(),
        }),
        outputSchema: z.any(),
        sideEffectLevel: "none",
        authRequirements: [],
        costClass: "cheap",
        latencyClass: "fast",
        preconditions: [],
        postconditions: [
            "per-turn maxEvidenceSlice is raised to settings.runtime.longTask.evidenceSliceFloor",
        ],
        async run() {
            // Never actually called — Runtime intercepts the tool name.
            return {
                tool: T.start_long_task,
                summary: "Long-task mode acknowledged.",
                data: {},
            };
        },
    },

    strategy: {
        extractEvidence() {
            return [];
        },
    },

    display: { icon: "⏳", labelPt: "Modo long-task" },
};
