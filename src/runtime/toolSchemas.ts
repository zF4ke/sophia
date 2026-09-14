import { buildToolDefinitions, buildVisibleToolDefinitions, type NativeToolDef } from "@/tools/registry";

export type { NativeToolDef };

const finishParams = {
    type: "object",
    properties: {
        notify: { type: "boolean", description: "For conditional scheduled checks only: false when the checked condition has no reportable change. Keep the findings in answer. Errors, uncertainty and required user action must be reported. Ignored in ordinary conversation." },
        answer: {
            type: "string",
            description: "Your final answer to the user. This will be sent as Sophia's response in Discord.",
        },
    },
    required: ["answer"],
} as const;

export const TOOL_DEFINITIONS: NativeToolDef[] = [
    ...buildToolDefinitions(),
    {
        type: "function",
        function: {
            name: "finish",
            description:
                "Call this when you have your final answer ready. The 'answer' field will be sent as Sophia's response in Discord. You MUST call this tool to deliver your response — do not just output text.",
            parameters: finishParams,
        },
    },
];

export function getToolDefinitions(discovered: Set<string> = new Set()): NativeToolDef[] {
    return [
        ...buildVisibleToolDefinitions(discovered),
        {
            type: "function",
            function: {
                name: "finish",
                description:
                    "Call this when you have your final answer ready. The 'answer' field will be sent as Sophia's response in Discord. You MUST call this tool to deliver your response — do not just output text.",
                parameters: finishParams,
            },
        },
    ];
}
