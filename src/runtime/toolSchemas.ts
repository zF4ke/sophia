import { buildToolDefinitions, type NativeToolDef } from "@/tools/registry";

export type { NativeToolDef };

const finishParams = {
    type: "object",
    properties: {
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
