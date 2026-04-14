export const RUNTIME_PROMPT_IDS = [
    "runtime/agent_loop",
] as const;

export type RuntimePromptId = (typeof RUNTIME_PROMPT_IDS)[number];
