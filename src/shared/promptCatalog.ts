export const RUNTIME_PROMPT_IDS = [
    "system/base",
    "system/grounded",
    "tasks/classify_request",
    "tasks/plan_discord_search",
    "tasks/synthesize_answer",
    "guards/insufficient_evidence",
] as const;

export type RuntimePromptId = (typeof RUNTIME_PROMPT_IDS)[number];
