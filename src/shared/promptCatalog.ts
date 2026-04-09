export const RUNTIME_PROMPT_IDS = [
    "system/base",
    "system/grounded",
    "tasks/classify_request",
    "tasks/retrieval_controller",
    "tasks/route_discord_intent",
    "tasks/plan_discord_search",
    "tasks/judge_grounding_sufficiency",
    "tasks/synthesize_answer",
    "guards/insufficient_evidence",
] as const;

export type RuntimePromptId = (typeof RUNTIME_PROMPT_IDS)[number];
