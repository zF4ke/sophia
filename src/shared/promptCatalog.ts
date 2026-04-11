export const RUNTIME_PROMPT_IDS = [
    "system/base",
    "system/personality",
    "runtime/plan_turn",
    "runtime/select_next_step",
    "runtime/judge_evidence",
    "runtime/synthesize_answer",
    "runtime/debug_summary",
    "guards/insufficient_evidence",
] as const;

export type RuntimePromptId = (typeof RUNTIME_PROMPT_IDS)[number];
