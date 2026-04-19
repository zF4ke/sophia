export const ACTIVE_PROMPT_IDS = [
    "runtime/agent_loop",
    "runtime/stall_classifier",
    "system/personality_mixed_override",
    "system/personality_classic_override",
] as const;

export const RUNTIME_PROMPT_IDS = [
    "runtime/agent_loop",
    "runtime/stall_classifier",
] as const;

export const SYSTEM_OVERRIDE_PROMPT_IDS = [
    "system/personality_mixed_override",
    "system/personality_classic_override",
] as const;
