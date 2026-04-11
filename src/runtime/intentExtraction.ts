import type { ActiveRetrievalSession, TurnIntent } from "@/runtime/contracts";
import type { RetrievalMode } from "@/shared/appTypes";

// ---------------------------------------------------------------------------
// Text normalization (shared with planning.ts)
// ---------------------------------------------------------------------------

export function normalize(text: string): string {
    return text
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .trim();
}

// ---------------------------------------------------------------------------
// Lexicon — add patterns per semantic group, not per language
// ---------------------------------------------------------------------------

const CONTINUATION_PATTERNS: RegExp[] = [
    /^(continue|keep going|again|all of them|until\b)/i,
    /^(de novo|tenta de novo|mais|continua|continue lendo|ate ontem|até ontem)/i,
];

const SEMANTIC_MODE_PATTERNS: RegExp[] = [
    /\b(find|search|reference|mentions?)\b/i,
    /\b(mencoes|menções|citou|cita|referencias|referências)\b/i,
];

const MIXED_MODE_PATTERNS: RegExp[] = [
    /\b(all|entire|whole|history)\b/i,
    /\b(todos|todas|historico|histórico)\b/i,
];

const TIME_PATTERNS = {
    lastWeek: [
        /\b(last week)\b/i,
        /\b(ultima semana|última semana)\b/i,
    ],
    today: [
        /\b(today)\b/i,
        /\b(hoje)\b/i,
    ],
    yesterday: [
        /\b(yesterday)\b/i,
        /\b(ontem)\b/i,
    ],
    upperBound: [
        /\b(before|until)\b/i,
        /\b(ate|até)\b/i,
    ],
    lowerBound: [
        /\b(after|since)\b/i,
        /\b(depois|desde)\b/i,
    ],
    explicitDate: /\b(20\d{2}-\d{2}-\d{2})\b/,
    beforeYesterday: [
        /before yesterday/i,
        /before ontem/i,
    ],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function matchesAny(text: string, patterns: RegExp[]): boolean {
    return patterns.some((p) => p.test(text));
}

type IntentSource = TurnIntent["source"];

// ---------------------------------------------------------------------------
// Deterministic extraction
// ---------------------------------------------------------------------------

export function extractDeterministicIntent(
    question: string,
    activeRetrievalSession?: ActiveRetrievalSession | null
): Partial<TurnIntent> {
    const compact = normalize(question);
    const result: Partial<TurnIntent> = {};
    const source: Partial<IntentSource> = {};

    // --- Continuation ---
    if (matchesAny(compact, CONTINUATION_PATTERNS)) {
        result.continuation = true;
        source.continuation = "deterministic";
    }

    // --- Retrieval mode ---
    if (matchesAny(compact, SEMANTIC_MODE_PATTERNS)) {
        result.retrievalMode = "semantic";
        source.retrievalMode = "deterministic";
    } else if (matchesAny(compact, MIXED_MODE_PATTERNS)) {
        result.retrievalMode = "mixed";
        source.retrievalMode = "deterministic";
    } else if (activeRetrievalSession?.mode) {
        result.retrievalMode = activeRetrievalSession.mode;
        source.retrievalMode = "session";
    }

    // --- Time bounds ---
    const timeBounds = extractDeterministicTimeBounds(compact);
    if (timeBounds.beforeTimestamp !== undefined || timeBounds.afterTimestamp !== undefined) {
        if (timeBounds.beforeTimestamp !== undefined) {
            result.beforeTimestamp = timeBounds.beforeTimestamp;
        }
        if (timeBounds.afterTimestamp !== undefined) {
            result.afterTimestamp = timeBounds.afterTimestamp;
        }
        source.timeBounds = "deterministic";
    }

    if (Object.keys(source).length) {
        result.source = source as IntentSource;
    }

    return result;
}

function extractDeterministicTimeBounds(compact: string): {
    beforeTimestamp?: number;
    afterTimestamp?: number;
} {
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const startOfYesterday = startOfToday - 24 * 60 * 60 * 1000;

    const explicitDateMatch = compact.match(TIME_PATTERNS.explicitDate);
    const explicitTimestamp = explicitDateMatch
        ? Date.parse(`${explicitDateMatch[1]}T00:00:00`)
        : NaN;
    const hasUpperBound = matchesAny(compact, TIME_PATTERNS.upperBound);

    if (matchesAny(compact, TIME_PATTERNS.lastWeek)) {
        return {
            afterTimestamp: startOfToday - 7 * 24 * 60 * 60 * 1000,
            beforeTimestamp: startOfToday,
        };
    }

    if (matchesAny(compact, TIME_PATTERNS.today)) {
        return { afterTimestamp: startOfToday };
    }

    if (matchesAny(compact, TIME_PATTERNS.yesterday)) {
        if (hasUpperBound) {
            return {
                beforeTimestamp: matchesAny(compact, TIME_PATTERNS.beforeYesterday)
                    ? startOfYesterday
                    : startOfToday,
            };
        }
        return {
            afterTimestamp: startOfYesterday,
            beforeTimestamp: startOfToday,
        };
    }

    if (!Number.isNaN(explicitTimestamp)) {
        if (hasUpperBound) {
            return { beforeTimestamp: explicitTimestamp + 24 * 60 * 60 * 1000 };
        }
        if (matchesAny(compact, TIME_PATTERNS.lowerBound)) {
            return { afterTimestamp: explicitTimestamp };
        }
    }

    return {};
}

// ---------------------------------------------------------------------------
// Model intent parsing
// ---------------------------------------------------------------------------

interface ModelIntentBlock {
    continuation?: boolean | null;
    retrievalMode?: RetrievalMode | null;
    beforeDate?: string | null;
    afterDate?: string | null;
}

export function parseModelIntent(raw: unknown): Partial<TurnIntent> {
    if (!raw || typeof raw !== "object") {
        return {};
    }

    const intent = (raw as Record<string, unknown>).intent as ModelIntentBlock | undefined;
    if (!intent || typeof intent !== "object") {
        return {};
    }

    const result: Partial<TurnIntent> = {};
    const source: Partial<IntentSource> = {};

    if (typeof intent.continuation === "boolean") {
        result.continuation = intent.continuation;
        source.continuation = "model";
    }

    if (
        intent.retrievalMode === "history" ||
        intent.retrievalMode === "semantic" ||
        intent.retrievalMode === "mixed"
    ) {
        result.retrievalMode = intent.retrievalMode;
        source.retrievalMode = "model";
    }

    const beforeMs = parseIsoDateToTimestamp(intent.beforeDate, "before");
    const afterMs = parseIsoDateToTimestamp(intent.afterDate, "after");
    if (beforeMs !== null || afterMs !== null) {
        if (beforeMs !== null) {
            result.beforeTimestamp = beforeMs;
        }
        if (afterMs !== null) {
            result.afterTimestamp = afterMs;
        }
        source.timeBounds = "model";
    }

    if (Object.keys(source).length) {
        result.source = source as IntentSource;
    }

    return result;
}

function parseIsoDateToTimestamp(
    value: unknown,
    bound: "before" | "after"
): number | null {
    if (typeof value !== "string" || !value) {
        return null;
    }
    const ms = Date.parse(`${value}T00:00:00`);
    if (Number.isNaN(ms)) {
        return null;
    }

    // Keep model-sourced date semantics aligned with deterministic parsing.
    if (bound === "before") {
        return ms + 24 * 60 * 60 * 1000;
    }

    return Number.isNaN(ms) ? null : ms;
}

// ---------------------------------------------------------------------------
// Merge — deterministic wins per field
// ---------------------------------------------------------------------------

const DEFAULT_INTENT: TurnIntent = {
    continuation: false,
    retrievalMode: null,
    beforeTimestamp: null,
    afterTimestamp: null,
    source: {
        continuation: "deterministic",
        retrievalMode: "none",
        timeBounds: "none",
    },
};

export function mergeIntent(
    deterministic: Partial<TurnIntent>,
    model: Partial<TurnIntent>
): TurnIntent {
    const continuation =
        deterministic.continuation ?? model.continuation ?? DEFAULT_INTENT.continuation;
    const continuationSource =
        deterministic.source?.continuation ??
        model.source?.continuation ??
        DEFAULT_INTENT.source.continuation;

    const retrievalMode =
        deterministic.retrievalMode ?? model.retrievalMode ?? DEFAULT_INTENT.retrievalMode;
    const retrievalModeSource =
        deterministic.source?.retrievalMode ??
        model.source?.retrievalMode ??
        DEFAULT_INTENT.source.retrievalMode;

    const hasDeterministicBefore = deterministic.beforeTimestamp !== undefined;
    const hasDeterministicAfter = deterministic.afterTimestamp !== undefined;
    const hasModelBefore = model.beforeTimestamp !== undefined;
    const hasModelAfter = model.afterTimestamp !== undefined;

    const beforeTimestamp = hasDeterministicBefore
        ? (deterministic.beforeTimestamp ?? null)
        : hasModelBefore
          ? (model.beforeTimestamp ?? null)
          : null;
    const afterTimestamp = hasDeterministicAfter
        ? (deterministic.afterTimestamp ?? null)
        : hasModelAfter
          ? (model.afterTimestamp ?? null)
          : null;

    const usesDeterministicTime = hasDeterministicBefore || hasDeterministicAfter;
    const usesModelTime =
        (!hasDeterministicBefore && hasModelBefore) ||
        (!hasDeterministicAfter && hasModelAfter);
    const timeBoundsSource = usesDeterministicTime
        ? (deterministic.source?.timeBounds ?? "deterministic")
        : usesModelTime
          ? (model.source?.timeBounds ?? "model")
          : "none";

    return {
        continuation,
        retrievalMode,
        beforeTimestamp,
        afterTimestamp,
        source: {
            continuation: continuationSource,
            retrievalMode: retrievalModeSource,
            timeBounds: timeBoundsSource,
        },
    };
}
