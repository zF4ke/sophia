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
    dayMonthDate:
        /\b(0?[1-9]|[12]\d|3[01])\s*(?:de\s+)?(jan(?:eiro)?|fev(?:ereiro)?|mar(?:co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?|january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)(?:\s*(?:de\s*)?(20\d{2}))?\b/i,
    beforeYesterday: [
        /before yesterday/i,
        /before ontem/i,
    ],
};

const MONTH_ALIASES: Record<string, number> = {
    jan: 0,
    janeiro: 0,
    january: 0,
    fev: 1,
    fevereiro: 1,
    feb: 1,
    february: 1,
    mar: 2,
    marco: 2,
    march: 2,
    abr: 3,
    abril: 3,
    apr: 3,
    april: 3,
    mai: 4,
    maio: 4,
    may: 4,
    jun: 5,
    junho: 5,
    june: 5,
    jul: 6,
    julho: 6,
    july: 6,
    ago: 7,
    agosto: 7,
    aug: 7,
    august: 7,
    set: 8,
    setembro: 8,
    sep: 8,
    september: 8,
    out: 9,
    outubro: 9,
    oct: 9,
    october: 9,
    nov: 10,
    novembro: 10,
    november: 10,
    dez: 11,
    dezembro: 11,
    dec: 11,
    december: 11,
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
    const nowMs = now.getTime();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const startOfYesterday = startOfToday - 24 * 60 * 60 * 1000;

    const explicitDateMatch = compact.match(TIME_PATTERNS.explicitDate);
    const explicitTimestamp = explicitDateMatch
        ? Date.parse(`${explicitDateMatch[1]}T00:00:00`)
        : NaN;
    const dayMonthDateMatch = compact.match(TIME_PATTERNS.dayMonthDate);
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
        return {
            afterTimestamp: explicitTimestamp,
            beforeTimestamp: explicitTimestamp + 24 * 60 * 60 * 1000,
        };
    }

    if (dayMonthDateMatch) {
        const day = Number(dayMonthDateMatch[1]);
        const rawMonth = normalize(dayMonthDateMatch[2] || "").replace(/\.$/, "");
        const month = MONTH_ALIASES[rawMonth];
        if (Number.isFinite(day) && day >= 1 && day <= 31 && month != null) {
            const explicitYear = dayMonthDateMatch[3] ? Number(dayMonthDateMatch[3]) : null;
            const inferredYear =
                explicitYear ??
                (() => {
                    const thisYearStart = new Date(now.getFullYear(), month, day).getTime();
                    // If date hasn't happened yet this year, assume previous year.
                    return thisYearStart > nowMs ? now.getFullYear() - 1 : now.getFullYear();
                })();

            const start = new Date(inferredYear, month, day).getTime();
            const end = start + 24 * 60 * 60 * 1000;

            if (hasUpperBound) {
                return { beforeTimestamp: end };
            }
            if (matchesAny(compact, TIME_PATTERNS.lowerBound)) {
                return { afterTimestamp: start };
            }
            return {
                afterTimestamp: start,
                beforeTimestamp: end,
            };
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
