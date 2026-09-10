import vm from "node:vm";

/**
 * Sandboxed per-card scripts. The model writes small JavaScript handlers at
 * send/edit time; they execute inside a bare `node:vm` context (no require,
 * no process, no network) with a bounded API surface and a hard timeout.
 *
 * Sandbox contract (exactly what the handler code may touch):
 * - `state`   — the card's persisted game state object (mutations are saved)
 * - `user`    — { id, username } of who clicked
 * - `values`  — selected values (string/user/role/mentionable/channel selects)
 * - `customId`, `cardId`
 * - `reply(text)`            — ephemeral reply to the clicker (last call wins, ≤1500 chars)
 * - `send(channelId, text)`  — deliver a message elsewhere (max 3 calls)
 * - `setTitle(t)`, `setSummary(t)`, `setSection(n)`, `setAccent(color)`, `setSpoiler(bool)`
 * - `log(text)`              — appended to the reply, useful while debugging
 * - return an object → merged over the state patches (equivalent to helpers)
 */

export interface ArtifactScriptInput {
    code: string;
    state: Record<string, unknown>;
    user: { id: string; username: string };
    values: string[];
    customId: string;
    cardId: string;
}

export interface ArtifactScriptResult {
    state: Record<string, unknown>;
    reply: string | null;
    sends: Array<{ channelId: string; content: string }>;
    title?: string;
    summary?: string;
    section?: number;
    accentColor?: number;
    spoiler?: boolean;
    logs: string[];
    error: string | null;
}

export const SCRIPT_LIMITS = {
    maxCodeChars: 4000,
    maxHandlers: 10,
    timeoutMs: 100,
    maxSends: 3,
    maxReplyChars: 1500,
    maxStateChars: 4000,
    maxLogs: 5,
} as const;

const SCRIPT_API_DOC = [
    "Available in handlers:",
    "- state (object, persisted; mutate freely or return a new object)",
    "- user: {id, username}; values: string[] (select choices); customId; cardId",
    "- reply(text) - ephemeral reply to the clicker (last call wins, max 1500 chars)",
    "- send(channelId, text) - deliver a message to a channel (max 3 per click)",
    "- setTitle(t) / setSummary(t) / setSection(n) / setAccent(color) / setSpoiler(bool)",
    "- log(text) - debug lines appended to the reply",
].join("; ");

export function describeScriptApi(): string {
    return SCRIPT_API_DOC;
}

export function runArtifactScript(input: ArtifactScriptInput): ArtifactScriptResult {
    const { code, state, user, values, customId, cardId } = input;

    if (code.length > SCRIPT_LIMITS.maxCodeChars) {
        return { state, reply: null, sends: [], title: undefined, accentColor: undefined, spoiler: undefined, logs: [], error: `handler exceeds ${SCRIPT_LIMITS.maxCodeChars} chars` };
    }

    const workingState: Record<string, unknown> = structuredClone(state);
    const sends: Array<{ channelId: string; content: string }> = [];
    const logs: string[] = [];
    let reply: string | null = null;
    let title: string | undefined;
    let summary: string | undefined;
    let section: number | undefined;
    let accentColor: number | undefined;
    let spoiler: boolean | undefined;

    const clamp = (text: string) => String(text ?? "").slice(0, SCRIPT_LIMITS.maxReplyChars);

    const sandbox: Record<string, unknown> = {
        state: workingState,
        user,
        values,
        customId,
        cardId,
        reply: (text: unknown) => { reply = clamp(String(text ?? "")); },
        send: (channelId: unknown, content: unknown) => {
            if (sends.length >= SCRIPT_LIMITS.maxSends) return;
            const cid = String(channelId ?? "").trim();
            const body = clamp(String(content ?? ""));
            if (/^\d{5,25}$/.test(cid) && body) sends.push({ channelId: cid, content: body });
        },
        setTitle: (value: unknown) => { title = String(value ?? "").slice(0, 120) || undefined; },
        setSummary: (value: unknown) => { summary = String(value ?? "").slice(0, 400) || undefined; },
        setSection: (value: unknown) => {
            const n = Number(value);
            section = Number.isFinite(n) && n >= 0 ? Math.floor(n) : undefined;
        },
        setAccent: (value: unknown) => {
            const n = Number(value);
            accentColor = Number.isInteger(n) && n >= 0 && n <= 0xffffff ? n : undefined;
        },
        setSpoiler: (value: unknown) => { spoiler = Boolean(value); },
        log: (text: unknown) => {
            if (logs.length < SCRIPT_LIMITS.maxLogs) logs.push(clamp(String(text ?? "")).slice(0, 200));
        },
    };

    const wrapped = `"use strict";\n(() => {\n${code}\n})();`;

    try {
        vm.runInNewContext(wrapped, sandbox, { timeout: SCRIPT_LIMITS.timeoutMs });
    } catch (error) {
        return {
            state,
            reply: null,
            sends: [],
            title: undefined,
            accentColor: undefined,
            spoiler: undefined,
            logs,
            error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
        };
    }

    // The script may have replaced `state` wholesale or mutated it; honor both.
    const finalState = (sandbox.state && typeof sandbox.state === "object" && !Array.isArray(sandbox.state))
        ? sandbox.state as Record<string, unknown>
        : workingState;
    const serialized = JSON.stringify(finalState);
    const stateOut = serialized && serialized.length <= SCRIPT_LIMITS.maxStateChars ? finalState : state;

    return {
        state: stateOut,
        reply: reply ?? (logs.length ? logs.join("\n") : null),
        sends: sends.slice(0, SCRIPT_LIMITS.maxSends),
        title,
        summary,
        section,
        accentColor,
        spoiler,
        logs,
        error: null,
    };
}
