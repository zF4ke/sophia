import type { RequestGoalRecord } from "@/memory/DiscordMemoryService";
import { ModelGateway } from "@/ai/ModelGateway";

/**
 * Renders the per-leg auto-continue status line through one tiny model call
 * instead of string assembly. Input is only the open goals, so context cost
 * is minimal (goals, not notes, not history). Every output rule is defensive:
 * single line, hard cap, word-boundary trim — so even a degenerate model
 * output cannot reproduce the duplicated/cut-off symptom.
 */

const MAX_GOAL_INPUT_CHARS = 480;
const MAX_OUTPUT_CHARS = 220;

export function formatOpenGoalsForStatus(openGoals: RequestGoalRecord[]): string {
    return openGoals
        .map((goal) => `#${goal.seq} [${goal.status}] ${goal.body.slice(0, MAX_GOAL_INPUT_CHARS)}`)
        .join("\n")
        .slice(0, openGoals.length * (MAX_GOAL_INPUT_CHARS + 24));
}

export function trimToWordBoundary(text: string, maxChars: number): string {
    const trimmed = text.trim();
    if (trimmed.length <= maxChars) return trimmed;
    const cut = trimmed.slice(0, maxChars);
    const lastSpace = cut.lastIndexOf(" ");
    // If there is no space at all (one giant token), fall back to the hard
    // cut rather than returning an empty message.
    return (lastSpace > 0 ? cut.slice(0, lastSpace) : cut).trim();
}

export async function renderStatusLine(
    openGoals: RequestGoalRecord[],
    _legNumber: number,
): Promise<string> {
    const fallback = buildDeterministicFallback(openGoals);
    if (openGoals.length === 0) return fallback;

    try {
        const output = await ModelGateway.generateText(
            [
                {
                    role: "system",
                    content:
                        "Write ONE short human progress line in the user's language (the goals are in Portuguese unless shown otherwise). " +
                        "Sound like a person giving a quick update, e.g. 'ok, já fiz X, tô em Y' or 'ainda na psico-análise, já com 30k mensagens'. " +
                        "Rules: single line, max 200 characters, merge goals that describe the same task into one mention, " +
                        "never repeat the same task twice, never end mid-word, no emojis, no markdown, " +
                        "no questions, no preamble, no 'A continuar (parte X):' framing. Output only the line.",
                },
                { role: "user", content: `Open goals:\n${formatOpenGoalsForStatus(openGoals)}` },
            ],
            {
                maxOutputTokens: 120,
                temperature: 0.4,
                traceContext: { traceLabel: "auto_continue_status", questionPreview: `status` },
            },
        );
        const singleLine = output.split("\n")[0]?.trim() ?? "";
        if (!singleLine) return fallback;
        return trimToWordBoundary(singleLine, MAX_OUTPUT_CHARS);
    } catch {
        return fallback;
    }
}

function buildDeterministicFallback(openGoals: RequestGoalRecord[]): string {
    if (openGoals.length === 0) return "";
    // Jaccard-based twin merge: two goals that share most words describe the
    // same task (e.g. two phrasings of the same psicoanálise). Collapse them
    // so the fallback never prints the same task twice.
    const wordSet = (body: string) =>
        new Set(
            body
                .toLowerCase()
                .normalize("NFD")
                .replace(/[\u0300-\u036f]/g, "")
                .replace(/[^a-z0-9]+/g, " ")
                .split(" ")
                .filter((word) => word.length > 2),
        );
    const jaccard = (a: Set<string>, b: Set<string>) => {
        let intersection = 0;
        for (const word of a) if (b.has(word)) intersection++;
        const union = a.size + b.size - intersection;
        return union === 0 ? 0 : intersection / union;
    };
    const unique: RequestGoalRecord[] = [];
    for (const goal of openGoals) {
        const candidate = wordSet(goal.body);
        const isTwin = unique.some((kept) => jaccard(candidate, wordSet(kept.body)) > 0.55);
        if (!isTwin) unique.push(goal);
    }
    // Conversational fallback, no "A continuar (parte X):" framing. Just a short
    // human line that still names the task once.
    const main = trimToWordBoundary(unique[0].body, 160);
    return trimToWordBoundary(`tô em: ${main}`, MAX_OUTPUT_CHARS);
}
