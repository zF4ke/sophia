const FILLER_WORDS = new Set([
    "ai",
    "ae",
    "blz",
    "beleza",
    "eae",
    "hey",
    "oi",
    "ola",
    "olá",
    "pf",
    "pls",
    "please",
]);

export function createQuestionFingerprint(question: string): string {
    const normalized = question
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/<#\d+>/g, " channel ")
        .replace(/<@!?\d+>/g, " user ")
        .replace(/https?:\/\/\S+/g, " ")
        .replace(/[^\p{L}\p{N}\s]/gu, " ")
        .split(/\s+/)
        .map((token) => token.trim())
        .filter((token) => token && !FILLER_WORDS.has(token));

    return normalized.join(" ").trim();
}
