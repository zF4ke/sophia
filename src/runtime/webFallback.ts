import type { GroundedAnswerMode, RequestClassification, WebMode } from "@/shared/appTypes";
import type { TurnTrigger } from "@/runtime/contracts";

type ConversationWebDecision = {
    webMode: WebMode;
    reason: string;
};

const EXTERNAL_WEB_HINTS = [
    "today",
    "hoje",
    "current",
    "atual",
    "latest",
    "ultimo",
    "ultima",
    "recent",
    "recente",
    "news",
    "noticia",
    "noticias",
    "release",
    "lancamento",
    "update",
    "updates",
    "version",
    "versao",
    "price",
    "preco",
    "weather",
    "forecast",
    "docs",
    "documentation",
    "api",
    "website",
    "site oficial",
    "internet",
    "web",
    "google",
    "pesquisa",
    "pesquise",
    "pesquisar",
    "search the web",
    "official",
    "oficial",
];

const DISCORD_LOCAL_HINTS = [
    "discord",
    "server",
    "servidor",
    "channel",
    "canal",
    "guild",
    "member",
    "membro",
    "role",
    "cargo",
    "message",
    "mensagem",
    "chat",
    "thread",
    "nesse servidor",
    "neste servidor",
    "this server",
];

function normalize(text: string): string {
    return text
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .replace(/\s+/g, " ")
        .trim();
}

function containsAny(normalizedText: string, hints: string[]): boolean {
    return hints.some((hint) => normalizedText.includes(hint));
}

function looksClearlyExternal(question: string): boolean {
    const normalized = normalize(question);
    if (/https?:\/\//.test(question) || /\b[a-z0-9-]+\.[a-z]{2,}\b/i.test(question)) {
        return true;
    }

    return containsAny(normalized, EXTERNAL_WEB_HINTS);
}

function looksPurelyDiscordLocal(question: string): boolean {
    return containsAny(normalize(question), DISCORD_LOCAL_HINTS);
}

export function decideConversationWebMode(options: {
    question: string;
    classification: RequestClassification;
    trigger?: TurnTrigger | null;
    conversationWebMode?: WebMode;
    groundedAnswerMode?: GroundedAnswerMode | null;
}): ConversationWebDecision {
    if (options.conversationWebMode !== "auto" || !options.trigger) {
        return {
            webMode: "off",
            reason: "Web disabled for this entrypoint.",
        };
    }

    const external = looksClearlyExternal(options.question);
    const localDiscord = looksPurelyDiscordLocal(options.question);

    if (options.classification.mode === "direct_answer") {
        return external
            ? {
                  webMode: "auto",
                  reason: "Conversational direct answer looks current or external.",
              }
            : {
                  webMode: "off",
                  reason: "Conversational direct answer does not need web search.",
              };
    }

    if (options.groundedAnswerMode && options.groundedAnswerMode !== "insufficient") {
        return {
            webMode: "off",
            reason: "Discord grounding already answered the request.",
        };
    }

    if (external) {
        return {
            webMode: "auto",
            reason: "Grounded conversational request still needs current or external context.",
        };
    }

    if (localDiscord) {
        return {
            webMode: "off",
            reason: "Remaining need appears Discord-specific.",
        };
    }

    return {
        webMode: "off",
        reason: "No strong external/current signal for web fallback.",
    };
}
