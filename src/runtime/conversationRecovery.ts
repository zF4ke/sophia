import type { ConversationTurnSummary, EvidenceItem, ReplyContext } from "@/runtime/contracts";
import type { GroundedAnswerMode } from "@/shared/appTypes";

function normalize(text: string): string {
    return text
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .trim();
}

function looksPortuguese(text: string): boolean {
    return /\b(nao|não|voce|você|entendi|mensagem|canal|servidor|quem|o que|oi|eae|tudo bem|como posso ajudar)\b/i.test(
        normalize(text)
    );
}

function formatPriorTurns(priorTurns: ConversationTurnSummary[]): string {
    if (!priorTurns.length) {
        return "";
    }

    const latest = priorTurns[0];
    const pt = looksPortuguese(latest.question) || looksPortuguese(latest.answer);
    return pt
        ? `Antes perguntaste "${latest.question}" e eu respondi "${latest.answer}".`
        : `Earlier you asked "${latest.question}" and I answered "${latest.answer}".`;
}

function formatEvidenceHint(evidence: EvidenceItem[], question: string): string {
    if (!evidence.length) {
        return "";
    }

    const item = evidence[0];
    const author = item.authorName || "someone";
    const pt = looksPortuguese(question);
    const channel = item.channelName
        ? pt ? ` no #${item.channelName}` : ` in #${item.channelName}`
        : "";
    return pt
        ? `${author}${channel} mencionou: \"${item.content}\".`
        : `${author}${channel} mentioned: \"${item.content}\".`;
}

export function buildDirectConversationFallback(question: string): string {
    const pt = looksPortuguese(question);
    return pt
        ? "Oi! Estou por aqui. Diz-me no que queres ajuda e, se precisar de contexto do Discord, eu vou procurar."
        : "Hey, I'm here. Tell me what you need and, if it depends on Discord context, I'll go look for it.";
}

export function buildConversationalRecovery(options: {
    question: string;
    confidence: GroundedAnswerMode;
    evidence: EvidenceItem[];
    replyContext?: ReplyContext | null;
    priorTurns?: ConversationTurnSummary[];
    stopReason?: string | null;
}): string {
    const pt = looksPortuguese(options.question);
    const prior = formatPriorTurns(options.priorTurns || []);
    const evidenceHint = formatEvidenceHint(options.evidence, options.question);
    const replyHint = options.replyContext?.content
        ? pt
            ? `Você está respondendo à mensagem: \"${options.replyContext.content}\".`
            : `You are replying to: \"${options.replyContext.content}\".`
        : "";

    const budgetHint =
        options.stopReason === "budget_exhausted"
            ? pt
                ? "Fiquei sem tempo de pesquisa antes de terminar de confirmar isso."
                : "I ran out of research time before I could finish confirming that."
            : "";

    if (options.evidence.length) {
        return pt
            ? [
                  evidenceHint || "So tenho uma parte do contexto por agora.",
                  prior,
                  replyHint,
                  options.confidence === "insufficient"
                      ? "Nao tenho a certeza toda, mas parece ser isso. Queres que eu procure mais?"
                      : "Essa e a minha melhor leitura. Queres que eu procure mais contexto?",
              ]
                  .filter(Boolean)
                  .join(" ")
            : [
                  evidenceHint || "I only have part of the context so far.",
                  prior,
                  replyHint,
                  options.confidence === "insufficient"
                      ? "I'm not fully sure, but that seems to be it. Want me to keep looking?"
                      : "That's my best read. Want me to look for more context?",
              ]
                  .filter(Boolean)
                  .join(" ");
    }

    return pt
        ? [
              prior,
              replyHint,
              budgetHint || "Ainda nao encontrei o que preciso para te responder bem.",
              "Podes dar-me mais detalhes? Tipo quem disse ou em que canal?",
          ]
              .filter(Boolean)
              .join(" ")
        : [
              prior,
              replyHint,
              budgetHint || "I haven't found what I need to give you a good answer yet.",
              "Can you give me more details? Like who said it or which channel?",
          ]
              .filter(Boolean)
              .join(" ");
}

export function sanitizeConversationalAnswer(answer: string | null | undefined): string {
    const normalized = (answer || "").trim();
    if (!normalized) {
        return "";
    }

    const banned = new Set([
        "I couldn't ground that in Discord evidence.",
        "I don't have enough Discord evidence to answer that yet.",
    ]);

    return banned.has(normalized) ? "" : normalized;
}
