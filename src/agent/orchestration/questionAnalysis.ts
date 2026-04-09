export function normalizeQuestion(question: string): string {
    return question
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase();
}

export function extractRequestedOrdinal(question: string): number | null {
    const normalized = normalizeQuestion(question);
    const ordinalPhrases: Array<[string, number]> = [
        ["decimo nono", 19],
        ["decimo oitavo", 18],
        ["decimo setimo", 17],
        ["decimo sexto", 16],
        ["decimo quinto", 15],
        ["decimo quarto", 14],
        ["decimo terceiro", 13],
        ["decimo segundo", 12],
        ["decimo primeiro", 11],
        ["vigesimo", 20],
        ["decimo", 10],
        ["nono", 9],
        ["oitavo", 8],
        ["setimo", 7],
        ["sexto", 6],
        ["quinto", 5],
        ["quarto", 4],
        ["terceiro", 3],
        ["segundo", 2],
        ["primeiro", 1],
    ];

    const match = ordinalPhrases.find(([phrase]) => normalized.includes(phrase));
    if (match) {
        return match[1];
    }

    const numericMatch = normalized.match(
        /\b(\d+)\s*(?:º|ª|th|st|nd|rd)?\s+(?:membro|member|integrante)\b/
    );
    return numericMatch ? Number(numericMatch[1]) : null;
}

export function isMemberDiscoveryQuestion(question: string): boolean {
    const normalized = normalizeQuestion(question);
    return (
        normalized.includes("membro") ||
        normalized.includes("member") ||
        normalized.includes("integrante") ||
        normalized.includes("quem e o") ||
        normalized.includes("who is the") ||
        normalized.includes("quem sao") ||
        normalized.includes("who are the") ||
        extractRequestedOrdinal(question) !== null
    );
}

export function isPersonCentricQuestion(question: string): boolean {
    const normalized = normalizeQuestion(question);
    return (
        normalized.includes("mensagens de ") ||
        normalized.includes("mensagens do ") ||
        normalized.includes("messages from ") ||
        normalized.includes("perfil de ") ||
        normalized.includes("perfil do ") ||
        normalized.includes("profile of ") ||
        normalized.includes("bio de ") ||
        normalized.includes("bio do ")
    );
}

export function isIdentityQuestion(question: string): boolean {
    const normalized = normalizeQuestion(question);
    return (
        normalized.includes("quem e ") ||
        normalized.includes("who is ") ||
        normalized.includes("que cargo") ||
        normalized.includes("qual cargo") ||
        normalized.includes("que papel") ||
        normalized.includes("qual papel") ||
        normalized.includes("qual o nick") ||
        normalized.includes("qual e o nick") ||
        normalized.includes("qual o username") ||
        normalized.includes("qual e o username") ||
        normalized.includes("perfil de ") ||
        normalized.includes("perfil do ") ||
        normalized.includes("profile of ")
    );
}

export function isPersonMessageQuestion(question: string): boolean {
    const normalized = normalizeQuestion(question);
    return (
        (normalized.includes("o que ") && normalized.includes(" falou")) ||
        (normalized.includes("o que ") && normalized.includes(" disse")) ||
        (normalized.includes("what did ") && normalized.includes(" say")) ||
        normalized.includes("mensagens de ") ||
        normalized.includes("mensagens do ") ||
        normalized.includes("messages from ") ||
        normalized.includes("falou sobre ") ||
        normalized.includes("disse sobre ")
    );
}

export function isGuildContextQuestion(question: string): boolean {
    const normalized = normalizeQuestion(question);

    if (!normalized.includes("servidor") && !normalized.includes("server")) {
        return false;
    }

    return (
        normalized.includes("que servidor") ||
        normalized.includes("qual servidor") ||
        normalized.includes("em que servidor") ||
        normalized.includes("neste servidor") ||
        normalized.includes("nesse servidor") ||
        normalized.includes("this server") ||
        normalized.includes("which server") ||
        normalized.includes("what server") ||
        normalized.includes("current server")
    );
}

export function extractLikelyPersonName(question: string): string | null {
    const match = question.match(
        /\b(?:mensagens de|mensagens do|perfil de|perfil do|bio de|bio do|from|profile of)\s+([^\n]+)$/i
    );
    return match ? match[1].trim().replace(/^#/, "") : null;
}

export function extractTopicHint(question: string): string | null {
    const match = question.match(
        /\b(?:sobre|about)\s+([^\n?!.,:;]+)$/i
    );
    if (match?.[1]) {
        return match[1].trim();
    }

    const saidMatch = question.match(
        /\b(?:falou|disse|said)\s+(?:sobre|about)\s+([^\n?!.,:;]+)$/i
    );
    return saidMatch?.[1]?.trim() || null;
}

export function isReferentialFollowUp(question: string): boolean {
    const normalized = normalizeQuestion(question);
    return (
        /\b(ele|ela|esse|essa|isso|aquele|aquela|tal)\b/.test(normalized) ||
        normalized.startsWith("e ") ||
        normalized.startsWith("hm ") ||
        normalized.startsWith("hmm ")
    );
}

export function extractMentionedChannelIds(question: string): string[] {
    return [...question.matchAll(/<#(\d+)>/g)].map((match) => match[1]);
}

export function extractMentionedUserIds(question: string): string[] {
    return [...question.matchAll(/<@!?(\d+)>/g)].map((match) => match[1]);
}

export function needsAiRouting(question: string): boolean {
    const normalized = normalizeQuestion(question);
    return (
        /\b(?:do|da|de|from)\s+\S+/i.test(question) ||
        normalized.includes("canal ") ||
        normalized.includes("channel ") ||
        normalized.includes("perfil ") ||
        normalized.includes("profile ") ||
        normalized.includes("bio ")
    );
}

export function requestsAllMembers(question: string): boolean {
    const normalized = normalizeQuestion(question);
    return (
        normalized.includes("todos os membros") ||
        normalized.includes("all members") ||
        normalized.includes("lista de membros") ||
        normalized.includes("member list")
    );
}
