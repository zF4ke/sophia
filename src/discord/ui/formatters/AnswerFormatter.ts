import type { AnswerCitation } from "@/shared/appTypes";

export class AnswerFormatter {
    public static format(answer: string, citations: AnswerCitation[] = []): string {
        if (!citations.length) {
            return answer.trim();
        }

        const formattedCitations = citations
            .map((citation, index) => `${index + 1}. [${citation.label}](${citation.jumpLink})`)
            .join("\n");

        return `${answer.trim()}\n\nFontes:\n${formattedCitations}`;
    }
}
