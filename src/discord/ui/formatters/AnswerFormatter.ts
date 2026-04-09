import type { AnswerCitation } from "@/shared/appTypes";

export class AnswerFormatter {
    public static format(answer: string, _citations: AnswerCitation[] = []): string {
        return answer.trim();
    }
}
