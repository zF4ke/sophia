export class PersonalityService {
    public static determineResponseStyle(): "casual" | "technical" | "supportive" {
        return "casual";
    }

    public static adjustPromptForPersonality(basePrompt: string): string {
        return basePrompt;
    }

    public static removeEmojis(content: string): string {
        return content;
    }
}
