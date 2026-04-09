import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";

type PromptParams = Record<string, string | number | boolean | undefined | null>;

export class PromptRegistry {
    private static readonly PROMPTS_ROOT = AppPaths.promptsRoot;
    private static readonly cache = new Map<string, string>();

    public static load(promptId: string): string {
        const normalizedId = promptId.replace(/\\/g, "/");
        const cached = this.cache.get(normalizedId);
        if (cached) {
            return cached;
        }

        const fullPath = path.join(this.PROMPTS_ROOT, `${normalizedId}.md`);
        const content = fs.readFileSync(fullPath, "utf8").trim();
        this.cache.set(normalizedId, content);
        return content;
    }

    public static render(promptId: string, params: PromptParams = {}): string {
        let template = this.load(promptId);

        for (const [key, value] of Object.entries(params)) {
            const replacement = value == null ? "" : String(value);
            template = template.split(`{{${key}}}`).join(replacement);
        }

        return template;
    }
}
