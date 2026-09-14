import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";

type PromptParams = Record<string, string | number | boolean | undefined | null>;

export class PromptRegistry {
    private static readonly cache = new Map<string, string>();

    public static load(promptId: string): string {
        const normalizedId = promptId.replace(/\\/g, "/");
        const cached = this.cache.get(normalizedId);
        if (cached) {
            return cached;
        }

        const fullPath = path.join(AppPaths.promptsRoot, `${normalizedId}.md`);
        const content = fs.readFileSync(fullPath, "utf8").trim();
        this.cache.set(normalizedId, content);
        return content;
    }

    public static render(promptId: string, params: PromptParams = {}): string {
        return this.load(promptId).replace(/\{\{([a-zA-Z0-9_]+)\}\}/g, (match, key: string) =>
            Object.prototype.hasOwnProperty.call(params, key) ? String(params[key] ?? "") : match);
    }
}
