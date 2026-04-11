import fs from "fs";
import path from "path";

export function discoverModuleFiles(root: string, suffix: string): string[] {
    const discovered: string[] = [];

    const visit = (currentDir: string) => {
        const entries = fs.readdirSync(currentDir, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = path.join(currentDir, entry.name);

            if (entry.isDirectory()) {
                visit(fullPath);
                continue;
            }

            if (
                entry.isFile() &&
                (entry.name.endsWith(`${suffix}.ts`) || entry.name.endsWith(`${suffix}.js`))
            ) {
                discovered.push(fullPath);
            }
        }
    };

    visit(root);
    return discovered.sort((left, right) => left.localeCompare(right));
}
