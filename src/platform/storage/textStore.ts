import fs from "fs";
import path from "path";
import { ensureDirectoryExists, getStorageFilePath } from "@/platform/storage/directories";

export function readTextFile(filePath: string): string | null {
    try {
        if (!fs.existsSync(filePath)) {
            return null;
        }

        return fs.readFileSync(filePath, "utf-8");
    } catch (error) {
        console.error(`Error reading text file ${filePath}:`, error);
        return null;
    }
}

export function readTextFromStoragePath(
    fileName: string,
    ...pathSegments: string[]
): string | null {
    return readTextFile(getStorageFilePath(fileName, ...pathSegments));
}

export function writeTextFile(filePath: string, content: string): boolean {
    try {
        ensureDirectoryExists(path.dirname(filePath));
        fs.writeFileSync(filePath, content, "utf-8");
        return true;
    } catch (error) {
        console.error(`Error writing text file ${filePath}:`, error);
        return false;
    }
}

export function writeTextToStoragePath(
    fileName: string,
    content: string,
    ...pathSegments: string[]
): boolean {
    return writeTextFile(getStorageFilePath(fileName, ...pathSegments), content);
}
