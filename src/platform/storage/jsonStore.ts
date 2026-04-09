import fs from "fs";
import path from "path";
import { ensureDirectoryExists, getStorageFilePath } from "@/platform/storage/directories";

export function readJsonFile<T>(filePath: string): T | null {
    try {
        if (!fs.existsSync(filePath)) {
            return null;
        }

        const content = fs.readFileSync(filePath, "utf-8");
        return JSON.parse(content) as T;
    } catch (error) {
        console.error(`Error reading JSON file ${filePath}:`, error);
        return null;
    }
}

export function readJsonFromStoragePath<T>(
    fileName: string,
    ...pathSegments: string[]
): T | null {
    return readJsonFile<T>(getStorageFilePath(fileName, ...pathSegments));
}

export function writeJsonFile(filePath: string, data: unknown): boolean {
    try {
        ensureDirectoryExists(path.dirname(filePath));
        fs.writeFileSync(filePath, JSON.stringify(data, null, 2), "utf-8");
        return true;
    } catch (error) {
        console.error(`Error writing JSON file ${filePath}:`, error);
        return false;
    }
}

export function writeJsonToStoragePath(
    fileName: string,
    data: unknown,
    ...pathSegments: string[]
): boolean {
    return writeJsonFile(getStorageFilePath(fileName, ...pathSegments), data);
}
