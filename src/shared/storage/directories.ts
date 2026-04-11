import fs from "fs";
import path from "path";
import { AppPaths } from "@/app/AppPaths";

export const storageRoot = AppPaths.storageRoot;

export function ensureDirectoryExists(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

export function getStorageDir(...pathSegments: string[]): string {
    const dirPath = path.join(storageRoot, ...pathSegments);
    ensureDirectoryExists(dirPath);
    return dirPath;
}

export function getStorageFilePath(
    fileName: string,
    ...pathSegments: string[]
): string {
    return path.join(getStorageDir(...pathSegments), fileName);
}
