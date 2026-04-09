import fs from "fs";
import path from "path";
import { ensureDirectoryExists, getStorageFilePath } from "@/platform/storage/directories";

export function readBinaryFile(filePath: string): Buffer | null {
    try {
        if (!fs.existsSync(filePath)) {
            return null;
        }

        return fs.readFileSync(filePath);
    } catch (error) {
        console.error(`Error reading binary file ${filePath}:`, error);
        return null;
    }
}

export function readBinaryFromStoragePath(
    fileName: string,
    ...pathSegments: string[]
): Buffer | null {
    return readBinaryFile(getStorageFilePath(fileName, ...pathSegments));
}

export function writeBinaryFile(filePath: string, content: Buffer): boolean {
    try {
        ensureDirectoryExists(path.dirname(filePath));
        fs.writeFileSync(filePath, content);
        return true;
    } catch (error) {
        console.error(`Error writing binary file ${filePath}:`, error);
        return false;
    }
}

export function writeBinaryToStoragePath(
    fileName: string,
    content: Buffer,
    ...pathSegments: string[]
): boolean {
    return writeBinaryFile(getStorageFilePath(fileName, ...pathSegments), content);
}
