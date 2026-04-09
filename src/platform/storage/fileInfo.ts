import fs from "fs";
import path from "path";
import { getStorageDir, getStorageFilePath } from "@/platform/storage/directories";

export function listFiles(dirPath: string, extension?: string): string[] {
    try {
        if (!fs.existsSync(dirPath)) {
            return [];
        }

        const files = fs.readdirSync(dirPath);
        return extension ? files.filter((file) => file.endsWith(extension)) : files;
    } catch (error) {
        console.error(`Error listing files in directory ${dirPath}:`, error);
        return [];
    }
}

export function listFilesInStoragePath(
    extension?: string,
    ...pathSegments: string[]
): string[] {
    return listFiles(getStorageDir(...pathSegments), extension);
}

export function deleteFile(filePath: string): boolean {
    try {
        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
        }
        return true;
    } catch (error) {
        console.error(`Error deleting file ${filePath}:`, error);
        return false;
    }
}

export function deleteFileFromStoragePath(
    fileName: string,
    ...pathSegments: string[]
): boolean {
    return deleteFile(getStorageFilePath(fileName, ...pathSegments));
}

export function clearStorageDirectory(
    extension?: string,
    ...pathSegments: string[]
): boolean {
    try {
        const dirPath = getStorageDir(...pathSegments);
        const files = listFiles(dirPath, extension);

        for (const file of files) {
            deleteFile(path.join(dirPath, file));
        }

        return true;
    } catch (error) {
        console.error(`Error clearing directory ${pathSegments.join("/")}:`, error);
        return false;
    }
}

export function fileExists(filePath: string): boolean {
    return fs.existsSync(filePath);
}

export function fileExistsInStoragePath(
    fileName: string,
    ...pathSegments: string[]
): boolean {
    return fileExists(getStorageFilePath(fileName, ...pathSegments));
}

export function getFileStats(filePath: string): fs.Stats | null {
    try {
        if (!fs.existsSync(filePath)) {
            return null;
        }

        return fs.statSync(filePath);
    } catch (error) {
        console.error(`Error getting file stats ${filePath}:`, error);
        return null;
    }
}

export function getFileStatsFromStoragePath(
    fileName: string,
    ...pathSegments: string[]
): fs.Stats | null {
    return getFileStats(getStorageFilePath(fileName, ...pathSegments));
}

export function getDirectorySize(directoryPath: string): number {
    if (!fs.existsSync(directoryPath)) {
        return 0;
    }

    let totalSize = 0;
    const files = fs.readdirSync(directoryPath);

    for (const file of files) {
        const filePath = path.join(directoryPath, file);
        const stats = fs.statSync(filePath);

        if (stats.isFile()) {
            totalSize += stats.size;
            continue;
        }

        if (stats.isDirectory()) {
            totalSize += getDirectorySize(filePath);
        }
    }

    return totalSize;
}

export function getDirectoryFiles(
    directoryPath: string
): Array<{ path: string; size: number; created: number }> {
    if (!fs.existsSync(directoryPath)) {
        return [];
    }

    return fs.readdirSync(directoryPath).flatMap((item) => {
        const fullPath = path.join(directoryPath, item);
        const stats = fs.statSync(fullPath);
        if (!stats.isFile()) {
            return [];
        }

        return [
            {
                path: fullPath,
                size: stats.size,
                created: stats.birthtime.getTime(),
            },
        ];
    });
}
