import type fs from "fs";
import {
    ensureDirectoryExists,
    getStorageDir,
    getStorageFilePath,
    storageRoot,
} from "@/platform/storage/directories";
import {
    readJsonFile,
    readJsonFromStoragePath,
    writeJsonFile,
    writeJsonToStoragePath,
} from "@/platform/storage/jsonStore";
import {
    readTextFile,
    readTextFromStoragePath,
    writeTextFile,
    writeTextToStoragePath,
} from "@/platform/storage/textStore";
import {
    readBinaryFile,
    readBinaryFromStoragePath,
    writeBinaryFile,
    writeBinaryToStoragePath,
} from "@/platform/storage/binaryStore";
import {
    clearStorageDirectory,
    deleteFile,
    deleteFileFromStoragePath,
    fileExists,
    fileExistsInStoragePath,
    getDirectoryFiles,
    getDirectorySize,
    getFileStats,
    getFileStatsFromStoragePath,
    listFiles,
    listFilesInStoragePath,
} from "@/platform/storage/fileInfo";

/**
 * Service for handling file system operations throughout the application
 * Acts as a centralized interface for all file I/O operations
 */
export class FileSystemService {
    public static ensureDirectoryExists(dirPath: string): void {
        ensureDirectoryExists(dirPath);
    }

    public static getBaseStorageDir(): string {
        return storageRoot;
    }

    public static getDir(...pathSegments: string[]): string {
        return getStorageDir(...pathSegments);
    }

    public static getFilePath(fileName: string, ...pathSegments: string[]): string {
        return getStorageFilePath(fileName, ...pathSegments);
    }

    public static readJsonFile<T>(filePath: string): T | null {
        return readJsonFile<T>(filePath);
    }

    public static readJsonFromPath<T>(fileName: string, ...pathSegments: string[]): T | null {
        return readJsonFromStoragePath<T>(fileName, ...pathSegments);
    }

    public static writeJsonFile(filePath: string, data: unknown): boolean {
        return writeJsonFile(filePath, data);
    }

    public static writeJsonToPath(
        fileName: string,
        data: unknown,
        ...pathSegments: string[]
    ): boolean {
        return writeJsonToStoragePath(fileName, data, ...pathSegments);
    }

    public static listFiles(dirPath: string, extension?: string): string[] {
        return listFiles(dirPath, extension);
    }

    public static listFilesInPath(extension?: string, ...pathSegments: string[]): string[] {
        return listFilesInStoragePath(extension, ...pathSegments);
    }

    public static deleteFile(filePath: string): boolean {
        return deleteFile(filePath);
    }

    public static deleteFileFromPath(fileName: string, ...pathSegments: string[]): boolean {
        return deleteFileFromStoragePath(fileName, ...pathSegments);
    }

    public static clearDirectory(extension?: string, ...pathSegments: string[]): boolean {
        return clearStorageDirectory(extension, ...pathSegments);
    }

    public static readTextFile(filePath: string): string | null {
        return readTextFile(filePath);
    }

    public static readTextFromPath(fileName: string, ...pathSegments: string[]): string | null {
        return readTextFromStoragePath(fileName, ...pathSegments);
    }

    public static writeTextFile(filePath: string, content: string): boolean {
        return writeTextFile(filePath, content);
    }

    public static writeTextToPath(
        fileName: string,
        content: string,
        ...pathSegments: string[]
    ): boolean {
        return writeTextToStoragePath(fileName, content, ...pathSegments);
    }

    public static readBinaryFile(filePath: string): Buffer | null {
        return readBinaryFile(filePath);
    }

    public static readBinaryFromPath(fileName: string, ...pathSegments: string[]): Buffer | null {
        return readBinaryFromStoragePath(fileName, ...pathSegments);
    }

    public static writeBinaryFile(filePath: string, content: Buffer): boolean {
        return writeBinaryFile(filePath, content);
    }

    public static writeBinaryToPath(
        fileName: string,
        content: Buffer,
        ...pathSegments: string[]
    ): boolean {
        return writeBinaryToStoragePath(fileName, content, ...pathSegments);
    }

    public static fileExists(filePath: string): boolean {
        return fileExists(filePath);
    }

    public static fileExistsInPath(fileName: string, ...pathSegments: string[]): boolean {
        return fileExistsInStoragePath(fileName, ...pathSegments);
    }

    public static getFileStats(filePath: string): fs.Stats | null {
        return getFileStats(filePath);
    }

    public static getFileStatsFromPath(fileName: string, ...pathSegments: string[]): fs.Stats | null {
        return getFileStatsFromStoragePath(fileName, ...pathSegments);
    }

    public static getDirectorySize(directoryPath: string): number {
        return getDirectorySize(directoryPath);
    }

    public static getDirectoryFiles(
        directoryPath: string
    ): Array<{ path: string; size: number; created: number }> {
        return getDirectoryFiles(directoryPath);
    }
}
