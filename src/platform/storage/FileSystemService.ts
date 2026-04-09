import * as fs from 'fs';
import * as path from 'path';
import { AppPaths } from "@/app/AppPaths";

/**
 * Service for handling file system operations throughout the application
 * Acts as a centralized interface for all file I/O operations
 */
export class FileSystemService {
    /**
     * Base directory for all mutable runtime storage
     */
    private static readonly BASE_STORAGE_DIR = AppPaths.storageRoot;

    /**
     * Ensures that a directory exists, creating it if necessary
     * @param dirPath Full path to the directory
     */
    public static ensureDirectoryExists(dirPath: string): void {
        if (!fs.existsSync(dirPath)) {
            fs.mkdirSync(dirPath, { recursive: true });
        }
    }

    /**
     * Gets the base data directory path
     * @returns Path to the base data directory
     */
    public static getBaseStorageDir(): string {
        return this.BASE_STORAGE_DIR;
    }

    /**
     * Gets the directory path within the data directory structure
     * Can handle arbitrarily deep paths
     * @param pathSegments Array of path segments to join
     * @returns Full path to the requested directory, which is created if it doesn't exist
     */
    public static getDir(...pathSegments: string[]): string {
        const dirPath = path.join(this.BASE_STORAGE_DIR, ...pathSegments);
        this.ensureDirectoryExists(dirPath);
        return dirPath;
    }

    /**
     * Gets the path for a file within the data directory structure
     * @param fileName Name of the file (should include extension)
     * @param pathSegments Array of path segments leading to the file location
     * @returns Full path to the file
     */
    public static getFilePath(fileName: string, ...pathSegments: string[]): string {
        const dirPath = this.getDir(...pathSegments);
        return path.join(dirPath, fileName);
    }

    /**
     * Reads a JSON file and parses its contents
     * @param filePath Full path to the JSON file
     * @returns Parsed JSON data or null if file doesn't exist or is invalid
     */
    public static readJsonFile<T>(filePath: string): T | null {
        try {
            if (!fs.existsSync(filePath)) {
                return null;
            }
            
            const content = fs.readFileSync(filePath, 'utf-8');
            return JSON.parse(content) as T;
        } catch (error) {
            console.error(`Error reading JSON file ${filePath}:`, error);
            return null;
        }
    }

    /**
     * Reads a JSON file from anywhere within the data directory structure
     * @param fileName Name of the JSON file
     * @param pathSegments Path segments leading to the file location 
     * @returns Parsed JSON data or null if file doesn't exist or is invalid
     */
    public static readJsonFromPath<T>(fileName: string, ...pathSegments: string[]): T | null {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.readJsonFile<T>(filePath);
    }

    /**
     * Writes data to a JSON file
     * @param filePath Full path to the JSON file
     * @param data Data to write
     * @returns True if write was successful, false otherwise
     */
    public static writeJsonFile(filePath: string, data: any): boolean {
        try {
            // Ensure the directory exists
            const dirPath = path.dirname(filePath);
            this.ensureDirectoryExists(dirPath);
            
            // Write the file
            fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf-8');
            return true;
        } catch (error) {
            console.error(`Error writing JSON file ${filePath}:`, error);
            return false;
        }
    }

    /**
     * Writes JSON data to a file within the data directory structure
     * @param fileName Name of the JSON file
     * @param data Data to write
     * @param pathSegments Path segments leading to the file location
     * @returns True if write was successful, false otherwise
     */
    public static writeJsonToPath(fileName: string, data: any, ...pathSegments: string[]): boolean {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.writeJsonFile(filePath, data);
    }

    /**
     * Lists all files in a directory
     * @param dirPath Directory to list files from
     * @param extension Optional file extension filter
     * @returns Array of file paths
     */
    public static listFiles(dirPath: string, extension?: string): string[] {
        try {
            if (!fs.existsSync(dirPath)) {
                return [];
            }
            
            const files = fs.readdirSync(dirPath);
            if (extension) {
                return files.filter(file => file.endsWith(extension));
            }
            return files;
        } catch (error) {
            console.error(`Error listing files in directory ${dirPath}:`, error);
            return [];
        }
    }

    /**
     * Lists all files in a directory within the data directory structure
     * @param pathSegments Path segments leading to the directory
     * @param extension Optional file extension filter
     * @returns Array of file names
     */
    public static listFilesInPath(extension?: string, ...pathSegments: string[]): string[] {
        const dirPath = this.getDir(...pathSegments);
        return this.listFiles(dirPath, extension);
    }

    /**
     * Deletes a file if it exists
     * @param filePath Full path to the file
     * @returns True if delete was successful or file didn't exist, false on error
     */
    public static deleteFile(filePath: string): boolean {
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

    /**
     * Deletes a file within the data directory structure
     * @param fileName Name of the file to delete
     * @param pathSegments Path segments leading to the file location
     * @returns True if delete was successful or file didn't exist, false on error
     */
    public static deleteFileFromPath(fileName: string, ...pathSegments: string[]): boolean {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.deleteFile(filePath);
    }

    /**
     * Clears all files in a directory within the data directory structure
     * @param pathSegments Path segments leading to the directory
     * @param extension Optional filter by file extension
     * @returns True if deletion was successful, false otherwise
     */
    public static clearDirectory(extension?: string, ...pathSegments: string[]): boolean {
        try {
            const dirPath = this.getDir(...pathSegments);
            
            if (fs.existsSync(dirPath)) {
                const files = this.listFiles(dirPath, extension);
                
                for (const file of files) {
                    const filePath = path.join(dirPath, file);
                    this.deleteFile(filePath);
                }
            }
            return true;
        } catch (error) {
            console.error(`Error clearing directory ${pathSegments.join('/')}:`, error);
            return false;
        }
    }

    /**
     * Reads raw file contents as a string (not parsed as JSON)
     * @param filePath Path to the file
     * @returns File contents as string or null if file doesn't exist
     */
    public static readTextFile(filePath: string): string | null {
        try {
            if (!fs.existsSync(filePath)) {
                return null;
            }
            
            return fs.readFileSync(filePath, 'utf-8');
        } catch (error) {
            console.error(`Error reading text file ${filePath}:`, error);
            return null;
        }
    }

    /**
     * Reads raw text file contents from anywhere within the data directory structure
     * @param fileName Name of the text file
     * @param pathSegments Path segments leading to the file location
     * @returns File contents as string or null if file doesn't exist
     */
    public static readTextFromPath(fileName: string, ...pathSegments: string[]): string | null {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.readTextFile(filePath);
    }

    /**
     * Writes raw text content to a file
     * @param filePath Path to the file
     * @param content Text content to write
     * @returns True if write was successful, false otherwise
     */
    public static writeTextFile(filePath: string, content: string): boolean {
        try {
            // Ensure the directory exists
            const dirPath = path.dirname(filePath);
            this.ensureDirectoryExists(dirPath);
            
            // Write the file
            fs.writeFileSync(filePath, content, 'utf-8');
            return true;
        } catch (error) {
            console.error(`Error writing text file ${filePath}:`, error);
            return false;
        }
    }

    /**
     * Writes raw text content to a file within the data directory structure
     * @param fileName Name of the text file
     * @param content Text content to write
     * @param pathSegments Path segments leading to the file location
     * @returns True if write was successful, false otherwise
     */
    public static writeTextToPath(fileName: string, content: string, ...pathSegments: string[]): boolean {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.writeTextFile(filePath, content);
    }

    /**
     * Reads binary file contents as a Buffer
     * @param filePath Path to the file
     * @returns File contents as Buffer or null if file doesn't exist
     */
    public static readBinaryFile(filePath: string): Buffer | null {
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

    /**
     * Reads binary file contents from anywhere within the data directory structure
     * @param fileName Name of the binary file
     * @param pathSegments Path segments leading to the file location
     * @returns File contents as Buffer or null if file doesn't exist
     */
    public static readBinaryFromPath(fileName: string, ...pathSegments: string[]): Buffer | null {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.readBinaryFile(filePath);
    }

    /**
     * Writes binary content to a file
     * @param filePath Path to the file
     * @param content Binary content to write
     * @returns True if write was successful, false otherwise
     */
    public static writeBinaryFile(filePath: string, content: Buffer): boolean {
        try {
            // Ensure the directory exists
            const dirPath = path.dirname(filePath);
            this.ensureDirectoryExists(dirPath);
            
            // Write the file
            fs.writeFileSync(filePath, content);
            return true;
        } catch (error) {
            console.error(`Error writing binary file ${filePath}:`, error);
            return false;
        }
    }

    /**
     * Writes binary content to a file within the data directory structure
     * @param fileName Name of the binary file
     * @param content Binary content to write
     * @param pathSegments Path segments leading to the file location
     * @returns True if write was successful, false otherwise
     */
    public static writeBinaryToPath(fileName: string, content: Buffer, ...pathSegments: string[]): boolean {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.writeBinaryFile(filePath, content);
    }

    /**
     * Checks if a file exists
     * @param filePath Path to the file
     * @returns True if file exists, false otherwise
     */
    public static fileExists(filePath: string): boolean {
        return fs.existsSync(filePath);
    }

    /**
     * Checks if a file exists within the data directory structure
     * @param fileName Name of the file
     * @param pathSegments Path segments leading to the file location
     * @returns True if file exists, false otherwise
     */
    public static fileExistsInPath(fileName: string, ...pathSegments: string[]): boolean {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.fileExists(filePath);
    }

    /**
     * Gets file stats (size, modification date, etc.)
     * @param filePath Path to the file
     * @returns File stats or null if file doesn't exist
     */
    public static getFileStats(filePath: string): fs.Stats | null {
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

    /**
     * Gets file stats for a file within the data directory structure
     * @param fileName Name of the file
     * @param pathSegments Path segments leading to the file location
     * @returns File stats or null if file doesn't exist
     */
    public static getFileStatsFromPath(fileName: string, ...pathSegments: string[]): fs.Stats | null {
        const filePath = this.getFilePath(fileName, ...pathSegments);
        return this.getFileStats(filePath);
    }

    /**
     * Calculates the total size of a directory in bytes
     * @param directoryPath Path to the directory
     * @returns Total size in bytes
     */
    public static getDirectorySize(directoryPath: string): number {
        let totalSize = 0;
        
        if (!fs.existsSync(directoryPath)) {
            return 0;
        }

        const files = fs.readdirSync(directoryPath);
        
        for (const file of files) {
            const filePath = path.join(directoryPath, file);
            const stats = fs.statSync(filePath);
            
            if (stats.isFile()) {
                totalSize += stats.size;
            } else if (stats.isDirectory()) {
                totalSize += this.getDirectorySize(filePath);
            }
        }
        
        return totalSize;
    }

    /**
     * Gets detailed information about files in a directory
     * @param directoryPath Path to the directory
     * @returns Array of objects with file information
     */
    public static getDirectoryFiles(directoryPath: string): Array<{ path: string, size: number, created: number }> {
        const files: Array<{ path: string, size: number, created: number }> = [];
        
        if (!fs.existsSync(directoryPath)) {
            return files;
        }

        const items = fs.readdirSync(directoryPath);
        
        for (const item of items) {
            const fullPath = path.join(directoryPath, item);
            const stats = fs.statSync(fullPath);
            
            if (stats.isFile()) {
                files.push({
                    path: fullPath,
                    size: stats.size,
                    created: stats.birthtime.getTime()
                });
            }
        }
        
        return files;
    }
}
